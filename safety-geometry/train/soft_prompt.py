"""
Soft-prompt safety alignment for LLaVA.

Learns a small set of continuous prefix tokens that steer image hidden states
toward the text-derived safety direction, bridging the cross-modal gap
identified by the geometry analysis (cos ≈ 0.06, principal angle ≈ 81°).

Architecture:
  Frozen LLaVA
  + learnable soft-prompt tokens prepended to the input
  + optional MLP adapter at target layers (13-16)
  + alignment loss at Layer 15 (best cross-modal transfer layer)

Training signal:
  L = L_lm + λ_align * L_align + λ_sep * L_sep
  where:
    L_lm    = standard language modelling loss (preserves capability)
    L_align = cosine distance between image safety dir and text safety dir
    L_sep   = hinge loss pushing safe images to + side, harmful to - side
              of the text safety direction

Usage:
    # Step 1: extract reference directions from cached analysis
    python -m train.soft_prompt --extract-ref \\
        --text-cache outputs/cache --image-cache outputs/cache

    # Step 2: train
    python -m train.soft_prompt \\
        --coco-root /path/to/COCO2014/val2014 \\
        --hod-root data/HOD \\
        --epochs 50 --lr 5e-4

    # Step 3: evaluate
    python -m train.soft_prompt --eval --ckpt outputs/soft_prompt/best.pt
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# =========================================================================== #
# Reference direction extraction (from cached .npy)
# =========================================================================== #

def extract_reference_directions(
    text_cache: str,
    image_cache: str,
    target_layers: list[int] = [13, 14, 15, 16],
    output_path: str = "outputs/soft_prompt/ref_directions.npz",
):
    """
    Extract DiM (difference-in-means) safety directions from cached hidden states.
    These become frozen supervision targets during training.
    """
    text_cache = Path(text_cache)
    image_cache = Path(image_cache)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    text_labels = np.load(text_cache / "text_labels.npy")
    image_labels = np.load(image_cache / "image_labels.npy")

    directions = {}
    for layer_idx in target_layers:
        # Text direction
        text_states = np.load(text_cache / f"text_layer{layer_idx}.npy")
        text_safe = text_states[text_labels == 1].mean(axis=0)
        text_harm = text_states[text_labels == 0].mean(axis=0)
        d_text = text_safe - text_harm
        d_text = d_text / (np.linalg.norm(d_text) + 1e-12)

        # Image direction (for monitoring)
        image_states = np.load(image_cache / f"image_layer{layer_idx}.npy")
        image_safe = image_states[image_labels == 1].mean(axis=0)
        image_harm = image_states[image_labels == 0].mean(axis=0)
        d_image = image_safe - image_harm
        d_image = d_image / (np.linalg.norm(d_image) + 1e-12)

        directions[f"d_text_layer{layer_idx}"] = d_text
        directions[f"d_image_layer{layer_idx}"] = d_image

        cos = float(np.dot(d_text, d_image))
        print(f"  Layer {layer_idx}: cos(d_text, d_image) = {cos:+.4f}")

    np.savez(output_path, **directions)
    print(f"\nSaved reference directions -> {output_path}")
    return directions


# =========================================================================== #
# Soft prompt module
# =========================================================================== #

class SoftPromptWrapper(nn.Module):
    """
    Wraps a frozen LLaVA model with learnable soft-prompt tokens
    and optional per-layer MLP adapters.

    The soft prompt is prepended to the input embeddings:
      [P1 P2 ... Pm] [image_tokens] [text_tokens]

    The MLP adapter (if enabled) transforms hidden states at target layers:
      h' = h + adapter(h)

    Supports llava-1.5-7b-hf, llava-v1.6 (LLaVA-NeXT), and other
    LlavaForConditionalGeneration variants from transformers ≥4.x/5.x.
    """

    def __init__(
        self,
        model: nn.Module,
        processor,
        n_soft_tokens: int = 16,
        target_layers: list[int] = [13, 14, 15, 16],
        primary_layer: int = 15,
        use_adapter: bool = False,
        adapter_hidden: int = 256,
        d_model: Optional[int] = None,
    ):
        super().__init__()
        self.model = model
        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self.target_layers = target_layers
        self.primary_layer = primary_layer

        # Freeze entire model
        for param in model.parameters():
            param.requires_grad = False

        # Detect architecture (sets self._lang_backbone, self._lang_causal,
        # self._embed_fn, self._has_vision) and auto-infer d_model
        self._detect_architecture()
        self._lang_backbone.gradient_checkpointing_enable()
        if d_model is None:
            d_model = self._infer_d_model()
        self.d_model = d_model

        # Determine device from frozen model parameters (works with device_map)
        _model_device = self._model_device()

        # Learnable soft-prompt embeddings, created on model device
        self.n_soft_tokens = n_soft_tokens
        self.soft_prompt = nn.Parameter(
            torch.randn(n_soft_tokens, d_model, device=_model_device) * 0.001
        )

        # Optional MLP adapter at each target layer
        self.use_adapter = use_adapter
        if use_adapter:
            self.adapters = nn.ModuleDict({
                str(layer_idx): nn.Sequential(
                    nn.Linear(d_model, adapter_hidden),
                    nn.GELU(),
                    nn.Linear(adapter_hidden, d_model),
                    nn.Dropout(0.1),
                )
                for layer_idx in target_layers
            }).to(_model_device)
            # Init near-zero so adapter starts as identity
            for adapter in self.adapters.values():
                nn.init.zeros_(adapter[-2].weight)
                nn.init.zeros_(adapter[-2].bias)

        # Hooks for capturing hidden states
        self._hidden_states: dict[int, torch.Tensor] = {}
        self._hooks = []

    # ------------------------------------------------------------------
    # Architecture detection helpers
    # ------------------------------------------------------------------

    def _detect_architecture(self):
        """
        Detect LLaVA model architecture and cache handles for language
        backbone, causal LM, embedding function, and vision components.

        Covers:
          - LlavaForConditionalGeneration (llava-1.5-7b-hf)
          - LlavaNextForConditionalGeneration (llava-v1.6-*)
          - LlavaModel (newer transformers: model.model.language_model.model.layers)
          - Any model with .language_model or .model + .layers
        """
        model = self.model

        # --- Language model backbone (the bare transformer) ---
        self._lang_causal = None
        self._lang_backbone = None

        # Strategy 1: model.language_model.model.layers
        if hasattr(model, 'language_model'):
            lang = model.language_model
            if hasattr(lang, 'model') and hasattr(lang.model, 'layers'):
                self._lang_backbone = lang.model
                self._lang_causal = lang

        # Strategy 2: model.language_model.layers (lang IS backbone)
        if self._lang_causal is None and hasattr(model, 'language_model'):
            lang = model.language_model
            if hasattr(lang, 'layers'):
                self._lang_backbone = lang
                self._lang_causal = lang

        # Strategy 3: model.model.language_model.layers
        # (newer transformers: LlavaForConditionalGeneration → LlavaModel → LlamaModel)
        if self._lang_causal is None and hasattr(model, 'model'):
            if hasattr(model.model, 'language_model'):
                lang = model.model.language_model
                if hasattr(lang, 'layers'):
                    self._lang_backbone = lang
                    self._lang_causal = model  # top-level model handles forward with labels

        # Strategy 4: model.model.layers
        if self._lang_causal is None and hasattr(model, 'model'):
            if hasattr(model.model, 'layers'):
                self._lang_backbone = model.model
                self._lang_causal = model

        # Strategy 5: Recursive search for .layers in nested .model attributes
        if self._lang_causal is None:
            def find_layers_recursive(obj, depth=0):
                """Find a module with .layers attribute."""
                if depth > 5:  # Limit recursion depth
                    return None
                if hasattr(obj, 'layers'):
                    return obj
                if hasattr(obj, 'model'):
                    return find_layers_recursive(obj.model, depth + 1)
                return None

            backbone = find_layers_recursive(model)
            if backbone is not None:
                self._lang_backbone = backbone
                # For causal forward pass, prefer language_model if it exists
                self._lang_causal = (
                    model.language_model if hasattr(model, 'language_model')
                    else model
                )

        # If still not found, error with helpful debug info
        if self._lang_causal is None or self._lang_backbone is None:
            attrs = []
            if hasattr(model, 'language_model'):
                lang = model.language_model
                attrs.append(f"model.language_model: {type(lang).__name__}")
                if hasattr(lang, 'model'):
                    attrs.append(f"  .model: {type(lang.model).__name__}")
                    if hasattr(lang.model, 'layers'):
                        attrs.append(f"    .layers: ✓")
                if hasattr(lang, 'layers'):
                    attrs.append(f"  .layers: ✓")
            if hasattr(model, 'model'):
                attrs.append(f"model.model: {type(model.model).__name__}")
                if hasattr(model.model, 'language_model'):
                    lang = model.model.language_model
                    attrs.append(f"  .language_model: {type(lang).__name__}")
                    if hasattr(lang, 'model') and hasattr(lang.model, 'layers'):
                        attrs.append(f"    .model.layers: ✓")
                if hasattr(model.model, 'layers'):
                    attrs.append(f"  .layers: ✓")

            debug_info = "\n  ".join(attrs) if attrs else "(no relevant attributes found)"
            raise ValueError(
                f"Cannot detect language backbone in {type(model).__name__}.\n"
                f"Expected .language_model or .model with .layers.\n"
                f"Found:\n  {debug_info}"
            )

        # --- Embedding function ---
        if hasattr(self._lang_backbone, 'embed_tokens'):
            self._embed_fn = self._lang_backbone.embed_tokens
        else:
            # Fallback: use get_input_embeddings() which all PreTrainedModel expose
            self._embed_fn = self._lang_causal.get_input_embeddings()

        # --- Vision components ---
        # In newer transformers, vision_tower may be nested under model.model
        def _find_attr(obj, name):
            if hasattr(obj, name):
                return getattr(obj, name)
            if hasattr(obj, 'model') and hasattr(obj.model, name):
                return getattr(obj.model, name)
            return None
 
        self._vision_tower = _find_attr(model, 'vision_tower')
        self._mm_projector = _find_attr(model, 'multi_modal_projector')
        self._has_vision = self._vision_tower is not None and self._mm_projector is not None

        arch = type(model).__name__
        lang_arch = type(self._lang_causal).__name__
        backbone_arch = type(self._lang_backbone).__name__
        n_layers = len(self._lang_backbone.layers)
        print(
            f"[SoftPromptWrapper] {arch} | lang: {lang_arch} | "
            f"backbone: {backbone_arch} | layers: {n_layers} | vision: {self._has_vision}"
        )

    def _infer_d_model(self) -> int:
        """Infer hidden dimension from model config or first layer weight."""
        cfg = getattr(self.model, 'config', None)
        # LLaVA stores text config nested under 'text_config'
        text_cfg = getattr(cfg, 'text_config', cfg)
        if text_cfg is not None and hasattr(text_cfg, 'hidden_size'):
            return text_cfg.hidden_size
        # Fallback: read from first layer's q_proj
        try:
            return self._lang_backbone.layers[0].self_attn.q_proj.in_features
        except Exception:
            return 4096  # safe default for 7B models

    def _model_device(self) -> torch.device:
        """Return device of the first non-meta model parameter."""
        for p in self.model.parameters():
            if p.device.type != 'meta':
                return p.device
        return torch.device('cpu')

    @property
    def device(self) -> torch.device:
        """Device of the learnable soft prompt (always equals model device)."""
        return self.soft_prompt.device

    # ------------------------------------------------------------------

    def _register_hooks(self):
        """Register forward hooks on target layers to capture hidden states."""
        self._remove_hooks()
        for layer_idx in self.target_layers:
            layer = self._lang_backbone.layers[layer_idx]
            hook = layer.register_forward_hook(
                self._make_hook(layer_idx)
            )
            self._hooks.append(hook)

    def _remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            # output is a tuple: (hidden_states, ...) for LLaMA layers
            hs = output[0] if isinstance(output, tuple) else output
            # Apply adapter if enabled
            if self.use_adapter and str(layer_idx) in self.adapters:
                residual = self.adapters[str(layer_idx)](hs)
                hs = hs + residual
                # Replace output
                if isinstance(output, tuple):
                    output = (hs,) + output[1:]
                else:
                    output = hs
            self._hidden_states[layer_idx] = hs
            return output
        return hook_fn

    # ------------------------------------------------------------------
    # Vision + text embedding preparation
    # ------------------------------------------------------------------

    def _get_merged_embeddings(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
    ):
        """
        Return (inputs_embeds, attention_mask, labels) with vision features
        merged into the text embedding sequence at <image> token positions.

        For text-only inputs (pixel_values=None) this is just embed_tokens(input_ids).
        For vision inputs the image features replace the <image> placeholder token(s),
        expanding the sequence length accordingly (handled by the model's internal
        _merge_input_ids_with_image_features).
        """
        device = self.device

        if not self._has_vision or pixel_values is None:
            return self._embed_fn(input_ids.to(device)), attention_mask, labels

        # Vision dtype may differ from text dtype (e.g. fp16 vs fp32)
        vision_dtype = next(self._vision_tower.parameters()).dtype
        with torch.no_grad():
            pv = pixel_values.to(device=device, dtype=vision_dtype)
 
            # Run vision tower
            vision_feature_layer = getattr(self.model.config, 'vision_feature_layer', -2)
            image_outputs = self._vision_tower(pv, output_hidden_states=True)
        selected = image_outputs.hidden_states[vision_feature_layer]
 
        # vision_feature_select_strategy: "default" keeps all patches,
        # "full" keeps CLS + patches. Handle both.
        strategy = getattr(self.model.config, 'vision_feature_select_strategy', 'default')
        if strategy == 'full':
            pass  # keep as-is (includes CLS token)
        else:
            selected = selected[:, 1:, :]  # drop CLS token
 
        image_features = self._mm_projector(selected)  # (B, n_patches, D)
 
        inputs_embeds = self._embed_fn(input_ids.to(device))
 
        # Use model's internal merge if available (handles sequence expansion)
        _merge_fn = getattr(self.model, '_merge_input_ids_with_image_features', None) or \
                    getattr(getattr(self.model, 'model', None), '_merge_input_ids_with_image_features', None)
        if _merge_fn is not None:
            try:
                result = _merge_fn(
                    image_features,
                    inputs_embeds,
                    input_ids.to(device),
                    attention_mask.to(device) if attention_mask is not None else None,
                    labels.to(device) if labels is not None else None,
                )
                # Normalize different return arities across transformers versions
                merged_embeds   = result[0].to(device)
                merged_mask     = result[1].to(device) if result[1] is not None else attention_mask
                merged_labels   = result[2].to(device) if (len(result) > 2 and result[2] is not None) else labels
                return merged_embeds, merged_mask, merged_labels
            except Exception as e:
                print(f"[Warning] _merge_input_ids_with_image_features failed ({e}); "
                      f"falling back to manual merge.")

        # Manual fallback: expand <image> token into all n_patches image features
        image_token_index = getattr(self.model.config, 'image_token_index', 32000)
        image_features = image_features.to(inputs_embeds.dtype)
        n_patches = image_features.shape[1]  # e.g. 576 for CLIP ViT-L/14@336

        new_embeds_list = []
        new_mask_list = []
        new_labels_list = []

        for b in range(inputs_embeds.size(0)):
            positions = (input_ids[b] == image_token_index).nonzero(as_tuple=True)[0]
            if len(positions) == 0:
                # No image token: keep as-is
                new_embeds_list.append(inputs_embeds[b])
                if attention_mask is not None:
                    new_mask_list.append(attention_mask[b])
                if labels is not None:
                    new_labels_list.append(labels[b])
                continue

            # Split at first <image> token position and insert all patch features
            pos = positions[0].item()
            before = inputs_embeds[b, :pos, :]       # (pos, D)
            after  = inputs_embeds[b, pos+1:, :]      # (T-pos-1, D)
            img    = image_features[b]                 # (n_patches, D)
            new_embeds_list.append(torch.cat([before, img, after], dim=0))

            if attention_mask is not None:
                m_before = attention_mask[b, :pos]
                m_after  = attention_mask[b, pos+1:]
                m_img    = torch.ones(n_patches, device=device, dtype=attention_mask.dtype)
                new_mask_list.append(torch.cat([m_before, m_img, m_after]))

            if labels is not None:
                l_before = labels[b, :pos]
                l_after  = labels[b, pos+1:]
                l_img    = torch.full((n_patches,), -100, device=device, dtype=labels.dtype)
                new_labels_list.append(torch.cat([l_before, l_img, l_after]))

        # Stack batch (all should have same length: T - 1 + n_patches)
        # Pad to same length
        max_len = max(e.shape[0] for e in new_embeds_list)
        inputs_embeds = torch.stack([
            F.pad(e, (0, 0, 0, max_len - e.shape[0]))
            for e in new_embeds_list
        ], dim=0)

        if new_mask_list:
            attention_mask = torch.stack([
                F.pad(m, (0, max_len - m.shape[0]), value=0)
                for m in new_mask_list
            ], dim=0)
        if new_labels_list:
            labels = torch.stack([
                F.pad(m, (0, max_len - m.shape[0]), value=0)
                for m in new_mask_list
            ], dim=0)

        return inputs_embeds, attention_mask, labels

    # ------------------------------------------------------------------

    def forward(
        self,
        images: Optional[list] = None,
        text_prompts: Optional[list[str]] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        compute_lm_loss: bool = False,
    ):
        """
        Forward pass with soft prompt prepended.

        Args:
            compute_lm_loss: If True, construct auto-regressive labels from
                input_ids so that the language model returns a next-token
                prediction loss (used as a capability-preservation regulariser).

        Returns:
            outputs: language model output (logits, loss, …)
            hidden_states: dict of {layer_idx: (B, T, D)} at target layers
        """
        device = self.device
        self._hidden_states = {}
        self._register_hooks()

        try:
            # ---- Prepare inputs ----
            if images is not None and text_prompts is not None:
                from extract.hidden_states import LLAVA_PROMPT_TEMPLATE
                formatted = [LLAVA_PROMPT_TEMPLATE.format(question=t) for t in text_prompts]
                inputs = self.processor(
                    text=formatted,
                    images=images,
                    return_tensors="pt",
                    padding=True,
                )
                input_ids      = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)
                pixel_values   = inputs.get("pixel_values")
            elif input_ids is None:
                raise ValueError("Provide either (images, text_prompts) or input_ids")
            else:
                input_ids = input_ids.to(device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                if labels is not None:
                    labels = labels.to(device)
                if pixel_values is not None:
                    pixel_values = pixel_values.to(device)

            B = input_ids.size(0)

            # ---- Merge vision features into text embeddings ----
            inputs_embeds, attention_mask, lm_labels = self._get_merged_embeddings(
                input_ids, pixel_values, attention_mask, labels
            )
            # inputs_embeds: (B, T', D)  T' may be > T if image tokens were expanded

            # ---- Build auto-regressive LM labels if requested ----
            if compute_lm_loss:
                # Standard causal LM: predict next token, label = input shifted left
                # Use the merged input_ids (after image expansion) to build labels.
                # Since we work with inputs_embeds (not input_ids) at this point,
                # we reconstruct labels from the embedding indices via nearest-neighbour,
                # BUT a much simpler approach: use the original input_ids before merge.
                # The merged sequence has image patches inserted, so we mark those as -100.
                T_merged = inputs_embeds.shape[1]
                # Build labels: shift input_ids left by 1 for next-token prediction
                # For positions we can't map back to token IDs (image patches), use -100
                lm_labels = torch.full((B, T_merged), -100, device=device, dtype=torch.long)
                # Map the original text token positions into the merged sequence
                image_token_index = getattr(self.model.config, 'image_token_index', 32000)
                for b in range(B):
                    # Find where text tokens are in the merged sequence
                    # Original: [text...] <image> [text...]
                    # Merged:   [text...] [patch1..patchN] [text...]
                    orig_ids = input_ids[b]  # (T_orig,)
                    img_pos = (orig_ids == image_token_index).nonzero(as_tuple=True)[0]
                    if len(img_pos) > 0:
                        pos = img_pos[0].item()
                        n_patches = T_merged - orig_ids.shape[0] + 1  # +1 for replaced token
                        # Before image: positions 0..pos-1 → same in merged
                        # After image:  positions pos+1.. → shifted by (n_patches - 1) in merged
                        if pos > 0:
                            lm_labels[b, :pos] = orig_ids[:pos]
                        after_len = orig_ids.shape[0] - pos - 1
                        if after_len > 0:
                            merged_start = pos + n_patches
                            lm_labels[b, merged_start:merged_start+after_len] = orig_ids[pos+1:]
                    else:
                        # No image token: direct copy
                        T = min(orig_ids.shape[0], T_merged)
                        lm_labels[b, :T] = orig_ids[:T]
                    # Shift left by 1 for next-token prediction
                    lm_labels[b, :-1] = lm_labels[b, 1:].clone()
                    lm_labels[b, -1] = -100
            else:
                lm_labels = None

            # ---- Prepend soft prompt ----
            soft_prompt_expanded = self.soft_prompt.unsqueeze(0).expand(B, -1, -1)  # (B, m, D)
            inputs_embeds = torch.cat(
                [soft_prompt_expanded.to(inputs_embeds.dtype), inputs_embeds], dim=1
            )  # (B, m+T', D)

            if attention_mask is not None:
                soft_mask = torch.ones(
                    B, self.n_soft_tokens, device=device, dtype=attention_mask.dtype
                )
                attention_mask = torch.cat([soft_mask, attention_mask], dim=1)

            if lm_labels is not None:
                label_pad = torch.full(
                    (B, self.n_soft_tokens), -100, device=device, dtype=lm_labels.dtype
                )
                lm_labels = torch.cat([label_pad, lm_labels], dim=1)

            # ---- Forward through language model only (vision already merged) ----
            outputs = self._lang_causal(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=lm_labels,
                output_hidden_states=False,  # we use hooks instead
                use_cache=False,             # disable KV cache to save VRAM
                output_attentions=False,
            )

            return outputs, dict(self._hidden_states)

        finally:
            self._remove_hooks()

    def get_pooled_hidden(
        self,
        hidden_states: dict[int, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        pooling: str = "mean",
    ) -> dict[int, torch.Tensor]:
        """Pool hidden states to (B, D) vectors."""
        result = {}
        for layer_idx, hs in hidden_states.items():
            # Skip soft prompt tokens (first n_soft_tokens)
            hs = hs[:, self.n_soft_tokens:, :]

            if pooling == "mean":
                if attention_mask is not None:
                    # Also skip soft prompt positions in mask
                    mask = attention_mask[:, self.n_soft_tokens:].unsqueeze(-1).float()
                    vec = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                else:
                    vec = hs.mean(dim=1)
            else:  # last
                if attention_mask is not None:
                    mask = attention_mask[:, self.n_soft_tokens:]
                    last_pos = mask.sum(dim=1) - 1
                    vec = hs[torch.arange(hs.size(0)), last_pos, :]
                else:
                    vec = hs[:, -1, :]

            result[layer_idx] = vec  # (B, D)

        return result


# =========================================================================== #
# Loss functions
# =========================================================================== #

class SafetyAlignmentLoss(nn.Module):
    """
    Combined loss for cross-modal safety alignment.

    L = λ_align * L_align + λ_sep * L_sep

    L_align: cosine similarity between current image DiM direction and text DiM direction
             (computed per-batch as an approximation)
    L_sep:   hinge loss pushing safe images to positive side and harmful images
             to negative side of the text safety direction
    """

    def __init__(
        self,
        ref_directions: dict[str, np.ndarray],
        target_layers: list[int] = [13, 14, 15, 16],
        primary_layer: int = 15,
        lambda_align: float = 1.0,
        lambda_sep: float = 1.0,
        margin: float = 0.5,
    ):
        super().__init__()
        self.target_layers = target_layers
        self.primary_layer = primary_layer
        self.lambda_align = lambda_align
        self.lambda_sep = lambda_sep
        self.margin = margin

        # Register reference directions as buffers (not parameters)
        for layer_idx in target_layers:
            d_text = torch.from_numpy(ref_directions[f"d_text_layer{layer_idx}"]).float()
            self.register_buffer(f"d_text_{layer_idx}", d_text)

    def forward(
        self,
        pooled_hidden: dict[int, torch.Tensor],  # {layer: (B, D)}
        image_labels: torch.Tensor,  # (B,) 1=safe, 0=harmful
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Returns:
            total_loss: scalar
            metrics: dict of logged values
        """
        total_loss = torch.tensor(0.0, device=image_labels.device)
        metrics = {}

        safe_mask = (image_labels == 1)
        harm_mask = (image_labels == 0)

        for layer_idx in self.target_layers:
            if layer_idx not in pooled_hidden:
                continue

            h = pooled_hidden[layer_idx]  # (B, D)
            d_text = getattr(self, f"d_text_{layer_idx}")  # (D,)


            # ---- Separation loss ----
            # Project onto text safety direction
            proj = F.normalize(h, dim=-1) @ d_text.to(h.dtype)  # (B,)

            # Safe should be positive, harmful should be negative
            loss_sep = torch.tensor(0.0, device=h.device)
            if safe_mask.any():
                loss_sep = loss_sep + F.relu(self.margin - proj[safe_mask]).mean()
            if harm_mask.any():
                loss_sep = loss_sep + F.relu(self.margin + proj[harm_mask]).mean()

            # ---- Alignment loss (batch-level DiM cosine) ----
            loss_align = torch.tensor(0.0, device=h.device)
            if safe_mask.any() and harm_mask.any():
                h_norm = F.normalize(h, dim=-1)
                batch_d_image = h_norm[safe_mask].mean(dim=0) - h_norm[harm_mask].mean(dim=0)
                batch_d_image = F.normalize(batch_d_image, dim=0)
                cos_sim = torch.dot(batch_d_image, d_text.to(batch_d_image.dtype))
                loss_align = 1.0 - cos_sim  # minimize distance -> maximize cosine

            # Weight primary layer more
            weight = 2.0 if layer_idx == self.primary_layer else 1.0
            layer_loss = weight * (self.lambda_sep * loss_sep + self.lambda_align * loss_align)
            total_loss = total_loss + layer_loss

            # Metrics
            with torch.no_grad():
                sep_val = self._separation(proj, image_labels)
                metrics[f"L{layer_idx}_sep"] = sep_val
                metrics[f"L{layer_idx}_cos"] = float(cos_sim) if safe_mask.any() and harm_mask.any() else 0.0
                metrics[f"L{layer_idx}_loss_sep"] = float(loss_sep)
                metrics[f"L{layer_idx}_loss_align"] = float(loss_align)

        # Normalize by number of layers
        total_loss = total_loss / len(self.target_layers)
        metrics["total_loss"] = float(total_loss.detach())

        return total_loss, metrics

    @staticmethod
    def _separation(proj: torch.Tensor, labels: torch.Tensor) -> float:
        safe = proj[labels == 1]
        harm = proj[labels == 0]
        if len(safe) <= 1 or len(harm) <= 1:
            return 0.0
        num = abs(float(safe.mean() - harm.mean()))
        denom = (float(safe.std()) + float(harm.std())) / 2 + 1e-12
        return num / denom


# =========================================================================== #
# Dataset
# =========================================================================== #

class SafetyImageDataset(Dataset):
    """
    Mixed safe (COCO) + harmful (HoD) image dataset.
    Loads from labels.csv generated by gen_merged_labels.py.
    """

    def __init__(self, labels_csv: str, prompt_template: str = "What is shown in this image?"):
        import pandas as pd
        from PIL import Image as PILImage

        self.prompt = prompt_template
        df = pd.read_csv(labels_csv)
        self.image_paths = df["image_path"].tolist()
        self.labels = df["label"].values.astype(int)

        # Verify at least some images exist
        valid = sum(1 for p in self.image_paths if Path(p).exists())
        print(f"[Dataset] {valid}/{len(self.image_paths)} images found "
              f"({(self.labels==1).sum()} safe, {(self.labels==0).sum()} harmful)")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        from PIL import Image as PILImage
        img = PILImage.open(self.image_paths[idx]).convert("RGB")
        return {
            "image": img,
            "label": self.labels[idx],
            "prompt": self.prompt,
        }


def collate_fn(batch):
    """Custom collate: keep images as list (processor handles batching)."""
    return {
        "images": [item["image"] for item in batch],
        "labels": torch.tensor([item["label"] for item in batch], dtype=torch.long),
        "prompts": [item["prompt"] for item in batch],
    }


# =========================================================================== #
# Trainer
# =========================================================================== #

class SoftPromptTrainer:
    """End-to-end training loop for soft prompt safety alignment."""

    def __init__(
        self,
        wrapper: SoftPromptWrapper,
        criterion: SafetyAlignmentLoss,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        lr: float = 5e-5,
        output_dir: str = "outputs/soft_prompt",
        pooling: str = "mean",
        lambda_lm: float = 0.0,
    ):
        self.wrapper = wrapper
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pooling = pooling
        self.lambda_lm = lambda_lm

        # Collect trainable params
        trainable = [wrapper.soft_prompt]
        if wrapper.use_adapter:
            trainable.extend(wrapper.adapters.parameters())
        self.optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)

        n_params = sum(p.numel() for p in trainable if isinstance(p, nn.Parameter) or p.requires_grad)
        print(f"[Trainer] Trainable parameters: {n_params:,}")

    def train(self, epochs: int = 50, eval_every: int = 5, log_every: int = 1):
        best_transfer = 0.0
        history = []

        for epoch in range(1, epochs + 1):
            # ---- Train ----
            self.wrapper.train()
            epoch_metrics = self._train_epoch(epoch)
            history.append({"epoch": epoch, "phase": "train", **epoch_metrics})

            if epoch % log_every == 0:
                self._print_metrics(epoch, "train", epoch_metrics)

            # ---- Eval ----
            if self.val_loader and epoch % eval_every == 0:
                self.wrapper.eval()
                val_metrics = self._eval_epoch()
                history.append({"epoch": epoch, "phase": "val", **val_metrics})
                self._print_metrics(epoch, "val  ", val_metrics)

                # Save best
                primary = self.wrapper.primary_layer
                transfer = val_metrics.get(f"L{primary}_transfer", 0.0)
                if transfer > best_transfer:
                    best_transfer = transfer
                    self._save_checkpoint(epoch, "best.pt")
                    print(f"  >> New best transfer ratio: {transfer:.4f}")

            # Save periodic
            if epoch % 10 == 0:
                self._save_checkpoint(epoch, f"epoch_{epoch}.pt")

        # Save final
        self._save_checkpoint(epochs, "final.pt")

        # Save history
        with open(self.output_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        print(f"\nTraining complete. Best transfer ratio: {best_transfer:.4f}")
        return history

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        agg = {}
        n_batches = 0
        use_lm = self.lambda_lm > 0

        for batch in tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=False):
            images = batch["images"]
            labels = batch["labels"].to(self.wrapper.soft_prompt.device)
            prompts = batch["prompts"]

            outputs, hidden_states = self.wrapper(
                images=images, text_prompts=prompts, compute_lm_loss=use_lm,
            )
            pooled = self.wrapper.get_pooled_hidden(hidden_states, pooling=self.pooling)

            align_loss, metrics = self.criterion(pooled, labels)

            # Add LM loss as capability-preservation regulariser
            if use_lm and outputs.loss is not None:
                lm_loss = outputs.loss
                loss = align_loss + self.lambda_lm * lm_loss
                metrics["lm_loss"] = float(lm_loss.detach())
            else:
                loss = align_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([self.wrapper.soft_prompt], max_norm=1.0)
            self.optimizer.step()

            for k, v in metrics.items():
                agg[k] = agg.get(k, 0.0) + v
            n_batches += 1

        return {k: v / n_batches for k, v in agg.items()}

    @torch.no_grad()
    def _eval_epoch(self) -> dict[str, float]:
        agg = {}
        all_projs = {l: [] for l in self.wrapper.target_layers}
        all_labels = []
        n_batches = 0

        for batch in self.val_loader:
            images = batch["images"]
            labels = batch["labels"].to(self.wrapper.soft_prompt.device)
            prompts = batch["prompts"]

            outputs, hidden_states = self.wrapper(images=images, text_prompts=prompts)
            pooled = self.wrapper.get_pooled_hidden(hidden_states, pooling=self.pooling)

            loss, metrics = self.criterion(pooled, labels)

            for k, v in metrics.items():
                agg[k] = agg.get(k, 0.0) + v

            # Collect projections for transfer ratio computation
            for layer_idx in self.wrapper.target_layers:
                if layer_idx in pooled:
                    d_text = getattr(self.criterion, f"d_text_{layer_idx}")
                    proj = F.normalize(pooled[layer_idx], dim=-1) @ d_text.to(pooled[layer_idx].dtype)
                    all_projs[layer_idx].append(proj.cpu())

            all_labels.append(labels.cpu())
            n_batches += 1

        result = {k: v / n_batches for k, v in agg.items()}

        # Compute global transfer ratio
        all_labels_cat = torch.cat(all_labels)
        for layer_idx in self.wrapper.target_layers:
            if all_projs[layer_idx]:
                projs = torch.cat(all_projs[layer_idx])
                sep = self._separation_torch(projs, all_labels_cat)
                result[f"L{layer_idx}_transfer"] = sep

        return result

    @staticmethod
    def _separation_torch(proj, labels):
        safe = proj[labels == 1]
        harm = proj[labels == 0]
        if len(safe) == 0 or len(harm) == 0:
            return 0.0
        num = abs(float(safe.mean() - harm.mean()))
        denom = (float(safe.std()) + float(harm.std())) / 2 + 1e-12
        return num / denom

    def _print_metrics(self, epoch, phase, metrics):
        primary = self.wrapper.primary_layer
        loss = metrics.get("total_loss", 0.0)
        cos = metrics.get(f"L{primary}_cos", 0.0)
        sep = metrics.get(f"L{primary}_sep", 0.0)
        transfer = metrics.get(f"L{primary}_transfer", sep)
        lm = metrics.get("lm_loss", None)

        line = (
            f"  [{phase}] Epoch {epoch:3d} | "
            f"loss={loss:.4f}  cos={cos:+.4f}  sep={sep:.3f}  "
            f"transfer={transfer:.4f}"
        )
        if lm is not None:
            line += f"  lm={lm:.4f}"
        print(line)

    def _save_checkpoint(self, epoch, filename):
        state = {
            "epoch": epoch,
            "soft_prompt": self.wrapper.soft_prompt.data.cpu(),
            "optimizer": self.optimizer.state_dict(),
        }
        if self.wrapper.use_adapter:
            state["adapters"] = {
                k: v.state_dict() for k, v in self.wrapper.adapters.items()
            }
        path = self.output_dir / filename
        torch.save(state, path)


# =========================================================================== #
# CLI
# =========================================================================== #

def main():
    parser = argparse.ArgumentParser(description="Soft-prompt safety alignment training")

    # Mode
    parser.add_argument("--extract-ref", action="store_true",
                        help="Extract reference directions from cache and exit")
    parser.add_argument("--eval", action="store_true",
                        help="Evaluate a trained checkpoint")

    # Data
    parser.add_argument("--labels-csv", default="data/visual_harm/labels.csv",
                        help="Path to merged labels.csv (from gen_merged_labels.py)")
    parser.add_argument("--text-cache", default="outputs/cache",
                        help="Cache dir for text hidden states")
    parser.add_argument("--image-cache", default="outputs/cache",
                        help="Cache dir for image hidden states")

    # Model
    parser.add_argument("--model-name",
                        default="/s/models/llava-series/llava-1.5-7b-hf",
                        help="Model path (local or HuggingFace ID)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")

    # Soft prompt config
    parser.add_argument("--n-soft-tokens", type=int, default=8)
    parser.add_argument("--target-layers", nargs="+", type=int, default=[13, 14, 15, 16])
    parser.add_argument("--primary-layer", type=int, default=15)
    parser.add_argument("--use-adapter", action="store_true",
                        help="Add MLP adapter at target layers")
    parser.add_argument("--adapter-hidden", type=int, default=256)

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lambda-align", type=float, default=1.0)
    parser.add_argument("--lambda-sep", type=float, default=1.0)
    parser.add_argument("--lambda-lm", type=float, default=0.0,
                        help="Weight for LM loss regulariser (0 = disabled)")
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--pooling", default="mean", choices=["mean", "last"])
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--prompt-template", default="What is shown in this image?")

    # Output
    parser.add_argument("--output-dir", default="outputs/soft_prompt")
    parser.add_argument("--ckpt", default=None, help="Checkpoint path for --eval")

    args = parser.parse_args()

    # ---- Mode: extract reference directions ----
    if args.extract_ref:
        print("Extracting reference directions from cache...")
        extract_reference_directions(
            text_cache=args.text_cache,
            image_cache=args.image_cache,
            target_layers=args.target_layers,
            output_path=str(Path(args.output_dir) / "ref_directions.npz"),
        )
        return

    # ---- Load reference directions ----
    ref_path = Path(args.output_dir) / "ref_directions.npz"
    if not ref_path.exists():
        print(f"Reference directions not found at {ref_path}")
        print("Run with --extract-ref first.")
        return
    ref_data = dict(np.load(ref_path))
    print(f"Loaded reference directions from {ref_path}")

    # ---- Load model ----
    # Treat as local path if it starts with /, ./, a Windows drive letter (C:\),
    # or is an existing directory on disk.
    import os
    is_local = (
        args.model_name.startswith("/")
        or args.model_name.startswith("./")
        or (len(args.model_name) >= 2 and args.model_name[1] == ":")  # Windows: C:\...
        or os.path.isdir(args.model_name)
    )
    if is_local:
        print(f"Loading model from local path: {args.model_name}")
        import torch as torch_module
        torch_dtype = {
            "float16": torch_module.float16,
            "bfloat16": torch_module.bfloat16,
            "float32": torch_module.float32,
        }[args.dtype]

        # Try different LLaVA variants (don't use local_files_only for actual local paths)
        processor = None
        model = None

        # Try 1: llava-1.5
        try:
            from transformers import LlavaProcessor, LlavaForConditionalGeneration
            processor = LlavaProcessor.from_pretrained(args.model_name)
            model = LlavaForConditionalGeneration.from_pretrained(
                args.model_name,
                torch_dtype=torch_dtype,
                device_map=args.device,
                low_cpu_mem_usage=True,
            )
            print(f"[Model] Loaded as LlavaForConditionalGeneration")
        except Exception as e:
            print(f"[Warning] LlavaForConditionalGeneration failed: {e}")

        # Try 2: llava-next / llava-1.6
        if model is None:
            try:
                from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
                processor = LlavaNextProcessor.from_pretrained(args.model_name)
                model = LlavaNextForConditionalGeneration.from_pretrained(
                    args.model_name,
                    torch_dtype=torch_dtype,
                    device_map=args.device,
                    low_cpu_mem_usage=True,
                )
                print(f"[Model] Loaded as LlavaNextForConditionalGeneration")
            except Exception as e:
                print(f"[Warning] LlavaNextForConditionalGeneration failed: {e}")

        # Try 3: Generic AutoModel/AutoProcessor
        if model is None:
            try:
                from transformers import AutoProcessor, AutoModel
                processor = AutoProcessor.from_pretrained(args.model_name)
                model = AutoModel.from_pretrained(
                    args.model_name,
                    torch_dtype=torch_dtype,
                    device_map=args.device,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                )
                print(f"[Model] Loaded as AutoModel")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load model from {args.model_name} with all strategies. "
                    f"Last error: {e}"
                )

        if processor is None:
            raise RuntimeError("Failed to load processor")
    else:
        from extract.hidden_states import load_model
        model, processor = load_model(args.model_name, device=args.device, dtype=args.dtype)

    # ---- Build wrapper ----
    # d_model is auto-detected from model config inside SoftPromptWrapper
    wrapper = SoftPromptWrapper(
        model=model,
        processor=processor,
        n_soft_tokens=args.n_soft_tokens,
        target_layers=args.target_layers,
        primary_layer=args.primary_layer,
        use_adapter=args.use_adapter,
        adapter_hidden=args.adapter_hidden,
    )

    # ---- Load checkpoint if eval ----
    if args.eval:
        ckpt_path = args.ckpt or str(Path(args.output_dir) / "best.pt")
        ckpt = torch.load(ckpt_path, map_location=args.device)
        wrapper.soft_prompt.data = ckpt["soft_prompt"].to(args.device)
        if args.use_adapter and "adapters" in ckpt:
            for k, sd in ckpt["adapters"].items():
                wrapper.adapters[k].load_state_dict(sd)
        print(f"Loaded checkpoint from {ckpt_path}")

    # ---- Dataset ----
    dataset = SafetyImageDataset(
        labels_csv=args.labels_csv,
        prompt_template=args.prompt_template,
    )

    # Train/val split
    n_val = int(len(dataset) * args.val_split)
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )


    import random
    train_indices = list(range(len(train_set)))
    random.shuffle(train_indices)
    train_loader = DataLoader(
        [train_set[i] for i in train_indices],
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    print(f"[Data] Train: {n_train}, Val: {n_val}")

    # ---- Loss ----
    criterion = SafetyAlignmentLoss(
        ref_directions=ref_data,
        target_layers=args.target_layers,
        primary_layer=args.primary_layer,
        lambda_align=args.lambda_align,
        lambda_sep=args.lambda_sep,
        margin=args.margin,
    ).to(args.device)

    # ---- Train or Eval ----
    trainer = SoftPromptTrainer(
        wrapper=wrapper,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=args.lr,
        output_dir=args.output_dir,
        pooling=args.pooling,
        lambda_lm=args.lambda_lm,
    )

    if args.eval:
        print("\nEvaluating...")
        wrapper.eval()
        metrics = trainer._eval_epoch()
        trainer._print_metrics(0, "eval", metrics)
    else:
        print("\nStarting training...")
        trainer.train(epochs=args.epochs)


if __name__ == "__main__":
    main()
