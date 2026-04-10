"""
Hidden-state extraction for VLMs (multi-model support).

Supports: LLaVA-Next, Qwen2-VL, Qwen2.5-VL, InternVL3.
Uses utils.model_registry for unified model loading.

Image direction probe  : hidden state of query tokens or image tokens
Text direction probe   : mean of last N tokens before generation marker

Both extracted from configurable transformer layer (default: layer 16).
"""

from __future__ import annotations
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image

from utils.model_registry import VLMWrapper, load_vlm


class LLaVAProber:
    """
    Extracts per-sample hidden states from VLMs.

    Parameters
    ----------
    model_path : str
        Local path or HF ID of any supported VLM.
    layer : int
        Which transformer layer to probe (0-indexed; default 16).
    device : str
        'cuda' or 'cpu'
    family : str, optional
        Model family override. Auto-detected from model_path if None.
    vlm : VLMWrapper, optional
        Pass an already-loaded VLMWrapper to share across scripts.
    """

    QUERY_SUFFIX = "describe this."

    def __init__(
        self,
        model_path: str = "",
        layer: int = 16,
        device: str = "cuda",
        family: str | None = None,
        vlm: VLMWrapper | None = None,
    ):
        self.layer = layer

        if vlm is not None:
            self.vlm = vlm
        else:
            self.vlm = load_vlm(model_path, family=family, device=device)

        self.device = self.vlm.device
        self.model = self.vlm.model
        self.processor = self.vlm.processor
        self.IMAGE_TOKEN_INDEX = self.vlm.get_image_token_id()

        print(f"[LLaVAProber] Ready. probe_layer={layer}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_query_span(self, input_ids_list: list) -> tuple[int, int]:
        """
        Find start/end positions of QUERY_SUFFIX ("describe this.") tokens in input_ids.
        Returns (start, end). Falls back to last 3 tokens before ASSISTANT:.
        """
        query_ids = self.vlm.tokenizer.encode(
            self.QUERY_SUFFIX, add_special_tokens=False
        )
        n = len(query_ids)
        for i in range(len(input_ids_list) - n, -1, -1):
            if input_ids_list[i: i + n] == query_ids:
                return i, i + n
        asst = self._find_assistant_start(input_ids_list)
        return max(0, asst - 3), asst

    def _find_assistant_start(self, input_ids) -> int:
        """
        Find the start position of 'ASSISTANT:' in token sequence.
        Tries multiple surface forms to handle BPE spacing variations.
        Returns position, or len(input_ids)-1 as fallback.
        """
        full = input_ids.tolist() if isinstance(input_ids, torch.Tensor) else input_ids

        # Try multiple candidates
        for surface in ["ASSISTANT:", " ASSISTANT:", "ASSISTANT"]:
            ids = self.vlm.tokenizer.encode(surface, add_special_tokens=False)
            for i in range(len(full) - len(ids), -1, -1):
                if full[i : i + len(ids)] == ids:
                    return i

        # Fallback
        return max(len(full) - 1, 0)

    def _build_image_prompt(self) -> str:
        """Build chat-template string with image placeholder."""
        return self.vlm.build_prompt(self.QUERY_SUFFIX, has_image=True)

    @torch.no_grad()
    def _forward_hidden(self, inputs: dict) -> torch.Tensor:
        """
        Run forward pass, return hidden states at self.layer.
        Returns: (seq_len, hidden_dim) float32 tensor on CPU
        """
        outputs = self.model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )
        hs = outputs.hidden_states[self.layer]  # (batch, seq_len, hidden_dim)
        return hs[0].cpu().float()  # (seq_len, hidden_dim)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_image(self, image: Image.Image) -> np.ndarray:
        """
        Extract mean hidden state over the query text tokens
        in the image input sequence.
        Returns: (hidden_dim,) L2-normalised array.
        """
        prompt_str = self._build_image_prompt()
        inputs = self.vlm.tokenize(prompt_str, image)

        ids = inputs["input_ids"][0].cpu().tolist()
        outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
        hs = outputs.hidden_states[self.layer][0].cpu().float()  # (T_expanded, D)
        hs_seq_len = hs.shape[0]
        start, end = self._find_query_span(ids)
        try:
            img_pos = ids.index(self.IMAGE_TOKEN_INDEX)
            expansion = hs_seq_len - len(ids)
            if start > img_pos:
                start += expansion
                end += expansion
        except ValueError:
            pass
        vec = hs[start:end].mean(dim=0).numpy()

        return vec / (np.linalg.norm(vec) + 1e-8)

    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        """
        Extracts mean hidden state of "describe this." tokens from text-only input.
        Returns: (hidden_dim,) L2-normalised array.
        """
        prompt_str = self.vlm.build_prompt(
            f"{text} {self.QUERY_SUFFIX}", has_image=False
        )
        inputs = self.vlm.tokenize(prompt_str, image=None)

        hs = self._forward_hidden(inputs)          # (T, D) on CPU
        ids = inputs["input_ids"][0].tolist()
        start, end = self._find_query_span(ids)
        vec = hs[start:end].mean(dim=0).numpy()

        return vec / (np.linalg.norm(vec) + 1e-8)

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    def encode_images(self, images: List[Image.Image]) -> np.ndarray:
        """Returns (N, hidden_dim) array. Uses 'Describe' token position."""
        return np.stack([self.encode_image(img) for img in images], axis=0)

    def encode_image_files(self, paths: List[Path]) -> np.ndarray:
        """Load images from file paths and encode. Uses 'Describe' token position."""
        images = []
        for p in paths:
            try:
                images.append(Image.open(p).convert("RGB"))
            except Exception as e:
                print(f"  [!] Cannot open {p}: {e} — using blank image")
                images.append(Image.new("RGB", (336, 336)))
        return self.encode_images(images)

    # ------------------------------------------------------------------
    # Image token mean pool
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_image_token_pool(self, image: Image.Image) -> np.ndarray:
        """
        Extract mean hidden state over image token positions only (no prompt tokens).
        Returns: (hidden_dim,) L2-normalised array.
        """
        prompt_str = self._build_image_prompt()
        inputs = self.vlm.tokenize(prompt_str, image)

        outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
        hs = outputs.hidden_states[self.layer][0].float()   # (T, D)

        input_ids = inputs["input_ids"][0]
        ids_list = input_ids.tolist()
        hs_seq_len = hs.shape[0]

        img_start, img_end = self.vlm.find_image_token_span(ids_list, hs_seq_len)
        if img_end > img_start:
            vec = hs[img_start:img_end].mean(dim=0)
        else:
            print("[!] No image tokens found, falling back to full-sequence mean.")
            vec = hs.mean(dim=0)

        vec = vec.cpu().numpy()
        return vec / (np.linalg.norm(vec) + 1e-8)

    def encode_image_token_pool_files(self, paths: List[Path]) -> np.ndarray:
        """Load images from file paths and encode using image token mean pool."""
        vecs = []
        for p in paths:
            try:
                img = Image.open(p).convert("RGB")
            except Exception as e:
                print(f"  [!] Cannot open {p}: {e} — using blank image")
                img = Image.new("RGB", (336, 336))
            vecs.append(self.encode_image_token_pool(img))
        return np.stack(vecs, axis=0)

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Returns (N, hidden_dim) array."""
        return np.stack([self.encode_text(t) for t in texts], axis=0)

    # ------------------------------------------------------------------
    # All-layer extraction  {layer_idx: (N, hidden_dim)}
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_text_all_layers(self, text: str) -> dict:
        """
        Returns {layer_idx: (hidden_dim,)} for all transformer layers.
        Probes the last user token before generation marker.
        """
        prompt_str = self.vlm.build_prompt(
            f"{text} {self.QUERY_SUFFIX}", has_image=False
        )
        inputs = self.vlm.tokenize(prompt_str, image=None)

        ids = inputs["input_ids"][0].tolist()
        asst_pos = self._find_assistant_start(ids)
        probe_pos = max(asst_pos - 1, 0)

        outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)

        result = {}
        for i, hs_layer in enumerate(outputs.hidden_states):
            hs = hs_layer[0].float()
            result[i] = hs[probe_pos].cpu().numpy()
        return result

    @torch.no_grad()
    def encode_image_token_pool_all_layers(self, image: Image.Image) -> dict:
        """
        Returns {layer_idx: (hidden_dim,)} for all transformer layers.
        Probes the last token before generation marker in the expanded sequence.
        """
        prompt_str = self._build_image_prompt()
        inputs = self.vlm.tokenize(prompt_str, image)

        ids = inputs["input_ids"][0].cpu().tolist()

        outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)

        hs_seq_len = outputs.hidden_states[0].shape[1]
        asst_pos = self._find_assistant_start(ids)
        probe_pos = max(asst_pos - 1, 0)
        # Correct for image token expansion
        probe_pos = self.vlm.offset_for_image_expansion(probe_pos, ids, hs_seq_len)
        probe_pos = min(probe_pos, hs_seq_len - 1)

        result = {}
        for i, hs_layer in enumerate(outputs.hidden_states):
            hs = hs_layer[0].float()              # (T_expanded, D)
            result[i] = hs[probe_pos].cpu().numpy()
        return result

    def encode_texts_all_layers(self, texts: List[str]) -> dict:
        """Returns {layer_idx: (N, hidden_dim)} for all layers."""
        collected = {}
        for text in texts:
            layer_vecs = self.encode_text_all_layers(text)
            for layer_idx, vec in layer_vecs.items():
                collected.setdefault(layer_idx, []).append(vec)
        return {k: np.stack(v, axis=0) for k, v in collected.items()}

    def encode_image_token_pool_files_all_layers(self, paths: List[Path]) -> dict:
        """Returns {layer_idx: (N, hidden_dim)} for all layers."""
        collected = {}
        for p in paths:
            try:
                img = Image.open(p).convert("RGB")
            except Exception as e:
                print(f"  [!] Cannot open {p}: {e} — using blank image")
                img = Image.new("RGB", (336, 336))
            layer_vecs = self.encode_image_token_pool_all_layers(img)
            for layer_idx, vec in layer_vecs.items():
                collected.setdefault(layer_idx, []).append(vec)
        return {k: np.stack(v, axis=0) for k, v in collected.items()}
