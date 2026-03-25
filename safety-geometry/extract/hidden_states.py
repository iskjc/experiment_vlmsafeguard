"""
Hidden state extraction for LLaVA-1.5 (llava-hf/llava-1.5-7b-hf).

Architecture reminder:
  CLIP-ViT-L/14 (vision encoder)
    → MLP projector (maps to 4096-dim LLaMA space)
      → LLaMA-2-7b (32 transformer layers, hidden_dim=4096)

output_hidden_states=True returns 33 tensors (embedding + 32 layers),
each of shape (B, T, 4096).

Text extraction:  use processor.tokenizer directly (no <image> token needed)
Image extraction: use full processor with LLaVA prompt format
                  "USER: <image>\n{question} ASSISTANT:"
"""

from __future__ import annotations
import numpy as np
import torch
from tqdm import tqdm
from typing import Literal, Optional
from PIL import Image


LLAVA_PROMPT_TEMPLATE = "USER: <image>\n{question} ASSISTANT:"


class HiddenStateExtractor:
    def __init__(
        self,
        model,
        processor,
        device: str = "cuda",
        pooling: Literal["last", "mean"] = "last",
        batch_size: int = 4,
        layers: Optional[list[int]] = None,
    ):
        self.model = model.eval()
        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self.device = device
        self.pooling = pooling
        self.batch_size = batch_size
        self.layers = layers  # None = all layers

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_text(
        self, texts: list[str], labels: np.ndarray
    ) -> dict[int, np.ndarray]:
        """
        Extract hidden states for plain text strings using LLaMA tokenizer.
        No image token involved — this is the text-only safety direction path.
        """
        return self._run_batches(texts, images=None, labels=labels)

    def extract_image(
        self,
        images: list[Image.Image],
        prompts: list[str],
        labels: np.ndarray,
    ) -> dict[int, np.ndarray]:
        """
        Extract hidden states for image+prompt pairs.
        Prompts are wrapped in LLaVA's conversation format automatically.
        """
        return self._run_batches(prompts, images=images, labels=labels)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_batches(
        self,
        texts: list[str],
        images: Optional[list[Image.Image]],
        labels: np.ndarray,
    ) -> dict[int, list[np.ndarray]]:
        collected: dict[int, list[np.ndarray]] = {}
        mode = "image" if images is not None else "text"

        for start in tqdm(range(0, len(texts), self.batch_size), desc=f"Extracting [{mode}]"):
            end = min(start + self.batch_size, len(texts))
            batch_texts = texts[start:end]
            batch_images = images[start:end] if images is not None else None

            batch_states = self._forward_batch(batch_texts, batch_images)
            for layer_idx, states in batch_states.items():
                collected.setdefault(layer_idx, []).append(states)

        return {k: np.concatenate(v, axis=0) for k, v in collected.items()}

    def _forward_batch(
        self,
        texts: list[str],
        images: Optional[list[Image.Image]],
    ) -> dict[int, np.ndarray]:
        if images is not None:
            # LLaVA multimodal format
            formatted = [LLAVA_PROMPT_TEMPLATE.format(question=t) for t in texts]
            inputs = self.processor(
                text=formatted,
                images=images,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
        else:
            # Text-only: use tokenizer directly
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        # hidden_states: tuple of (n_layers+1) tensors, shape (B, T, D)
        # Index 0 = token embeddings, 1..32 = transformer layer outputs
        hidden_states = outputs.hidden_states

        layer_indices = (
            self.layers if self.layers is not None
            else list(range(len(hidden_states)))
        )

        result = {}
        attention_mask = inputs.get("attention_mask")

        for i in layer_indices:
            if i >= len(hidden_states):
                continue
            hs = hidden_states[i]  # (B, T, D)

            if self.pooling == "last":
                if attention_mask is not None:
                    last_pos = attention_mask.sum(dim=1) - 1  # (B,)
                    vec = hs[torch.arange(hs.size(0)), last_pos, :]
                else:
                    vec = hs[:, -1, :]
            else:  # mean
                if attention_mask is not None:
                    mask = attention_mask.unsqueeze(-1).float()
                    vec = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                else:
                    vec = hs.mean(dim=1)

            result[i] = vec.float().cpu().numpy()  # (B, D=4096)

        return result


# ------------------------------------------------------------------
# LLaVA-1.5 specific loader
# ------------------------------------------------------------------

def load_model(model_name: str, device: str = "cuda", dtype: str = "float16"):
    """
    Load LLaVA-1.5 model + processor.
    Falls back to generic AutoModel for other architectures.
    """
    import transformers
    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]

    print(f"[Model] Loading {model_name} ...")

    is_llava = "llava" in model_name.lower()

    if is_llava:
        from transformers import LlavaForConditionalGeneration, LlavaProcessor
        processor = LlavaProcessor.from_pretrained(model_name)
        model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device,
            low_cpu_mem_usage=True,
        )
    else:
        from transformers import AutoProcessor, AutoModelForVision2Seq
        try:
            processor = AutoProcessor.from_pretrained(model_name)
            model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map=device,
                low_cpu_mem_usage=True,
            )
        except Exception:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            processor = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map=device,
                low_cpu_mem_usage=True,
            )

    n_layers = sum(1 for _ in model.parameters())  # just to trigger model load
    print(f"[Model] Loaded. dtype={dtype}, device={device}")
    return model, processor
