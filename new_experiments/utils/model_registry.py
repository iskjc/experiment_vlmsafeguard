"""
Multi-model registry for VLM experiments.

Provides a unified interface to load different VLMs and extract hidden states,
generate text, find special token positions, and build prompts.

Supported model families:
  - llava-next (llava-v1.6-vicuna-7b-hf, etc.)
  - qwen2-vl   (Qwen2-VL-7B-Instruct, Qwen2.5-VL-7B-Instruct, etc.)
  - internvl3   (InternVL3-8B-HF, etc.)

Usage:
    from utils.model_registry import load_vlm

    vlm = load_vlm("/path/to/model")          # auto-detect family
    vlm = load_vlm("/path/to/model", family="qwen2-vl")  # explicit

    # Hidden states
    hs = vlm.forward_hidden(image, text)       # dict of layer → (T, D)
    # Generation
    response = vlm.generate(image, text)
    # Internals
    layer_norm, lm_head = vlm.get_lm_head_components()
"""

from __future__ import annotations

import abc
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class VLMWrapper(abc.ABC):
    """Unified interface for Vision-Language Models."""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self._load_model()

    @abc.abstractmethod
    def _load_model(self):
        """Load model and processor into self.model / self.processor."""

    @abc.abstractmethod
    def build_prompt(self, text: str, has_image: bool) -> str:
        """Build a chat-template prompt string."""

    @abc.abstractmethod
    def tokenize(self, prompt: str, image: Optional[Image.Image] = None) -> dict:
        """Tokenize prompt (+ optional image) → dict ready for model(**inputs)."""

    @abc.abstractmethod
    def get_image_token_id(self) -> int:
        """Return the placeholder token ID for <image> in input_ids."""

    @abc.abstractmethod
    def get_lm_head_components(self) -> tuple:
        """Return (layer_norm, lm_head) modules for logit lens."""

    @property
    @abc.abstractmethod
    def tokenizer(self):
        """Return the text tokenizer."""

    @property
    @abc.abstractmethod
    def num_layers(self) -> int:
        """Number of transformer layers (not counting embedding)."""

    # ---- shared helpers ----

    @torch.no_grad()
    def forward_hidden(
        self, image: Optional[Image.Image], text: str
    ) -> tuple[list[torch.Tensor], dict]:
        """
        Run forward pass, return (hidden_states, inputs).
        hidden_states: list of (1, T, D) tensors, len = num_layers + 1.
        inputs: the tokenized inputs dict (for position calculation).
        """
        prompt = self.build_prompt(text, has_image=image is not None)
        inputs = self.tokenize(prompt, image)
        outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
        return list(outputs.hidden_states), inputs

    @torch.no_grad()
    def generate(
        self,
        image: Optional[Image.Image],
        text: str,
        max_new_tokens: int = 128,
    ) -> str:
        """Generate a text response."""
        prompt = self.build_prompt(text, has_image=image is not None)
        inputs = self.tokenize(prompt, image)
        input_len = inputs["input_ids"].shape[1]
        out_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
        generated = out_ids[0, input_len:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def find_image_token_span(
        self, input_ids_list: list[int], hs_seq_len: int
    ) -> tuple[int, int]:
        """
        Find the start and end positions of expanded image tokens in hidden states.
        Returns (start, end) in the hidden-state coordinate system.
        """
        img_token_id = self.get_image_token_id()
        n_placeholders = input_ids_list.count(img_token_id)
        if n_placeholders == 0:
            return 0, 0  # no image tokens
        img_start = input_ids_list.index(img_token_id)
        total_img_tokens = hs_seq_len - len(input_ids_list) + n_placeholders
        return img_start, img_start + total_img_tokens

    def get_last_token_pos(
        self, input_ids_list: list[int], hs_seq_len: int
    ) -> int:
        """
        Get the last input token position in the hidden-state coordinate system,
        accounting for image token expansion.
        """
        # Last position in hidden states = hs_seq_len - 1
        # But for models that expand image tokens, input_ids length != hs_seq_len
        return hs_seq_len - 1

    def offset_for_image_expansion(
        self, pos: int, input_ids_list: list[int], hs_seq_len: int
    ) -> int:
        """
        Adjust a position in input_ids space to hidden-state space,
        accounting for image token expansion.
        """
        img_token_id = self.get_image_token_id()
        try:
            img_pos = input_ids_list.index(img_token_id)
            expansion = hs_seq_len - len(input_ids_list)
            if pos > img_pos:
                return pos + expansion
        except ValueError:
            pass
        return pos


# ---------------------------------------------------------------------------
# LLaVA-Next
# ---------------------------------------------------------------------------

class LLaVANextWrapper(VLMWrapper):

    def _load_model(self):
        from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
        print(f"[LLaVA-Next] Loading from {self.model_path} ...")
        self.processor = LlavaNextProcessor.from_pretrained(self.model_path)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map=self.device,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        print("[LLaVA-Next] Ready.")

    def build_prompt(self, text: str, has_image: bool) -> str:
        if has_image:
            conversation = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": text},
                ]}
            ]
        else:
            conversation = [
                {"role": "user", "content": text}
            ]
        return self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )

    def tokenize(self, prompt: str, image: Optional[Image.Image] = None) -> dict:
        if image is not None:
            return self.processor(
                images=image, text=prompt, return_tensors="pt",
            ).to(self.device, torch.float16)
        return self.processor.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024,
        ).to(self.device)

    def get_image_token_id(self) -> int:
        return -200

    def get_lm_head_components(self) -> tuple:
        layer_norm = lm_head = None
        for name, module in self.model.named_modules():
            if name == "language_model.norm":
                layer_norm = module
            if name == "lm_head":
                lm_head = module
        if layer_norm is None or lm_head is None:
            raise AttributeError(
                f"Cannot find layer_norm/lm_head. "
                f"Norms found: {[n for n, _ in self.model.named_modules() if 'norm' in n.lower()]}"
            )
        return layer_norm, lm_head

    @property
    def tokenizer(self):
        return self.processor.tokenizer

    @property
    def num_layers(self) -> int:
        return self.model.config.text_config.num_hidden_layers


# ---------------------------------------------------------------------------
# Qwen2-VL / Qwen2.5-VL
# ---------------------------------------------------------------------------

class Qwen2VLWrapper(VLMWrapper):

    def _load_model(self):
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        print(f"[Qwen2-VL] Loading from {self.model_path} ...")
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map=self.device,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        print("[Qwen2-VL] Ready.")

    def build_prompt(self, text: str, has_image: bool) -> str:
        if has_image:
            conversation = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": text},
                ]}
            ]
        else:
            conversation = [
                {"role": "user", "content": text}
            ]
        return self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )

    def tokenize(self, prompt: str, image: Optional[Image.Image] = None) -> dict:
        if image is not None:
            return self.processor(
                images=image, text=prompt, return_tensors="pt",
            ).to(self.device, torch.float16)
        return self.processor.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024,
        ).to(self.device)

    def get_image_token_id(self) -> int:
        # Qwen2-VL uses <|image_pad|> token, ID = 151655
        # TODO: fill in the correct token ID for your model version
        return 151655

    def get_lm_head_components(self) -> tuple:
        layer_norm = lm_head = None
        for name, module in self.model.named_modules():
            if name == "model.norm":
                layer_norm = module
            if name == "lm_head":
                lm_head = module
        if layer_norm is None or lm_head is None:
            raise AttributeError(
                f"Cannot find layer_norm/lm_head in Qwen2-VL. "
                f"Norms: {[n for n, _ in self.model.named_modules() if 'norm' in n.lower() and 'layer' not in n.lower()]}"
            )
        return layer_norm, lm_head

    @property
    def tokenizer(self):
        return self.processor.tokenizer

    @property
    def num_layers(self) -> int:
        return self.model.config.num_hidden_layers


# ---------------------------------------------------------------------------
# InternVL3
# ---------------------------------------------------------------------------

class InternVL3Wrapper(VLMWrapper):

    def _load_model(self):
        from transformers import AutoModel, AutoTokenizer
        print(f"[InternVL3] Loading from {self.model_path} ...")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map=self.device,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        self.model.eval()
        self.processor = None  # InternVL3 uses its own chat method
        print("[InternVL3] Ready.")

    def build_prompt(self, text: str, has_image: bool) -> str:
        # InternVL3 uses <image> tag inline
        if has_image:
            return f"<image>\n{text}"
        return text

    def tokenize(self, prompt: str, image: Optional[Image.Image] = None) -> dict:
        # InternVL3 has its own tokenization via model.chat() internals
        # TODO: fill in the exact tokenization pipeline for InternVL3
        # This is a placeholder — InternVL3 uses trust_remote_code and
        # has custom image preprocessing. You may need to call
        # model.build_input_ids() or similar.
        raise NotImplementedError(
            "InternVL3 tokenize() needs to be implemented based on "
            "the specific model's API. Run diagnose_model_v2.py first."
        )

    def get_image_token_id(self) -> int:
        # TODO: fill in correct image token ID for InternVL3
        return -100

    def get_lm_head_components(self) -> tuple:
        layer_norm = lm_head = None
        for name, module in self.model.named_modules():
            # TODO: verify these paths with diagnose_model_v2.py
            if "language_model.model.norm" in name and "layer" not in name:
                layer_norm = module
            if name.endswith("lm_head"):
                lm_head = module
        if layer_norm is None or lm_head is None:
            raise AttributeError(
                f"Cannot find layer_norm/lm_head in InternVL3. "
                f"Run diagnose_model_v2.py to find the correct paths."
            )
        return layer_norm, lm_head

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def num_layers(self) -> int:
        # TODO: verify — typically InternVL3-8B has 32 layers
        return self.model.config.llm_config.num_hidden_layers


# ---------------------------------------------------------------------------
# Auto-detection & factory
# ---------------------------------------------------------------------------

_FAMILY_MAP = {
    "llava": LLaVANextWrapper,
    "llava-next": LLaVANextWrapper,
    "qwen2-vl": Qwen2VLWrapper,
    "qwen2.5-vl": Qwen2VLWrapper,
    "internvl": InternVL3Wrapper,
    "internvl3": InternVL3Wrapper,
}


def detect_family(model_path: str) -> str:
    """Guess model family from path/name."""
    p = model_path.lower()
    if "internvl" in p:
        return "internvl3"
    if "qwen2" in p and "vl" in p:
        return "qwen2-vl"
    if "llava" in p:
        return "llava-next"
    raise ValueError(
        f"Cannot auto-detect model family from '{model_path}'. "
        f"Pass family= explicitly. Supported: {list(_FAMILY_MAP.keys())}"
    )


def load_vlm(model_path: str, family: str | None = None, device: str = "cuda") -> VLMWrapper:
    """
    Load a VLM with unified interface.

    Parameters
    ----------
    model_path : str
        Path or HuggingFace ID
    family : str, optional
        Model family. Auto-detected if None.
        Options: "llava-next", "qwen2-vl", "qwen2.5-vl", "internvl3"
    device : str
        "cuda" or "cpu"
    """
    if family is None:
        family = detect_family(model_path)
    family = family.lower()

    cls = _FAMILY_MAP.get(family)
    if cls is None:
        raise ValueError(f"Unknown family '{family}'. Supported: {list(_FAMILY_MAP.keys())}")

    return cls(model_path, device=device)
