"""
Hidden-state extraction from LLaVA-v1.6-7b-vicuna.
 
Image direction probe  : hidden state of the "Describe" token
Text direction probe   : mean of last N tokens before "ASSISTANT:" marker
                         (the system-end tokens after user query)
 
Both extracted from configurable transformer layer (default: layer 16).
"""
 
from __future__ import annotations
from pathlib import Path
from typing import List
 
import numpy as np
import torch
from PIL import Image
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
 
 
class LLaVAProber:
    """
    Extracts per-sample hidden states from LLaVA-v1.6-vicuna.
 
    Parameters
    ----------
    model_path : str
        Local path or HF ID of llava-hf/llava-v1.6-vicuna-7b-hf
    layer : int
        Which transformer layer to probe (0-indexed; default 16).
    device : str
        'cuda' or 'cpu'
    """
 
    QUERY_SUFFIX = "describe this."
 
    def __init__(self, model_path: str, layer: int = 16, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.layer = layer
 
        print(f"[LLaVAProber] Loading model from {model_path} ...")
        self.processor = LlavaNextProcessor.from_pretrained(model_path)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=self.device,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
 
        print(f"[LLaVAProber] Ready. probe_layer={layer}")
 
    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
 
    def _find_query_span(self, input_ids_list: list) -> tuple[int, int]:
        """
        Find start/end positions of QUERY_SUFFIX ("describe this.") tokens in input_ids.
        Returns (start, end). Falls back to last 3 tokens before ASSISTANT:.
        """
        query_ids = self.processor.tokenizer.encode(
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
            ids = self.processor.tokenizer.encode(surface, add_special_tokens=False)
            for i in range(len(full) - len(ids), -1, -1):
                if full[i : i + len(ids)] == ids:
                    return i
 
        # Fallback
        return max(len(full) - 1, 0)
 
    def _build_image_prompt(self) -> str:
        """Build chat-template string: USER: <image> describe this. ASSISTANT:"""
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.QUERY_SUFFIX},
                ],
            }
        ]
        return self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
 
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
        Extract mean hidden state over the query text tokens ("Describe this image.")
        in the image input sequence — paper-aligned approach.
        Returns: (hidden_dim,) L2-normalised array.
        """
        prompt_str = self._build_image_prompt()
        inputs = self.processor(
            images=image,
            text=prompt_str,
            return_tensors="pt",
        ).to(self.device, torch.float16)

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
        Prompt: USER: {text} describe this. ASSISTANT:
        Extracts mean hidden state of "describe this." tokens.
        Returns: (hidden_dim,) L2-normalised array.
        """
        prompt_str = (
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions. "
            f"USER: {text} {self.QUERY_SUFFIX} ASSISTANT:"
        )
        inputs = self.processor.tokenizer(
            prompt_str,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

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
    # Image token mean pool (paper-aligned approach)
    # ------------------------------------------------------------------

    IMAGE_TOKEN_INDEX = -200   # input_ids placeholder for <image> in llava-hf models

    @torch.no_grad()
    def encode_image_token_pool(self, image: Image.Image) -> np.ndarray:
        """
        Extract mean hidden state over image token positions only (no prompt tokens).
        More aligned with Cross-Modal paper approach than 'Describe' token.
        Returns: (hidden_dim,) L2-normalised array.
        """
        prompt_str = self._build_image_prompt()
        inputs = self.processor(
            images=image,
            text=prompt_str,
            return_tensors="pt",
        ).to(self.device, torch.float16)

        outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
        hs = outputs.hidden_states[self.layer][0].float()   # (T, D)

        input_ids = inputs["input_ids"][0]
        img_mask = (input_ids == self.IMAGE_TOKEN_INDEX)

        # Fallback: try model config image token index
        if img_mask.sum().item() == 0:
            cfg_idx = getattr(getattr(self.model, "config", None), "image_token_index", None)
            if cfg_idx is not None:
                img_mask = (input_ids == cfg_idx)

        if img_mask.sum().item() > 0:
            vec = hs[img_mask].mean(dim=0)
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
        Probes the last user token before ASSISTANT: — this position attends
        to all preceding context and carries the strongest content signal.
        """
        # Manually build Vicuna-style prompt to avoid apply_chat_template issues
        prompt_str = (
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions. "
            f"USER: {text} {self.QUERY_SUFFIX} ASSISTANT:"
        )
        inputs = self.processor.tokenizer(
            prompt_str,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

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
        Probes the last token before ASSISTANT: in the expanded sequence.
        """
        prompt_str = self._build_image_prompt()
        inputs = self.processor(
            images=image,
            text=prompt_str,
            return_tensors="pt",
        ).to(self.device, torch.float16)

        ids = inputs["input_ids"][0].cpu().tolist()

        outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)

        hs_seq_len = outputs.hidden_states[0].shape[1]
        asst_pos = self._find_assistant_start(ids)
        probe_pos = max(asst_pos - 1, 0)
        # Correct for image token expansion (-200 → num_image_tokens)
        try:
            img_pos = ids.index(self.IMAGE_TOKEN_INDEX)
            expansion = hs_seq_len - len(ids)
            if probe_pos > img_pos:
                probe_pos += expansion
        except ValueError:
            pass
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