"""
Semantic-paired cross-modal cosine similarity analysis.

Computes per-layer cosine similarity between:
  - text refusal direction   (mean-diff on text hidden states)
  - image refusal direction  (mean-diff on image-token-only hidden states)

where text and image inputs share the same semantics (paired dataset).

Image hidden states: use prompt="describe this image", but pool ONLY over
the 576 image token positions in the LLM sequence — not the prompt tokens.
LLaVA-1.5 expands the <image> placeholder to 576 tokens whose input
embeddings come from the vision encoder; we identify them via
input_ids == IMAGE_TOKEN_INDEX (default -200 in llava-hf).

Direction method: mean-diff
  direction = mean(safe_states) - mean(harmful_states), then L2-normalised.
  Sign convention: positive projection = safe side.

Usage:
    python -m analysis.semantic_cosine_similarity \\
        --cache-dir outputs/cache \\
        --layers 14 15 16 17

    Or as a module:
        from analysis.semantic_cosine_similarity import SemanticCosineAnalyzer
        analyzer = SemanticCosineAnalyzer()
        results = analyzer.compare(text_states, text_labels, image_states, image_labels)
        analyzer.print_report(results)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image


# --------------------------------------------------------------------------- #
# Result dataclass
# --------------------------------------------------------------------------- #

@dataclass
class LayerResult:
    layer_idx: int
    text_direction: np.ndarray    # (D,) unit vector
    image_direction: np.ndarray   # (D,) unit vector
    cosine_similarity: float
    abs_cosine: float


# --------------------------------------------------------------------------- #
# Direction: mean-diff
# --------------------------------------------------------------------------- #

def mean_diff_direction(states: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Compute refusal direction as (mean_safe - mean_harmful), L2-normalised.

    Parameters
    ----------
    states : (N, D) hidden states
    labels : (N,)  1=safe, 0=harmful

    Returns
    -------
    (D,) unit vector pointing from harmful toward safe
    """
    safe_mask = labels == 1
    harm_mask = labels == 0

    if not safe_mask.any():
        raise ValueError("No safe samples found (label==1).")
    if not harm_mask.any():
        raise ValueError("No harmful samples found (label==0).")

    direction = states[safe_mask].mean(axis=0) - states[harm_mask].mean(axis=0)
    norm = np.linalg.norm(direction)
    if norm < 1e-12:
        raise ValueError("Direction vector is near-zero; check your hidden states.")
    return direction / norm


# --------------------------------------------------------------------------- #
# Analyzer
# --------------------------------------------------------------------------- #

class SemanticCosineAnalyzer:
    """
    Per-layer cosine similarity between text and image refusal directions.
    Directions are computed via mean-diff (no PLS).
    """

    def __init__(self):
        self.results_: dict[int, LayerResult] = {}

    def compare(
        self,
        text_states: dict[int, np.ndarray],   # {layer: (N, D)}
        text_labels: np.ndarray,              # (N,) 1=safe 0=harmful
        image_states: dict[int, np.ndarray],  # {layer: (N, D)} — image-token pooled
        image_labels: np.ndarray,             # (N,) 1=safe 0=harmful
        layers: Optional[list[int]] = None,
    ) -> dict[int, LayerResult]:
        shared = set(text_states.keys()) & set(image_states.keys())
        if layers is not None:
            shared = shared & set(layers)
        shared = sorted(shared)

        if not shared:
            raise ValueError(
                f"No overlapping layers. "
                f"Text: {sorted(text_states.keys())}, "
                f"Image: {sorted(image_states.keys())}"
            )

        print(f"\n[SemanticCosine] {len(shared)} layers: {shared}")

        for layer_idx in shared:
            text_dir = mean_diff_direction(text_states[layer_idx], text_labels)
            image_dir = mean_diff_direction(image_states[layer_idx], image_labels)

            cos_sim = float(np.dot(text_dir, image_dir))
            cos_sim = max(-1.0, min(1.0, cos_sim))

            result = LayerResult(
                layer_idx=layer_idx,
                text_direction=text_dir,
                image_direction=image_dir,
                cosine_similarity=cos_sim,
                abs_cosine=abs(cos_sim),
            )
            self.results_[layer_idx] = result
            tag = "ALIGNED" if abs(cos_sim) > 0.5 else ("ORTHO" if abs(cos_sim) < 0.2 else "PARTIAL")
            print(f"  Layer {layer_idx:3d} | cos={cos_sim:+.4f}  |cos|={abs(cos_sim):.4f}  {tag}")

        return self.results_

    def print_report(self, results: Optional[dict] = None) -> None:
        if results is None:
            results = self.results_
        if not results:
            print("[SemanticCosine] No results.")
            return

        cos_vals = [r.cosine_similarity for r in results.values()]
        abs_vals = [r.abs_cosine for r in results.values()]

        print("\n" + "=" * 58)
        print("SEMANTIC-PAIRED CROSS-MODAL COSINE SIMILARITY")
        print("=" * 58)
        print(f"{'Layer':>6}  {'cos_sim':>8}  {'|cos|':>6}  Interpretation")
        print("-" * 58)
        for layer_idx in sorted(results.keys()):
            r = results[layer_idx]
            if r.abs_cosine > 0.5:
                interp = "ALIGNED"
            elif r.abs_cosine < 0.2:
                interp = "ORTHOGONAL"
            else:
                interp = "PARTIAL"
            print(f"{layer_idx:>6}  {r.cosine_similarity:>+8.4f}  {r.abs_cosine:>6.4f}  {interp}")
        print("-" * 58)
        print(f"  Mean cos_sim : {np.mean(cos_vals):+.4f}")
        print(f"  Mean |cos|   : {np.mean(abs_vals):.4f}")
        print(f"  Std  |cos|   : {np.std(abs_vals):.4f}")
        print("=" * 58)


# --------------------------------------------------------------------------- #
# Image hidden state extraction — image-token positions only
# --------------------------------------------------------------------------- #

# LLaVA-1.5 (llava-hf) uses -200 as the image token index in input_ids
# before the model replaces them with vision encoder embeddings.
IMAGE_TOKEN_INDEX = -200
IMAGE_PROMPT = "describe this image"
LLAVA_PROMPT_TEMPLATE = "USER: <image>\n{question} ASSISTANT:"


class ImageTokenExtractor:
    """
    Extracts per-layer hidden states pooled ONLY over the image token positions
    (the 576 tokens that replace <image> in the LLM sequence).

    The prompt text tokens are excluded from pooling.
    """

    def __init__(
        self,
        model,
        processor,
        device: str = "cuda",
        batch_size: int = 2,
        layers: Optional[list[int]] = None,
        image_token_index: int = IMAGE_TOKEN_INDEX,
    ):
        self.model = model.eval()
        self.processor = processor
        self.device = device
        self.batch_size = batch_size
        self.layers = layers
        self.image_token_index = image_token_index

    def extract(
        self,
        images: list[Image.Image],
        labels: np.ndarray,
        prompt: str = IMAGE_PROMPT,
    ) -> dict[int, np.ndarray]:
        """
        Returns {layer_idx: (N, D)} where each row is the mean hidden state
        over the 576 image token positions for that sample.
        """
        collected: dict[int, list[np.ndarray]] = {}
        formatted_prompt = LLAVA_PROMPT_TEMPLATE.format(question=prompt)

        for start in tqdm(range(0, len(images), self.batch_size), desc="Extracting [image tokens]"):
            end = min(start + self.batch_size, len(images))
            batch_images = images[start:end]
            batch_texts = [formatted_prompt] * len(batch_images)

            batch_states = self._forward_batch(batch_texts, batch_images)
            for layer_idx, states in batch_states.items():
                collected.setdefault(layer_idx, []).append(states)

        return {k: np.concatenate(v, axis=0) for k, v in collected.items()}

    def _forward_batch(
        self,
        texts: list[str],
        images: list[Image.Image],
    ) -> dict[int, np.ndarray]:
        inputs = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states  # tuple: (n_layers+1,) each (B, T, D)

        # Find image token mask for each sample in the batch.
        # input_ids shape: (B, T)
        # After processor, the <image> token is IMAGE_TOKEN_INDEX (-200).
        # NOTE: some versions use a different sentinel; adjust image_token_index if needed.
        input_ids = inputs["input_ids"]  # (B, T)
        image_mask = (input_ids == self.image_token_index)  # (B, T) bool

        layer_indices = (
            self.layers if self.layers is not None
            else list(range(len(hidden_states)))
        )

        result = {}
        for i in layer_indices:
            if i >= len(hidden_states):
                continue
            hs = hidden_states[i]  # (B, T, D)

            batch_vecs = []
            for b in range(hs.size(0)):
                img_positions = image_mask[b]  # (T,) bool
                n_img_tokens = img_positions.sum().item()

                if n_img_tokens == 0:
                    # Fallback: warn and use mean over all tokens
                    print(
                        f"[Warning] Sample {b} has no image tokens at layer {i}. "
                        f"Check image_token_index={self.image_token_index}. "
                        f"Falling back to full-sequence mean."
                    )
                    vec = hs[b].mean(dim=0)
                else:
                    # Mean pool over image token positions only
                    vec = hs[b][img_positions].mean(dim=0)  # (D,)

                batch_vecs.append(vec.float().cpu().numpy())

            result[i] = np.stack(batch_vecs, axis=0)  # (B, D)

        return result


# --------------------------------------------------------------------------- #
# CLI — loads cached .npy files
# --------------------------------------------------------------------------- #

def _load_cache(prefix: str, cache_dir: Path) -> tuple[dict[int, np.ndarray], np.ndarray]:
    labels = np.load(cache_dir / f"{prefix}_labels.npy")
    states = {}
    for f in sorted(cache_dir.glob(f"{prefix}_layer*.npy")):
        layer_idx = int(f.stem.replace(f"{prefix}_layer", ""))
        states[layer_idx] = np.load(f)
    print(f"[Cache] {prefix}: {len(states)} layers, {labels.shape[0]} samples")
    return states, labels


def main():
    parser = argparse.ArgumentParser(
        description="Semantic-paired cross-modal cosine similarity (mean-diff directions)."
    )
    parser.add_argument("--cache-dir", type=str, default="outputs/cache")
    parser.add_argument("--text-cache", type=str, default=None)
    parser.add_argument("--image-cache", type=str, default=None)
    parser.add_argument("--layers", nargs="+", type=int, default=None)
    args = parser.parse_args()

    text_dir = Path(args.text_cache or args.cache_dir)
    image_dir = Path(args.image_cache or args.cache_dir)

    text_states, text_labels = _load_cache("text", text_dir)
    image_states, image_labels = _load_cache("image", image_dir)

    analyzer = SemanticCosineAnalyzer()
    results = analyzer.compare(
        text_states, text_labels,
        image_states, image_labels,
        layers=args.layers,
    )
    analyzer.print_report(results)


if __name__ == "__main__":
    main()
