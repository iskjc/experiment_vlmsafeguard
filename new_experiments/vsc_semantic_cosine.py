"""
VSCBench Semantic-Paired Cross-Modal Cosine Similarity per Category
===================================================================
Per-category, per-layer cosine similarity between:
  - text refusal direction   : mean-diff on safe/harmful text descriptions
  - image refusal direction  : mean-diff on image-token-only hidden states

Output: heatmap matrix (category × layer) showing text vs image direction alignment.

Multi-model support via utils.model_registry.

Usage:
    python vsc_semantic_cosine.py \
        --model_path /path/to/model \
        --captions_jsonl /path/to/captions.jsonl \
        --image_dir /path/to/images \
        --output_dir ./results/vsc_semantic_cosine \
        [--family llava-next]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from utils.model_registry import load_vlm, VLMWrapper


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_vsc(jsonl_path: str) -> tuple[
    dict[str, list[str]], dict[str, list[str]], dict[str, list[str]], dict[str, list[str]]
]:
    safe_texts_by_cat = defaultdict(list)
    harmful_texts_by_cat = defaultdict(list)
    safe_imgs_by_cat = defaultdict(list)
    harmful_imgs_by_cat = defaultdict(list)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            cat = obj.get("category", "unknown")
            if not all(k in obj for k in ("safe_description", "harmful_description",
                                          "safe_image_path", "harmful_image_path")):
                continue
            safe_texts_by_cat[cat].append(obj["safe_description"])
            harmful_texts_by_cat[cat].append(obj["harmful_description"])
            safe_imgs_by_cat[cat].append(obj["safe_image_path"])
            harmful_imgs_by_cat[cat].append(obj["harmful_image_path"])

    total = sum(len(v) for v in safe_texts_by_cat.values())
    print(f"[Data] Loaded {total} pairs from {jsonl_path}")
    for cat in sorted(safe_texts_by_cat.keys()):
        print(f"       {cat}: {len(safe_texts_by_cat[cat])} pairs")

    return dict(safe_texts_by_cat), dict(harmful_texts_by_cat), dict(safe_imgs_by_cat), dict(harmful_imgs_by_cat)


# ---------------------------------------------------------------------------
# Hidden state extraction (model-agnostic)
# ---------------------------------------------------------------------------

class VSCExtractor:
    """Extracts per-layer hidden states using VLMWrapper."""

    QUERY_SUFFIX = "Please describe this image in detail."

    def __init__(self, vlm: VLMWrapper, layers: Optional[list[int]] = None):
        self.vlm = vlm
        self.layers = layers

    def _layer_indices(self, n_layers: int) -> list[int]:
        if self.layers is None:
            return list(range(n_layers))
        return [i for i in self.layers if i < n_layers]

    def _find_query_span(self, input_ids_list: list) -> tuple[int, int]:
        query_ids = self.vlm.tokenizer.encode(
            self.QUERY_SUFFIX, add_special_tokens=False
        )
        n = len(query_ids)
        for i in range(len(input_ids_list) - n, -1, -1):
            if input_ids_list[i: i + n] == query_ids:
                return i, i + n
        return max(0, len(input_ids_list) - 3), len(input_ids_list)

    def extract_texts(self, texts: list[str]) -> dict[int, np.ndarray]:
        """Returns {layer: (N, D)}. Extracts mean hidden state of QUERY_SUFFIX tokens."""
        collected: dict[int, list[np.ndarray]] = {}

        query_token_len = len(
            self.vlm.tokenizer.encode(self.QUERY_SUFFIX, add_special_tokens=False)
        )
        MAX_LENGTH = 512
        TEMPLATE_OVERHEAD = 30
        desc_max_len = MAX_LENGTH - query_token_len - TEMPLATE_OVERHEAD

        for text in tqdm(texts, desc="Extracting [text]"):
            desc_ids = self.vlm.tokenizer.encode(
                text, add_special_tokens=False, truncation=True, max_length=desc_max_len
            )
            truncated_text = self.vlm.tokenizer.decode(desc_ids, skip_special_tokens=True)

            user_text = f"{truncated_text}\n{self.QUERY_SUFFIX}"
            hidden_states, inputs = self.vlm.forward_hidden(None, user_text)

            ids = inputs["input_ids"][0].tolist()
            start, end = self._find_query_span(ids)

            layer_idxs = self._layer_indices(len(hidden_states))
            for i in layer_idxs:
                hs = hidden_states[i][0].float()
                vec = hs[start:end].mean(dim=0)
                collected.setdefault(i, []).append(vec.cpu().numpy())

            del hidden_states

        return {k: np.stack(v, axis=0) for k, v in collected.items()}

    def extract_images(self, image_paths: list[str], image_root: Path) -> dict[int, np.ndarray]:
        """Returns {layer: (N, D)}. Extracts mean hidden state of QUERY_SUFFIX tokens."""
        collected: dict[int, list[np.ndarray]] = {}

        for img_path in tqdm(image_paths, desc="Extracting [image]"):
            full_path = image_root / Path(img_path)
            try:
                image = Image.open(full_path).convert("RGB")
            except Exception as e:
                print(f"  [!] Cannot open {full_path}: {e} — using blank image")
                image = Image.new("RGB", (336, 336))

            hidden_states, inputs = self.vlm.forward_hidden(image, self.QUERY_SUFFIX)

            ids = inputs["input_ids"][0].cpu().tolist()
            hs_seq_len = hidden_states[0].shape[1]

            start, end = self._find_query_span(ids)
            # Correct for image token expansion
            start = self.vlm.offset_for_image_expansion(start, ids, hs_seq_len)
            end = self.vlm.offset_for_image_expansion(end, ids, hs_seq_len)

            layer_idxs = self._layer_indices(len(hidden_states))
            for i in layer_idxs:
                hs = hidden_states[i][0].float()
                vec = hs[start:end].mean(dim=0)
                collected.setdefault(i, []).append(vec.cpu().numpy())

            del hidden_states

        return {k: np.stack(v, axis=0) for k, v in collected.items()}


# ---------------------------------------------------------------------------
# Mean-diff direction
# ---------------------------------------------------------------------------

def mean_diff_direction(safe: np.ndarray, harmful: np.ndarray) -> np.ndarray:
    d = safe.mean(axis=0) - harmful.mean(axis=0)
    norm = np.linalg.norm(d)
    if norm < 1e-12:
        raise ValueError("Direction vector is near-zero.")
    return d / norm


# ---------------------------------------------------------------------------
# Per-layer cosine similarity
# ---------------------------------------------------------------------------

def compute_per_category_cosine(
    text_safe_by_cat, text_harmful_by_cat,
    img_safe_by_cat, img_harmful_by_cat,
):
    all_categories = sorted(set(text_safe_by_cat.keys()) & set(text_harmful_by_cat.keys()) &
                            set(img_safe_by_cat.keys()) & set(img_harmful_by_cat.keys()))

    shared_layers = sorted(
        set.intersection(
            *[set(text_safe_by_cat[c].keys()) for c in all_categories],
            *[set(text_harmful_by_cat[c].keys()) for c in all_categories],
            *[set(img_safe_by_cat[c].keys()) for c in all_categories],
            *[set(img_harmful_by_cat[c].keys()) for c in all_categories],
        )
    )

    results = {}
    matrix = np.zeros((len(all_categories), len(shared_layers)))

    print(f"\n[Cosine] Computing {len(all_categories)} categories × {len(shared_layers)} layers ...")
    for cat_idx, cat in enumerate(all_categories):
        results[cat] = {}
        for layer_idx, layer in enumerate(shared_layers):
            t_dir = mean_diff_direction(text_safe_by_cat[cat][layer], text_harmful_by_cat[cat][layer])
            i_dir = mean_diff_direction(img_safe_by_cat[cat][layer], img_harmful_by_cat[cat][layer])
            cos = float(np.clip(np.dot(t_dir, i_dir), -1.0, 1.0))
            results[cat][layer] = cos
            matrix[cat_idx, layer_idx] = cos

    return results, matrix, all_categories


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_category_heatmap(matrix, categories, layers, save_path):
    fig, ax = plt.subplots(figsize=(14, max(4, len(categories) * 0.3)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)

    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, fontsize=8)
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories, fontsize=9)
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Category", fontsize=11)
    ax.set_title("VSCBench Cross-Modal Refusal Direction Alignment\n(Text vs Image per Category per Layer)", fontsize=12)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Cosine similarity", fontsize=10)

    for i in range(len(categories)):
        for j in range(len(layers)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    color="white" if abs(matrix[i, j]) > 0.5 else "black", fontsize=7)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved to {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_image_states(npz_path):
    data = np.load(npz_path)
    img_safe, img_harmful = {}, {}
    for key in data.files:
        if key.startswith("img_safe_layer"):
            img_safe[int(key[len("img_safe_layer"):])] = data[key]
        elif key.startswith("img_harmful_layer"):
            img_harmful[int(key[len("img_harmful_layer"):])] = data[key]
    print(f"[Cache] Loaded image hidden states from {npz_path} ({len(img_safe)} layers)")
    return img_safe, img_harmful


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--family", type=str, default=None)
    parser.add_argument("--captions_jsonl", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layer indices, e.g. '1,5,10,20'. Default: all layers.")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    image_root = Path(args.image_dir)
    img_npz = out / "img_hidden_states.npz"

    layers = None
    if args.layers:
        layers = [int(x) for x in args.layers.split(",")]

    safe_texts_by_cat, harmful_texts_by_cat, safe_imgs_by_cat, harmful_imgs_by_cat = load_vsc(args.captions_jsonl)
    all_categories = sorted(safe_texts_by_cat.keys())

    safe_imgs_flat = [p for cat in all_categories for p in safe_imgs_by_cat[cat]]
    harmful_imgs_flat = [p for cat in all_categories for p in harmful_imgs_by_cat[cat]]

    vlm = load_vlm(args.model_path, family=args.family, device=args.device)
    extractor = VSCExtractor(vlm, layers=layers)

    # Image hidden states: extract once, cache to disk
    if img_npz.exists():
        print(f"\n[Image] Cache found at {img_npz}, skipping extraction.")
        img_safe_flat, img_harmful_flat = _load_image_states(img_npz)
    else:
        print("\n[Image] Extracting safe image hidden states ...")
        img_safe_flat = extractor.extract_images(safe_imgs_flat, image_root)
        print("[Image] Extracting harmful image hidden states ...")
        img_harmful_flat = extractor.extract_images(harmful_imgs_flat, image_root)
        np.savez(
            img_npz,
            **{f"img_safe_layer{k}": v for k, v in img_safe_flat.items()},
            **{f"img_harmful_layer{k}": v for k, v in img_harmful_flat.items()},
        )
        print(f"[Saved] {img_npz}")

    import gc

    # Extract text hidden states per category
    text_safe_by_cat = {}
    text_harmful_by_cat = {}

    for cat in all_categories:
        print(f"\n[{cat}] Extracting text hidden states ...")
        text_safe_by_cat[cat] = extractor.extract_texts(safe_texts_by_cat[cat])
        text_harmful_by_cat[cat] = extractor.extract_texts(harmful_texts_by_cat[cat])
        gc.collect()
        torch.cuda.empty_cache()

    img_safe_flat, img_harmful_flat = _load_image_states(img_npz)

    # Partition image states by category
    img_safe_by_cat = {}
    img_harmful_by_cat = {}

    start_idx = 0
    for cat in all_categories:
        n = len(safe_texts_by_cat[cat])
        img_safe_by_cat[cat] = {
            layer: img_safe_flat[layer][start_idx:start_idx + n]
            for layer in img_safe_flat.keys()
        }
        start_idx += n

    start_idx = 0
    for cat in all_categories:
        n = len(harmful_texts_by_cat[cat])
        img_harmful_by_cat[cat] = {
            layer: img_harmful_flat[layer][start_idx:start_idx + n]
            for layer in img_harmful_flat.keys()
        }
        start_idx += n

    results, matrix, cat_labels = compute_per_category_cosine(
        text_safe_by_cat, text_harmful_by_cat,
        img_safe_by_cat, img_harmful_by_cat
    )

    # Print report
    print("\n" + "=" * 60)
    print("VSCBench: Text vs Image Refusal Direction per Category")
    print("=" * 60)
    for cat in sorted(results.keys()):
        cos_vals = list(results[cat].values())
        mean_cos = np.mean(cos_vals)
        print(f"[{cat:20s}] Mean cos = {mean_cos:+.4f}  Std = {np.std(cos_vals):.4f}")
    print("=" * 60)

    shared_layers = sorted(list(results[all_categories[0]].keys()))
    plot_category_heatmap(matrix, cat_labels, shared_layers, out / "vsc_heatmap.png")

    with open(out / "cosine_results.json", "w") as f:
        json.dump({cat: {str(layer): cos for layer, cos in results[cat].items()}
                   for cat in results.keys()}, f, indent=2)
    print(f"[Saved] cosine_results.json -> {out}")


if __name__ == "__main__":
    main()
