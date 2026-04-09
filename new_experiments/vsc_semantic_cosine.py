"""
VSCBench Semantic-Paired Cross-Modal Cosine Similarity per Category
===================================================================
Per-category, per-layer cosine similarity between:
  - text refusal direction   : mean-diff on safe/harmful text descriptions
  - image refusal direction  : mean-diff on image-token-only hidden states

Output: heatmap matrix (category × layer) showing text vs image direction alignment.

Dataset: vscbench_captions.jsonl (produced by create_data/vscbench_caption.py)
  {
    "pair_id":             "discrimination/0_0",
    "category":            "discrimination",
    "safe_image_path":     "discrimination/safe_discrimination_0_0.png",
    "harmful_image_path":  "discrimination/unsafe_discrimination_0_0.png",
    "safe_description":    "...",
    "harmful_description": "..."
  }

Text extraction: raw text description, mean or last token pooling, on GPU.
Image extraction: "Describe this image." prompt, image token (576 tokens) mean pooling, on GPU.
Direction: mean(safe) - mean(harmful), L2-normalised, computed per category per layer.

Usage:
    python new_experiments/vsc_semantic_cosine.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

# ---------------------------------------------------------------------------
# Config — fill these in
# ---------------------------------------------------------------------------
LLAVA_MODEL_PATH  = ""   # e.g. "/model/llava-v1.6-vicuna-7b-hf"
VSC_CAPTIONS_JSONL = ""  # e.g. "outputs/vscbench_captions.jsonl"
VSC_IMAGE_DIR      = ""  # e.g. "/data/vscbench/vscbench_image_centric_images"
OUTPUT_DIR         = ""  # e.g. "./results/vsc_semantic_cosine"

# Which layers to compute (None = all 33 layers including embedding layer 0)
LAYERS: Optional[list[int]] = None  # e.g. list(range(1, 33))

# LLaVA-NeXT image token placeholder in input_ids (before model expansion)
# Default -200 works for llava-hf models; override if needed.
IMAGE_TOKEN_INDEX = -200


BATCH_SIZE = 1   # increase if VRAM allows; LLaVA-NeXT is memory-heavy


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_vsc(jsonl_path: str) -> tuple[
    dict[str, list[str]], dict[str, list[str]], dict[str, list[str]], dict[str, list[str]]
]:
    """
    Returns data grouped by category:
        {category: [safe_texts]}, {category: [harmful_texts]},
        {category: [safe_img_paths]}, {category: [harmful_img_paths]}
    """
    from collections import defaultdict
    safe_texts_by_cat = defaultdict(list)
    harmful_texts_by_cat = defaultdict(list)
    safe_imgs_by_cat = defaultdict(list)
    harmful_imgs_by_cat = defaultdict(list)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            cat = obj.get("category", "unknown")
            safe_texts_by_cat[cat].append(obj["safe_description"])
            harmful_texts_by_cat[cat].append(obj["harmful_description"])
            safe_imgs_by_cat[cat].append(obj["safe_image_path"])
            harmful_imgs_by_cat[cat].append(obj["harmful_image_path"])

    total = sum(len(v) for v in safe_texts_by_cat.values())
    print(f"[Data] Loaded {total} pairs from {jsonl_path}")
    print(f"[Data] Categories: {sorted(safe_texts_by_cat.keys())}")
    for cat in sorted(safe_texts_by_cat.keys()):
        print(f"       {cat}: {len(safe_texts_by_cat[cat])} pairs")

    return dict(safe_texts_by_cat), dict(harmful_texts_by_cat), dict(safe_imgs_by_cat), dict(harmful_imgs_by_cat)


# ---------------------------------------------------------------------------
# Hidden state extraction
# ---------------------------------------------------------------------------

class VSCExtractor:
    """
    Extracts per-layer hidden states for VSCBench text descriptions and images.
    """

    QUERY_SUFFIX = "Please describe this image in detail."   # appended to both modalities

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        layers: Optional[list[int]] = None,
        image_token_index: int = IMAGE_TOKEN_INDEX,
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.layers = layers
        self.image_token_index = image_token_index

        print(f"[Extractor] Loading LLaVA-NeXT from {model_path} ...")
        self.processor = LlavaNextProcessor.from_pretrained(model_path)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=self.device,
            low_cpu_mem_usage=True,
        ).eval()

        self._image_prompt_str = self._build_image_prompt()
        print(f"[Extractor] Ready. image_token_index={image_token_index}")

    # ------------------------------------------------------------------ #
    # Prompt builders
    # ------------------------------------------------------------------ #

    def _build_image_prompt(self) -> str:
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

    def _build_text_prompt(self, text: str) -> str:
        # Append QUERY_SUFFIX after description.
        # Truncation is handled at encode time (max_length=512) so the description
        # may be cut, but QUERY_SUFFIX tokens at the end are always preserved.
        conversation = [{"role": "user", "content": f"{text}\n{self.QUERY_SUFFIX}"}]
        return self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )

    def _find_query_span(self, input_ids_list: list) -> tuple[int, int]:
        """
        Find positions of QUERY_SUFFIX tokens in input_ids.
        Returns (start, end) slice covering those tokens.
        """
        query_ids = self.processor.tokenizer.encode(
            self.QUERY_SUFFIX, add_special_tokens=False
        )
        n = len(query_ids)
        for i in range(len(input_ids_list) - n, -1, -1):
            if input_ids_list[i: i + n] == query_ids:
                return i, i + n
        # Fallback: last 3 tokens before end
        return max(0, len(input_ids_list) - 3), len(input_ids_list)

    def _find_query_span_image(self, input_ids_list: list, hs_seq_len: int) -> tuple[int, int]:
        """
        Same as _find_query_span but corrects for the image token expansion.
        input_ids has one token (-200) for <image>; hidden_states has num_image_tokens
        tokens there. All query positions after the image token must be offset.
        """
        start, end = self._find_query_span(input_ids_list)
        try:
            img_pos = input_ids_list.index(self.image_token_index)
            expansion = hs_seq_len - len(input_ids_list)  # e.g. 576 - 1 = 575
            if start > img_pos:
                start += expansion
                end += expansion
        except ValueError:
            pass
        return start, end

    # ------------------------------------------------------------------ #
    # Core forward
    # ------------------------------------------------------------------ #

    def _layer_indices(self, n_layers: int) -> list[int]:
        if self.layers is None:
            return list(range(n_layers))
        return [i for i in self.layers if i < n_layers]

    # ------------------------------------------------------------------ #
    # Text extraction
    # ------------------------------------------------------------------ #

    def extract_texts(self, texts: list[str]) -> dict[int, np.ndarray]:
        """
        Returns {layer: (N, D)}.
        Prompt: USER: {description}\nPlease describe this image in detail. ASSISTANT:
        Extracts mean hidden state of QUERY_SUFFIX tokens.

        To guarantee QUERY_SUFFIX is never truncated, we first tokenize the
        description alone, truncate it so that the full prompt fits in max_length,
        then rebuild the prompt with the truncated description.
        """
        collected: dict[int, list[np.ndarray]] = {}

        # Pre-compute how many tokens QUERY_SUFFIX + template overhead consume
        query_token_len = len(
            self.processor.tokenizer.encode(self.QUERY_SUFFIX, add_special_tokens=False)
        )
        MAX_LENGTH   = 512
        # ~30 tokens of overhead for chat template (system/role tokens, etc.)
        TEMPLATE_OVERHEAD = 30
        desc_max_len = MAX_LENGTH - query_token_len - TEMPLATE_OVERHEAD

        for text in tqdm(texts, desc="Extracting [text]"):
            # Truncate description so the full prompt will fit within MAX_LENGTH
            desc_ids = self.processor.tokenizer.encode(
                text, add_special_tokens=False, truncation=True, max_length=desc_max_len
            )
            truncated_text = self.processor.tokenizer.decode(desc_ids, skip_special_tokens=True)

            prompt_str = self._build_text_prompt(truncated_text)
            inputs = self.processor.tokenizer(
                prompt_str,
                return_tensors="pt",
                truncation=False,   # already truncated above; no further cut needed
                add_special_tokens=False,
            ).to(self.device)

            ids = inputs["input_ids"][0].tolist()
            start, end = self._find_query_span(ids)

            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)

            layer_idxs = self._layer_indices(len(outputs.hidden_states))
            for i in layer_idxs:
                hs = outputs.hidden_states[i][0].float()  # (T, D) on GPU
                vec = hs[start:end].mean(dim=0)
                collected.setdefault(i, []).append(vec.cpu().numpy())

            del outputs

        return {k: np.stack(v, axis=0) for k, v in collected.items()}

    # ------------------------------------------------------------------ #
    # Image extraction — image tokens only
    # ------------------------------------------------------------------ #

    def extract_images(self, image_paths: list[str], image_root: Path) -> dict[int, np.ndarray]:
        """
        Returns {layer: (N, D)}.
        Prompt: USER: <image>\nPlease describe this image in detail. ASSISTANT:
        Extracts mean hidden state of QUERY_SUFFIX tokens.
        """
        collected: dict[int, list[np.ndarray]] = {}

        for img_path in tqdm(image_paths, desc="Extracting [image]"):
            full_path = image_root / Path(img_path)
            try:
                image = Image.open(full_path).convert("RGB")
            except Exception as e:
                print(f"  [!] Cannot open {full_path}: {e} — using blank image")
                image = Image.new("RGB", (336, 336))

            inputs = self.processor(
                images=image,
                text=self._image_prompt_str,
                return_tensors="pt",
            ).to(self.device, torch.float16)

            ids = inputs["input_ids"][0].tolist()

            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)

            hs_seq_len = outputs.hidden_states[0].shape[1]
            start, end = self._find_query_span_image(ids, hs_seq_len)

            layer_idxs = self._layer_indices(len(outputs.hidden_states))
            for i in layer_idxs:
                hs = outputs.hidden_states[i][0].float()  # (T_expanded, D) on GPU
                vec = hs[start:end].mean(dim=0)
                collected.setdefault(i, []).append(vec.cpu().numpy())

            del outputs

        return {k: np.stack(v, axis=0) for k, v in collected.items()}


# ---------------------------------------------------------------------------
# Mean-diff direction
# ---------------------------------------------------------------------------

def mean_diff_direction(safe: np.ndarray, harmful: np.ndarray) -> np.ndarray:
    """(mean_safe - mean_harmful), L2-normalised. safe/harmful: (N, D)"""
    d = safe.mean(axis=0) - harmful.mean(axis=0)
    norm = np.linalg.norm(d)
    if norm < 1e-12:
        raise ValueError("Direction vector is near-zero.")
    return d / norm


# ---------------------------------------------------------------------------
# Per-layer cosine similarity
# ---------------------------------------------------------------------------

def compute_per_category_cosine(
    text_safe_by_cat: dict[str, dict[int, np.ndarray]],
    text_harmful_by_cat: dict[str, dict[int, np.ndarray]],
    img_safe_by_cat: dict[str, dict[int, np.ndarray]],
    img_harmful_by_cat: dict[str, dict[int, np.ndarray]],
) -> tuple[dict[str, dict[int, float]], np.ndarray, list[str]]:
    """
    Compute per-category, per-layer text vs image cosine similarity.

    Returns:
        results: {category: {layer: cos_sim}}
        matrix: (n_cat, n_layer) array for heatmap
        cat_labels: sorted category names
    """
    all_categories = sorted(set(text_safe_by_cat.keys()) & set(text_harmful_by_cat.keys()) &
                            set(img_safe_by_cat.keys()) & set(img_harmful_by_cat.keys()))

    shared_layers = sorted(
        set(text_safe_by_cat[all_categories[0]].keys()) &
        set(text_harmful_by_cat[all_categories[0]].keys()) &
        set(img_safe_by_cat[all_categories[0]].keys()) &
        set(img_harmful_by_cat[all_categories[0]].keys())
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
            tag = "ALIGNED" if cos > 0.5 else ("ORTHO" if abs(cos) < 0.2 else "PARTIAL")
            if layer_idx % 5 == 0:  # print every 5th layer to avoid spam
                print(f"  [{cat:20s}] Layer {layer:3d} | cos={cos:+.4f}  {tag}")

    return results, matrix, all_categories


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_category_heatmap(matrix: np.ndarray, categories: list[str], layers: list[int], save_path: Path):
    """
    Plot category × layer heatmap of text vs image cosine similarity.

    Parameters
    ----------
    matrix : (n_cat, n_layer) array
    categories : category names
    layers : layer indices
    """
    fig, ax = plt.subplots(figsize=(14, max(4, len(categories) * 0.3)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)

    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, fontsize=8)
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories, fontsize=9)
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Category", fontsize=11)
    ax.set_title("VSCBench Cross-Modal Refusal Direction Alignment\n(Text vs Image per Category per Layer)", fontsize=12)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Cosine similarity", fontsize=10)

    # Add text annotations for each cell
    for i in range(len(categories)):
        for j in range(len(layers)):
            text = ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                          color="white" if abs(matrix[i, j]) > 0.5 else "black", fontsize=7)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved to {save_path}")


def print_report(results: dict[str, dict[int, float]]):
    print("\n" + "=" * 60)
    print("VSCBench: Text vs Image Refusal Direction per Category")
    print("=" * 60)
    for cat in sorted(results.keys()):
        cos_vals = list(results[cat].values())
        mean_cos = np.mean(cos_vals)
        print(f"[{cat:20s}] Mean cos = {mean_cos:+.4f}  Std = {np.std(cos_vals):.4f}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_image_states(npz_path: Path) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    data = np.load(npz_path)
    img_safe, img_harmful = {}, {}
    for key in data.files:
        if key.startswith("img_safe_layer"):
            img_safe[int(key[len("img_safe_layer"):])] = data[key]
        elif key.startswith("img_harmful_layer"):
            img_harmful[int(key[len("img_harmful_layer"):])] = data[key]
    print(f"[Cache] Loaded image hidden states from {npz_path} "
          f"({len(img_safe)} layers)")
    return img_safe, img_harmful


def main():
    assert LLAVA_MODEL_PATH,   "Fill in LLAVA_MODEL_PATH"
    assert VSC_CAPTIONS_JSONL, "Fill in VSC_CAPTIONS_JSONL"
    assert VSC_IMAGE_DIR,      "Fill in VSC_IMAGE_DIR"
    assert OUTPUT_DIR,         "Fill in OUTPUT_DIR"

    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    image_root = Path(VSC_IMAGE_DIR)
    img_npz = out / "img_hidden_states.npz"

    safe_texts_by_cat, harmful_texts_by_cat, safe_imgs_by_cat, harmful_imgs_by_cat = load_vsc(VSC_CAPTIONS_JSONL)

    # Flatten for image extraction (all categories mixed)
    all_categories = sorted(safe_texts_by_cat.keys())
    safe_texts_flat = [t for cat in all_categories for t in safe_texts_by_cat[cat]]
    harmful_texts_flat = [t for cat in all_categories for t in harmful_texts_by_cat[cat]]
    safe_imgs_flat = [p for cat in all_categories for p in safe_imgs_by_cat[cat]]
    harmful_imgs_flat = [p for cat in all_categories for p in harmful_imgs_by_cat[cat]]

    extractor = VSCExtractor(
        LLAVA_MODEL_PATH,
        layers=LAYERS,
        image_token_index=IMAGE_TOKEN_INDEX,
    )

    # ── Image hidden states: extract once, cache to disk ──
    if img_npz.exists():
        print(f"\n[Image] Cache found at {img_npz}, skipping extraction.")
        img_safe_flat, img_harmful_flat = _load_image_states(img_npz)
    else:
        print("\n[Image] Extracting safe image hidden states (image tokens, mean pool) ...")
        img_safe_flat    = extractor.extract_images(safe_imgs_flat,    image_root)
        print("[Image] Extracting harmful image hidden states (image tokens, mean pool) ...")
        img_harmful_flat = extractor.extract_images(harmful_imgs_flat, image_root)
        np.savez(
            img_npz,
            **{f"img_safe_layer{k}":    v for k, v in img_safe_flat.items()},
            **{f"img_harmful_layer{k}": v for k, v in img_harmful_flat.items()},
        )
        print(f"[Saved] {img_npz}")

    import json as _json
    import gc

    def _save_text_states(states: dict[int, np.ndarray], path: Path):
        np.savez(path, **{f"layer{k}": v for k, v in states.items()})

    def _load_text_states(path: Path) -> dict[int, np.ndarray]:
        data = np.load(path)
        return {int(k[len("layer"):]): data[k] for k in data.files}

    # ── Extract text hidden states per category ──
    text_safe_by_cat: dict[str, dict[int, np.ndarray]] = {}
    text_harmful_by_cat: dict[str, dict[int, np.ndarray]] = {}

    for cat in all_categories:
        print(f"\n[{cat}] Extracting text hidden states ...")
        del img_safe_flat, img_harmful_flat
        gc.collect()

        text_safe_by_cat[cat]    = extractor.extract_texts(safe_texts_by_cat[cat])
        text_harmful_by_cat[cat] = extractor.extract_texts(harmful_texts_by_cat[cat])

        gc.collect()
        torch.cuda.empty_cache()

        img_safe_flat, img_harmful_flat = _load_image_states(img_npz)

    # ── Partition image states by category ──
    img_safe_by_cat: dict[str, dict[int, np.ndarray]] = {}
    img_harmful_by_cat: dict[str, dict[int, np.ndarray]] = {}

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

    # ── Compute per-category cosine similarity ──
    results, matrix, cat_labels = compute_per_category_cosine(
        text_safe_by_cat, text_harmful_by_cat,
        img_safe_by_cat, img_harmful_by_cat
    )

    print_report(results)

    layers = sorted(list(results[all_categories[0]].keys()))
    plot_category_heatmap(matrix, cat_labels, layers, out / "vsc_heatmap.png")

    with open(out / "cosine_results.json", "w") as f:
        _json.dump({cat: {str(layer): cos for layer, cos in results[cat].items()}
                   for cat in results.keys()}, f, indent=2)
    print(f"[Saved] cosine_results.json -> {out}")


if __name__ == "__main__":
    main()
