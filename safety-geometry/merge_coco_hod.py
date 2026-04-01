"""
Merge COCO safe images with HoD harmful images, extract hidden states, and recompute cosine similarity.

This bridges the gap:
  - HoD = 2000 harmful images (already extracted)
  - COCO = ~2000 safe images (new, to extract)
  - Merged = balanced dataset for cross-modal cosine similarity analysis

Usage:
    python merge_coco_hod.py \\
        --coco-root /path/to/COCO2014/val2014 \\
        --hod-cache outputs/cache \\
        --n-coco-samples 2000 \\
        --output-cache outputs/cache_merged

Then run cosine similarity on merged cache:
    python -m analysis.cosine_similarity --cache-dir outputs/cache_merged
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import yaml

from extract.hidden_states import load_model, HiddenStateExtractor
from data.coco_safe import load_coco_safe


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_cached_states(prefix: str, cache_dir: Path) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Load cached hidden states and labels."""
    labels = np.load(cache_dir / f"{prefix}_labels.npy")
    states = {}
    for f in sorted(cache_dir.glob(f"{prefix}_layer*.npy")):
        layer_idx = int(f.stem.replace(f"{prefix}_layer", ""))
        states[layer_idx] = np.load(f)
    print(f"[Cache] Loaded {prefix}: {len(states)} layers, {labels.shape[0]} samples")
    return states, labels


def save_cached_states(prefix: str, states: dict[int, np.ndarray], labels: np.ndarray, cache_dir: Path) -> None:
    """Save hidden states and labels."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / f"{prefix}_labels.npy", labels)
    for layer_idx, arr in states.items():
        np.save(cache_dir / f"{prefix}_layer{layer_idx}.npy", arr)
    print(f"[Cache] Saved {prefix} → {cache_dir} ({len(states)} layers, {labels.shape[0]} samples)")


def merge_states(
    states1: dict[int, np.ndarray],
    labels1: np.ndarray,
    states2: dict[int, np.ndarray],
    labels2: np.ndarray,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Concatenate two sets of hidden states and labels."""
    shared_layers = set(states1.keys()) & set(states2.keys())
    if not shared_layers:
        raise ValueError("No overlapping layers between the two datasets")

    merged_states = {}
    for layer_idx in sorted(shared_layers):
        merged_states[layer_idx] = np.vstack([states1[layer_idx], states2[layer_idx]])

    merged_labels = np.concatenate([labels1, labels2])
    return merged_states, merged_labels


def main():
    parser = argparse.ArgumentParser(
        description="Extract COCO images, merge with HoD, save merged cache."
    )
    parser.add_argument("--config", default="config.yaml", help="Config file with model/extraction settings")
    parser.add_argument("--coco-root", required=True, help="Path to COCO2014 split (e.g., val2014)")
    parser.add_argument("--hod-cache", default="outputs/cache", help="Existing HoD cache dir")
    parser.add_argument("--n-coco-samples", type=int, default=360, help="Number of COCO images to sample")
    parser.add_argument("--output-cache", default="outputs/cache_merged", help="Output cache directory")
    parser.add_argument("--pooling", choices=["last", "mean"], default=None,
                        help="Override pooling strategy (default: use config.yaml)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    

    # Override pooling if specified
    if args.pooling:
        cfg["extraction"]["pooling"] = args.pooling

    # ------------------------------------------------------------------ #
    # Load model
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print(f"Loading model for hidden state extraction (pooling={cfg['extraction']['pooling']})...")
    print("=" * 60)
    model, processor = load_model(
        cfg["model"]["name"],
        device=cfg["model"]["device"],
        dtype=cfg["model"]["dtype"],
    )
    extractor = HiddenStateExtractor(
        model=model,
        processor=processor,
        device=cfg["model"]["device"],
        pooling=cfg["extraction"]["pooling"],
        batch_size=cfg["extraction"]["batch_size"],
        layers=cfg["extraction"]["layers"],
    )

    # ------------------------------------------------------------------ #
    # Load and extract COCO
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("STEP 1: Load COCO safe images and extract hidden states")
    print("=" * 60)
    coco_images, coco_prompts, coco_labels = load_coco_safe(
        args.coco_root,
        n_samples=args.n_coco_samples,
        prompt_template=cfg["data"]["visual_harm"]["prompt_template"],
    )
    coco_states = extractor.extract_image(coco_images, coco_prompts, coco_labels)
    print(f"[COCO] Extracted hidden states: {len(coco_states)} layers")

    # ------------------------------------------------------------------ #
    # Load cached HoD
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("STEP 2: Load cached HoD harmful images")
    print("=" * 60)
    hod_cache_dir = Path(args.hod_cache)
    hod_states, hod_labels = load_cached_states("image", hod_cache_dir)

    # ------------------------------------------------------------------ #
    # Merge
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("STEP 3: Merge COCO + HoD")
    print("=" * 60)
    print(f"COCO:  {(coco_labels == 1).sum()} safe images")
    print(f"HoD:   {(hod_labels == 0).sum()} harmful images")

    merged_states, merged_labels = merge_states(coco_states, coco_labels, hod_states, hod_labels)
    print(f"Merged: {(merged_labels == 1).sum()} safe + {(merged_labels == 0).sum()} harmful")
    print(f"Total: {len(merged_labels)} samples across {len(merged_states)} layers")

    # ------------------------------------------------------------------ #
    # Save
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("STEP 4: Save merged cache")
    print("=" * 60)
    output_cache = Path(args.output_cache)
    save_cached_states("image", merged_states, merged_labels, output_cache)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"\nNext: compute cosine similarity on merged data:")
    print(f"  python -m analysis.cosine_similarity --cache-dir {args.output_cache}")


if __name__ == "__main__":
    main()
