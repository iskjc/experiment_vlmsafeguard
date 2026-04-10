"""
Idea 2: Category-Dependent Safety Geometry (multi-model support)
================================================================
VSCBench only. Computes per-category mean-diff refusal directions for all layers
and plots category×category cosine-similarity heatmaps.

Outputs (saved to OUTPUT_DIR):
  VSCBench_text_layer_heatmaps/layer_XX.png   -- text modality, per layer
  VSCBench_image_layer_heatmaps/layer_XX.png  -- image modality, per layer
  VSCBench_text_summary.png                   -- mean off-diag |cos-sim| per layer
  VSCBench_image_summary.png
  directions.npz                              -- all direction vectors

Usage:
    python idea2_heatmap.py \
        --model_path /path/to/model \
        --captions_jsonl /path/to/captions.jsonl \
        --image_dir /path/to/images \
        --output_dir ./results/idea2 \
        [--family llava-next]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import numpy as np
import seaborn as sns

import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils.embed import LLaVAProber
from utils.directions import mean_diff_direction, cosine_similarity_matrix


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_vsc(jsonl_path: str) -> dict[str, dict[str, list]]:
    data: dict[str, dict] = defaultdict(lambda: {
        "harmful_texts": [], "safe_texts": [],
        "harmful_image_paths": [], "safe_image_paths": []
    })
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            cat = obj.get("category", "unknown")
            if obj.get("harmful_description"):
                data[cat]["harmful_texts"].append(obj["harmful_description"])
            if obj.get("safe_description"):
                data[cat]["safe_texts"].append(obj["safe_description"])
            if obj.get("harmful_image_path"):
                data[cat]["harmful_image_paths"].append(obj["harmful_image_path"])
            if obj.get("safe_image_path"):
                data[cat]["safe_image_paths"].append(obj["safe_image_path"])
    return dict(data)


# ---------------------------------------------------------------------------
# Heatmap plotting
# ---------------------------------------------------------------------------

def plot_heatmap(cos_sim: np.ndarray, labels: list[str],
                 title: str, save_path: Path, annot: bool = True):
    fig, ax = plt.subplots(figsize=(max(6, len(labels)), max(5, len(labels) - 1)))
    sns.heatmap(
        cos_sim, annot=annot, fmt=".2f",
        xticklabels=labels, yticklabels=labels,
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        linewidths=0.5, ax=ax,
    )
    ax.set_title(title, fontsize=13, pad=12)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def offdiag_abs_cosine(cos_sim: np.ndarray) -> list[float]:
    n = cos_sim.shape[0]
    return [abs(float(cos_sim[i, j]))
            for i in range(n) for j in range(i + 1, n)]


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_heatmaps_all_layers(
    encoder: LLaVAProber,
    data: dict,
    image_root: Path | None,
    dataset_name: str,
    output_dir: Path,
    modality: str,
) -> dict[str, dict[int, np.ndarray]]:
    print(f"\n[{dataset_name}/{modality}] Extracting hidden states for all layers ...")
    safe_by_cat: dict[str, dict[int, np.ndarray]] = {}
    harm_by_cat: dict[str, dict[int, np.ndarray]] = {}

    for cat, samples in data.items():
        if modality == "text":
            harm_list = samples.get("harmful_texts", [])
            safe_list = samples.get("safe_texts", [])
            if len(harm_list) < 2 or len(safe_list) < 2:
                print(f"  [{cat}] too few samples, skipped")
                continue
            print(f"  [{cat}] encoding {len(safe_list)} safe + {len(harm_list)} harmful texts ...")
            safe_by_cat[cat] = encoder.encode_texts_all_layers(safe_list)
            harm_by_cat[cat] = encoder.encode_texts_all_layers(harm_list)

        elif modality == "image":
            if image_root is None:
                continue
            if "harmful_image_paths" not in samples:
                print(f"  [{cat}] no safe/harmful image pairs, skipped")
                continue
            h_paths = [image_root / Path(p) for p in samples["harmful_image_paths"]]
            s_paths = [image_root / Path(p) for p in samples["safe_image_paths"]]
            if len(h_paths) < 2 or len(s_paths) < 2:
                print(f"  [{cat}] too few image samples, skipped")
                continue
            print(f"  [{cat}] encoding {len(s_paths)} safe + {len(h_paths)} harmful images ...")
            safe_by_cat[cat] = encoder.encode_image_token_pool_files_all_layers(s_paths)
            harm_by_cat[cat] = encoder.encode_image_token_pool_files_all_layers(h_paths)
        else:
            raise ValueError(f"Unknown modality: {modality}")

    if len(safe_by_cat) < 2:
        print("  Not enough categories.")
        return {}

    categories = sorted(safe_by_cat.keys())
    all_layers = sorted(next(iter(safe_by_cat.values())).keys())

    directions_by_layer: dict[int, dict[str, np.ndarray]] = {}
    mean_offdiag_per_layer: list[float] = []

    layer_heatmap_dir = output_dir / f"{dataset_name.lower()}_{modality}_layer_heatmaps"
    layer_heatmap_dir.mkdir(exist_ok=True)

    for layer in all_layers:
        dirs = {}
        for cat in categories:
            h_emb = harm_by_cat[cat][layer]
            s_emb = safe_by_cat[cat][layer]
            dirs[cat] = mean_diff_direction(h_emb, s_emb)
        directions_by_layer[layer] = dirs

        cos_sim, labels = cosine_similarity_matrix(dirs)
        od = offdiag_abs_cosine(cos_sim)
        mean_offdiag_per_layer.append(float(np.mean(od)))

        title = f"{dataset_name} {modality.capitalize()} — Layer {layer}"
        plot_heatmap(cos_sim, labels, title,
                     layer_heatmap_dir / f"layer_{layer:02d}.png", annot=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(all_layers, mean_offdiag_per_layer, marker="o", markersize=4,
            linewidth=1.5, color="#d62728")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean off-diagonal |cos-sim|")
    ax.set_title(f"{dataset_name} — {modality.capitalize()} Refusal Direction\n"
                 f"Category Similarity Across Layers")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    summary_path = output_dir / f"{dataset_name.lower()}_{modality}_summary.png"
    fig.savefig(summary_path, dpi=150)
    plt.close(fig)
    print(f"  Summary saved: {summary_path}")

    return {cat: {layer: directions_by_layer[layer][cat] for layer in all_layers}
            for cat in categories}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--family", type=str, default=None)
    parser.add_argument("--captions_jsonl", type=str, required=True)
    parser.add_argument("--image_dir", type=str, default="")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("[*] Loading prober ...")
    encoder = LLaVAProber(args.model_path, family=args.family, device=args.device)

    vsc_data = load_vsc(args.captions_jsonl)
    print(f"[*] VSCBench categories: {sorted(vsc_data.keys())}")

    vsc_root = Path(args.image_dir) if args.image_dir else None

    vsc_text_dirs = run_heatmaps_all_layers(encoder, vsc_data, vsc_root, "VSCBench", out, modality="text")
    vsc_image_dirs = run_heatmaps_all_layers(encoder, vsc_data, vsc_root, "VSCBench", out, modality="image")

    save_dict = {}
    for dataset_name, dirs_by_layer in zip(
        ["vsc_text", "vsc_image"],
        [vsc_text_dirs, vsc_image_dirs]
    ):
        for cat, layer_vecs in dirs_by_layer.items():
            for layer, vec in layer_vecs.items():
                save_dict[f"{dataset_name}/{cat}/layer{layer}"] = vec
    np.savez(out / "directions.npz", **save_dict)
    print(f"\n[*] Direction vectors saved to {out / 'directions.npz'}")
    print("[*] Done.")


if __name__ == "__main__":
    main()
