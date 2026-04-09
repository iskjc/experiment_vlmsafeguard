"""
Replot logit lens curves from saved curves.json without rerunning the model.

Usage:
    python replot.py --output_dir ./logit_lens_outputs
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CONDITION_STYLES = {
    "text_harmful":  {"color": "#d62728", "linestyle": "-",  "label": "Text Harmful"},
    "image_harmful": {"color": "#ff7f0e", "linestyle": "--", "label": "Image Harmful"},
    "text_benign":   {"color": "#2ca02c", "linestyle": "-",  "label": "Text Benign"},
    "image_benign":  {"color": "#1f77b4", "linestyle": "--", "label": "Image Benign"},
}


def load_curves(output_dir: Path) -> dict[str, np.ndarray]:
    json_path = output_dir / "curves.json"
    with open(json_path, "r") as f:
        raw = json.load(f)
    return {k: np.array(v) for k, v in raw.items()}


def plot_linear(curves: dict[str, np.ndarray], output_path: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    for condition, curve in curves.items():
        style = CONDITION_STYLES[condition]
        x = np.arange(len(curve))
        ax.plot(x, curve, label=style["label"], color=style["color"],
                linestyle=style["linestyle"], linewidth=2, marker="o", markersize=3)
    ax.set_xlabel("Layer", fontsize=13)
    ax.set_ylabel("Refusal Token Probability", fontsize=13)
    ax.set_title("Safety Logit Lens: Refusal Signal Across Layers", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(len(c) for c in curves.values()) - 1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"[*] Linear plot saved: {output_path}")
    plt.close()


def plot_log(curves: dict[str, np.ndarray], output_path: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    for condition, curve in curves.items():
        style = CONDITION_STYLES[condition]
        x = np.arange(len(curve))
        ax.plot(x, curve + 1e-6, label=style["label"], color=style["color"],
                linestyle=style["linestyle"], linewidth=2, marker="o", markersize=3)
    ax.set_yscale("log")
    ax.set_xlabel("Layer", fontsize=13)
    ax.set_ylabel("Refusal Token Probability (log scale)", fontsize=13)
    ax.set_title("Safety Logit Lens (Log Scale)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xlim(0, max(len(c) for c in curves.values()) - 1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"[*] Log-scale plot saved: {output_path}")
    plt.close()


def plot_zoom(curves: dict[str, np.ndarray], output_path: str, layer_start: int = 20):
    """Zoom into later layers where text_harmful rises."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for condition, curve in curves.items():
        style = CONDITION_STYLES[condition]
        x = np.arange(len(curve))
        mask = x >= layer_start
        ax.plot(x[mask], curve[mask], label=style["label"], color=style["color"],
                linestyle=style["linestyle"], linewidth=2, marker="o", markersize=4)
    ax.set_xlabel("Layer", fontsize=13)
    ax.set_ylabel("Refusal Token Probability", fontsize=13)
    ax.set_title(f"Safety Logit Lens (Layers {layer_start}–32)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"[*] Zoom plot saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./logit_lens_outputs")
    parser.add_argument("--zoom_from", type=int, default=20,
                        help="Start layer for zoom plot")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    curves = load_curves(output_dir)

    print("[*] Loaded curves:")
    for cond, curve in curves.items():
        peak_layer = curve.argmax()
        print(f"  {cond:15s}: peak_layer={peak_layer}, peak={curve.max():.4f}, final={curve[-1]:.4f}")

    plot_linear(curves, str(output_dir / "logit_lens.png"))
    plot_log(curves, str(output_dir / "logit_lens_log.png"))
    plot_zoom(curves, str(output_dir / "logit_lens_zoom.png"), layer_start=args.zoom_from)


if __name__ == "__main__":
    main()
