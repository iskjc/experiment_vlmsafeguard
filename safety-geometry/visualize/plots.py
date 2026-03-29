"""
Visualization for safety geometry experiments.

Plots produced:
  1. layer_accuracy.png   — linear probe accuracy per layer
  2. projection_{layer}.png — distribution of PLS scores: text safe/harmful + images
  3. boundary_{layer}.png   — boundary sample projections (zoomed in near 0)
  4. summary_heatmap.png    — mean PLS score per group across all top layers
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from typing import Optional

sns.set_theme(style="whitegrid", font_scale=1.1)
COLORS = {
    "text_safe": "#2ecc71",
    "text_harm": "#e74c3c",
    "img_safe":  "#3498db",
    "img_harm":  "#e67e22",
    "boundary":  "#9b59b6",
}


def plot_layer_accuracy(
    layer_accuracy: dict[int, float],
    top_layers: list[int],
    save_dir: str,
) -> None:
    layers = sorted(layer_accuracy)
    accs = [layer_accuracy[l] for l in layers]

    fig, ax = plt.subplots(figsize=(12, 4))
    bars = ax.bar(layers, accs, color="#95a5a6", width=0.7)
    for l in top_layers:
        idx = layers.index(l)
        bars[idx].set_color("#e74c3c")

    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8, label="Chance")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("CV accuracy (safe vs harmful)")
    ax.set_title("Linear probe accuracy per layer\n(red = selected safety layers)")
    ax.legend()
    fig.tight_layout()
    path = Path(save_dir) / "layer_accuracy.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Plot] Saved {path}")


def plot_projection(
    layer_idx: int,
    text_scores: np.ndarray,
    text_labels: np.ndarray,
    image_scores: Optional[np.ndarray],
    image_labels: Optional[np.ndarray],
    save_dir: str,
    pooling: str = "last",
) -> None:
    """
    KDE + rug plot of PLS projection scores.
    Left of 0 = harmful text region; right = safe text region.
    Shows where image hidden states land.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    # ---- Left panel: KDE distributions ----
    ax = axes[0]
    t_safe = text_scores[text_labels == 1]
    t_harm = text_scores[text_labels == 0]

    sns.kdeplot(t_safe, ax=ax, color=COLORS["text_safe"], fill=True, alpha=0.35, label=f"Text safe (n={len(t_safe)})")
    sns.kdeplot(t_harm, ax=ax, color=COLORS["text_harm"], fill=True, alpha=0.35, label=f"Text harmful (n={len(t_harm)})")

    if image_scores is not None:
        i_harm = image_scores[image_labels == 0] if image_labels is not None else image_scores
        i_safe_mask = (image_labels == 1) if image_labels is not None else np.zeros(len(image_scores), bool)
        i_safe = image_scores[i_safe_mask]

        if len(i_harm) > 1:
            sns.kdeplot(i_harm, ax=ax, color=COLORS["img_harm"], fill=True, alpha=0.5,
                        linestyle="--", label=f"Image harmful (n={len(i_harm)})")
        if len(i_safe) > 1:
            sns.kdeplot(i_safe, ax=ax, color=COLORS["img_safe"], fill=True, alpha=0.5,
                        linestyle="--", label=f"Image safe (n={len(i_safe)})")

    ax.axvline(0, color="black", linestyle=":", linewidth=1.2, label="Decision boundary")
    ax.set_xlabel("PLS projection score  (+ = safe side)")
    ax.set_title(f"Layer {layer_idx} | pooling={pooling}")
    ax.legend(fontsize=9)

    # ---- Right panel: strip/swarm for images only ----
    ax2 = axes[1]
    if image_scores is not None and image_labels is not None:
        for lbl, name, color in [(0, "img_harm", COLORS["img_harm"]), (1, "img_safe", COLORS["img_safe"])]:
            s = image_scores[image_labels == lbl]
            y = np.random.uniform(-0.2, 0.2, size=len(s))
            ax2.scatter(s, y, color=color, alpha=0.5, s=20, label=name)
        ax2.axvline(0, color="black", linestyle=":", linewidth=1.2)
        ax2.axvspan(text_scores[text_labels == 0].min(), text_scores[text_labels == 0].max(),
                    alpha=0.07, color=COLORS["text_harm"], label="Text harmful range")
        ax2.set_xlabel("PLS projection score")
        ax2.set_yticks([])
        ax2.set_title("Image projections in text safety space")
        ax2.legend(fontsize=9)
    else:
        ax2.set_visible(False)

    fig.suptitle(f"Safe/Harmful Geometry — Layer {layer_idx}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = Path(save_dir) / f"projection_layer{layer_idx}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Plot] Saved {path}")


def plot_boundary_analysis(
    layer_idx: int,
    text_scores: np.ndarray,
    text_labels: np.ndarray,
    image_scores: Optional[np.ndarray],
    image_labels: Optional[np.ndarray],
    threshold: float,
    save_dir: str,
) -> None:
    """
    Zoom into the boundary region [-threshold, +threshold].
    Boundary samples are the most ambiguous — their positions are strongest evidence.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    def _strip(scores, labels, label_val, color, name):
        mask = (labels == label_val) & (np.abs(scores) < threshold)
        s = scores[mask]
        y = np.random.uniform(-0.3, 0.3, size=len(s))
        ax.scatter(s, y, color=color, alpha=0.6, s=30, label=f"{name} (n={len(s)})")

    _strip(text_scores, text_labels, 1, COLORS["text_safe"], "text_safe")
    _strip(text_scores, text_labels, 0, COLORS["text_harm"], "text_harm")
    if image_scores is not None and image_labels is not None:
        _strip(image_scores, image_labels, 1, COLORS["img_safe"], "img_safe")
        _strip(image_scores, image_labels, 0, COLORS["img_harm"], "img_harm")

    ax.axvline(0, color="black", linestyle=":", linewidth=1.5)
    ax.axvspan(-threshold, threshold, alpha=0.05, color="gray", label="Boundary region")
    ax.set_xlabel("PLS projection score")
    ax.set_yticks([])
    ax.set_title(f"Boundary samples (|score| < {threshold}) — Layer {layer_idx}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = Path(save_dir) / f"boundary_layer{layer_idx}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Plot] Saved {path}")


def plot_summary_heatmap(
    stats_per_layer: dict[int, dict],
    save_dir: str,
) -> None:
    """
    Heatmap: rows = groups (text_safe, text_harm, img_safe, img_harm),
             cols = layers, values = mean PLS score.
    """
    layers = sorted(stats_per_layer)
    keys = ["safe_mean", "harm_mean", "img_safe_mean", "img_harm_mean"]
    display = ["Text safe", "Text harmful", "Image safe", "Image harmful"]

    data = np.full((len(keys), len(layers)), np.nan)
    for j, l in enumerate(layers):
        s = stats_per_layer[l]
        for i, k in enumerate(keys):
            if k in s:
                data[i, j] = s[k]

    fig, ax = plt.subplots(figsize=(max(6, len(layers) * 1.2), 4))
    sns.heatmap(
        data,
        ax=ax,
        xticklabels=layers,
        yticklabels=display,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        linewidths=0.5,
    )
    ax.set_xlabel("Layer")
    ax.set_title("Mean PLS projection score per group and layer\n(green=safe side, red=harmful side)")
    fig.tight_layout()
    path = Path(save_dir) / "summary_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Plot] Saved {path}")
