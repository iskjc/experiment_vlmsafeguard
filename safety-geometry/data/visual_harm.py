"""
Loader for visual harm datasets (HOD / ToViLaG or similar).

Expected local format (data/visual_harm/):
  labels.csv   ← columns: image_path (relative), label (0=safe / 1=harmful)
  images/      ← image files referenced in labels.csv

If your dataset has a different structure, adapt load_visual_harm() below.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from typing import Tuple


def load_visual_harm(
    data_path: str,
    n_samples: int = 200,
    image_col: str = "image_path",
    label_col: str = "label",
    prompt_template: str = "Describe what you see in this image.",
    seed: int = 42,
) -> Tuple[list[Image.Image], list[str], np.ndarray]:
    """
    Returns
    -------
    images  : list of PIL.Image
    prompts : list of str (same prompt for every image)
    labels  : np.ndarray of int, 1=harmful, 0=safe
    """
    root = Path(data_path)
    csv_path = root / "labels.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Expected {csv_path}. "
            "Place your dataset labels at data/visual_harm/labels.csv"
        )

    df = pd.read_csv(csv_path)
    rng = np.random.default_rng(seed)

    # Balance if both classes present; otherwise take what we have
    safe_idx = df[df[label_col] == 0].index.tolist()
    harm_idx = df[df[label_col] == 1].index.tolist()

    if safe_idx and harm_idx:
        n = min(n_samples, len(safe_idx), len(harm_idx))
        safe_idx = rng.choice(safe_idx, n, replace=False).tolist()
        harm_idx = rng.choice(harm_idx, n, replace=False).tolist()
        selected = df.loc[safe_idx + harm_idx].reset_index(drop=True)
    else:
        n = min(n_samples, len(df))
        selected = df.sample(n, random_state=seed).reset_index(drop=True)

    images, prompts, valid_labels = [], [], []
    for _, row in selected.iterrows():
        img_path = root / row[image_col]
        if not img_path.exists():
            print(f"[WARN] Missing image: {img_path}, skipping")
            continue
        images.append(Image.open(img_path).convert("RGB"))
        prompts.append(prompt_template)
        valid_labels.append(int(row[label_col]))

    labels = np.array(valid_labels)
    print(f"[VisualHarm] Loaded {(labels==0).sum()} safe + {(labels==1).sum()} harmful images")
    return images, prompts, labels
