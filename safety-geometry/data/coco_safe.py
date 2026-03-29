"""
Loader for COCO2014 as "safe" images.

COCO images are benign, everyday scenes (people, animals, food, sports, etc.)
and serve as a safe reference dataset to contrast with harmful datasets like HoD.

Usage:
    images, prompts, labels = load_coco_safe(
        "/path/to/COCO2014/val2014",
        n_samples=360,
        prompt_template="What is shown in this image?"
    )
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from PIL import Image
from typing import Tuple


def load_coco_safe(
    coco_root: str,
    n_samples: int = 360,
    prompt_template: str = "What is shown in this image?",
    seed: int = 42,
) -> Tuple[list[Image.Image], list[str], np.ndarray]:
    """
    Load random COCO images and label them as safe (1).

    Parameters
    ----------
    coco_root : str
        Path to COCO split directory, e.g., "/path/to/COCO2014/val2014"
    n_samples : int
        Number of images to randomly sample
    prompt_template : str
        Prompt shown to the model alongside each image
    seed : int
        Random seed for reproducibility

    Returns
    -------
    images  : list of PIL.Image (RGB)
    prompts : list of str (same prompt for every image)
    labels  : np.ndarray of int, all zeros (label=1 → safe)
    """
    root = Path(coco_root)
    if not root.exists():
        raise FileNotFoundError(f"COCO directory not found: {coco_root}")

    # Collect all image files
    image_files = list(root.glob("COCO_*.jpg")) + list(root.glob("*.jpg"))
    if not image_files:
        raise ValueError(f"No .jpg images found in {coco_root}")

    print(f"[COCO] Found {len(image_files)} images in {coco_root}")

    # Sample
    rng = np.random.default_rng(seed)
    n = min(n_samples, len(image_files))
    sampled_files = rng.choice(image_files, n, replace=False)

    images, prompts, valid_labels = [], [], []
    for img_path in sampled_files:
        try:
            img = Image.open(img_path).convert("RGB")
            images.append(img)
            prompts.append(prompt_template)
            valid_labels.append(1)  # label=1 → safe
        except Exception as e:
            print(f"[WARN] Failed to load {img_path}: {e}, skipping")

    labels = np.array(valid_labels)
    print(f"[COCO] Loaded {len(images)} safe images (label=1)")
    return images, prompts, labels
