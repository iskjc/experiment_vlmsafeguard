"""
Generate merged labels.csv containing both COCO (safe) and HoD (harmful) images.

This allows run_pipeline.py to extract both modalities in one go.
"""

from pathlib import Path
import csv
import argparse


def gen_merged_labels(
    coco_root: str,
    hod_root: str = "data/HOD",
    output_csv: str = "data/visual_harm/labels.csv",
    n_coco: int = 360,
    seed: int = 42,
):
    """
    Generate merged labels.csv with COCO safe + HoD harmful.

    Parameters
    ----------
    coco_root : str
        Path to COCO2014/val2014 (e.g., /s/datasets/COCO2014/val2014)
    hod_root : str
        Path to HoD directory (e.g., data/HOD)
    output_csv : str
        Output CSV path (default: data/visual_harm/labels.csv)
    n_coco : int
        Number of COCO images to sample (default: 360)
    seed : int
        Random seed
    """

    coco_path = Path(coco_root)
    hod_path = Path(hod_root)
    output_path = Path(output_csv)

    # Collect COCO images
    coco_images = sorted(list(coco_path.glob("COCO_*.jpg")) + list(coco_path.glob("*.jpg")))
    print(f"[COCO] Found {len(coco_images)} images")

    # Sample COCO
    import numpy as np
    rng = np.random.default_rng(seed)
    n_coco = min(n_coco, len(coco_images))
    coco_sampled = rng.choice(coco_images, n_coco, replace=False)
    print(f"[COCO] Sampled {n_coco} safe images")

    # Collect HoD images
    hod_images = sorted(hod_path.glob("*/*.jpg"))
    print(f"[HoD] Found {len(hod_images)} harmful images")

    # Generate rows
    rows = []

    # COCO as safe (label=1)
    for img_path in coco_sampled:
        rows.append({
            "image_path": str(img_path),
            "label": 1,
            "category": "coco_safe"
        })

    # HoD as harmful (label=0) — use absolute paths to avoid root/ prefix issues
    for img_path in hod_images:
        rows.append({
            "image_path": str(img_path.resolve()),
            "label": 0,
            "category": img_path.parent.name  # e.g., "blood", "gun", etc.
        })

    # Write CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "label", "category"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✓ Generated {len(rows)} labels → {output_path}")
    print(f"  - COCO safe:  {n_coco} (label=1)")
    print(f"  - HoD harmful: {len(hod_images)} (label=0)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate merged COCO+HoD labels.csv")
    parser.add_argument("--coco-root", required=True, help="Path to COCO2014/val2014")
    parser.add_argument("--hod-root", default="data/HOD", help="Path to HoD directory")
    parser.add_argument("--output", default="data/visual_harm/labels.csv", help="Output CSV")
    parser.add_argument("--n-coco", type=int, default=360, help="Number of COCO samples")
    args = parser.parse_args()

    gen_merged_labels(
        coco_root=args.coco_root,
        hod_root=args.hod_root,
        output_csv=args.output,
        n_coco=args.n_coco,
    )
