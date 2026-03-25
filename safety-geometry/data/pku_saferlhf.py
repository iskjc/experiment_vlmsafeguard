"""
Loader for PKU-SafeRLHF dataset.

Expected local format (data/pku_saferlhf/):
  train.jsonl or train.parquet  ← HF snapshot, or download via:
  huggingface-cli download PKU-Alignment/PKU-SafeRLHF --local-dir data/pku_saferlhf

Each record has: prompt, response_0, response_1,
                 is_response_0_safe, is_response_1_safe, ...
We treat each (prompt + response) pair as one sample.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple


def load_pku_saferlhf(
    data_path: str,
    n_samples: int = 500,
    text_col: str = "prompt",
    response_col: str = "response_0",
    label_col: str = "is_response_0_safe",
    seed: int = 42,
) -> Tuple[list[str], np.ndarray]:
    """
    Returns
    -------
    texts  : list of str, each is "prompt\\n\\nresponse"
    labels : np.ndarray of int, 1=safe, 0=harmful
    """
    path = Path(data_path)

    # Try common file formats
    if (path / "train.parquet").exists():
        df = pd.read_parquet(path / "train.parquet")
    elif (path / "train.jsonl").exists():
        df = pd.read_json(path / "train.jsonl", lines=True)
    elif (path / "train.csv").exists():
        df = pd.read_csv(path / "train.csv")
    else:
        # Try HuggingFace datasets
        from datasets import load_dataset
        ds = load_dataset(data_path, split="train")
        df = ds.to_pandas()

    df = df.dropna(subset=[text_col, response_col, label_col])

    # Balance classes
    rng = np.random.default_rng(seed)
    safe_idx = df[df[label_col] == True].index.tolist()
    harm_idx = df[df[label_col] == False].index.tolist()

    n = min(n_samples, len(safe_idx), len(harm_idx))
    safe_idx = rng.choice(safe_idx, n, replace=False).tolist()
    harm_idx = rng.choice(harm_idx, n, replace=False).tolist()

    selected = df.loc[safe_idx + harm_idx].reset_index(drop=True)
    texts = (selected[text_col] + "\n\n" + selected[response_col]).tolist()
    labels = selected[label_col].astype(int).values  # 1=safe, 0=harmful

    print(f"[PKU-SafeRLHF] Loaded {n} safe + {n} harmful = {2*n} samples")
    return texts, labels
