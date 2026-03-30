"""
Low-rank Orthogonal Procrustes alignment of cross-modal safety subspaces.

Goal:
  Find an orthogonal rotation R that aligns the IMAGE safety subspace
  to the TEXT safety subspace so that the model's text-trained safety
  mechanisms can recognise visual harm.

Why low-rank?
  Full Procrustes on 4096-dim hidden states requires paired samples and
  is heavily underdetermined.  Instead we work in the PLS subspace:
    - Extract the top-k PLS directions from text and image separately.
    - Find R (k×k orthogonal) that maps image directions → text directions.
    - Lift back to full 4096-dim space for downstream use.

Pipeline per layer:
  1. Fit PLS on text → W_text  (D, k)   (k safety-relevant directions)
  2. Fit PLS on image → W_img  (D, k)
  3. Solve Orthogonal Procrustes: R* = argmin_R ||W_img @ R - W_text||_F  s.t. R^T R = I
     Solution: SVD of W_img^T @ W_text = U Σ V^T  →  R* = U V^T
  4. Compute aligned directions: W_img_aligned = W_img @ R*

Metrics:
  - Loss_orig  = ||W_img - W_text||_F^2         (before alignment)
  - Loss_rot   = ||W_img @ R* - W_text||_F^2    (after alignment)
  - Cosine similarity distribution before/after
  - Separation d(L) after projecting images onto aligned direction

Usage (standalone):
    python -m analysis.procrustes_align \\
        --text-cache outputs/cache \\
        --image-cache outputs/cache_merged \\
        --k 5

Usage (as module):
    from analysis.procrustes_align import LowRankProcrustes
    aligner = LowRankProcrustes(k=5)
    aligner.fit(text_states, text_labels, image_states, image_labels)
    aligner.print_report()
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler


@dataclass
class LayerAlignmentResult:
    layer_idx: int

    # Subspace bases (D, k)
    W_text: np.ndarray
    W_image: np.ndarray

    # Rotation matrix (k, k) orthogonal
    R: np.ndarray

    # Aligned image subspace (D, k)
    W_image_aligned: np.ndarray

    # Losses
    loss_orig: float      # ||W_img - W_text||_F^2
    loss_aligned: float   # ||W_img @ R - W_text||_F^2
    loss_reduction: float # (orig - aligned) / orig

    # Per-component cosine similarities (k,) — before and after
    cosine_before: np.ndarray
    cosine_after: np.ndarray

    # Separation d(L) — image projected onto aligned first direction
    separation_text: float               # text safe/harm separation on text dir
    separation_image_before: float       # image safe/harm on ORIGINAL image dir
    separation_image_aligned: float      # image safe/harm on ALIGNED dir

    # Scalers for downstream use
    text_scaler: StandardScaler
    image_scaler: StandardScaler


class LowRankProcrustes:
    """
    Low-rank Orthogonal Procrustes alignment between text and image
    safety subspaces extracted via PLS.
    """

    def __init__(self, k: int = 5, n_pls_components: int = 5):
        """
        Parameters
        ----------
        k : int
            Number of PLS directions to align. k <= n_pls_components.
        n_pls_components : int
            Number of components passed to PLSRegression.
        """
        self.k = k
        self.n_pls = max(k, n_pls_components)
        self.results_: dict[int, LayerAlignmentResult] = {}

    # ------------------------------------------------------------------ #
    # Main entry
    # ------------------------------------------------------------------ #

    def fit(
        self,
        text_states: dict[int, np.ndarray],
        text_labels: np.ndarray,
        image_states: dict[int, np.ndarray],
        image_labels: np.ndarray,
        layers: Optional[list[int]] = None,
    ) -> "LowRankProcrustes":

        shared = set(text_states.keys()) & set(image_states.keys())
        if layers is not None:
            shared &= set(layers)
        shared = sorted(shared)

        if not shared:
            raise ValueError("No overlapping layers between text and image caches.")

        print(f"\n[Procrustes] Low-rank (k={self.k}) alignment on {len(shared)} layers: {shared}")

        for layer_idx in shared:
            result = self._align_layer(
                layer_idx,
                text_states[layer_idx], text_labels,
                image_states[layer_idx], image_labels,
            )
            self.results_[layer_idx] = result

        return self

    # ------------------------------------------------------------------ #
    # Per-layer alignment
    # ------------------------------------------------------------------ #

    def _align_layer(
        self,
        layer_idx: int,
        text_states: np.ndarray,   # (N_text, D)
        text_labels: np.ndarray,
        image_states: np.ndarray,  # (N_img, D)
        image_labels: np.ndarray,
    ) -> LayerAlignmentResult:

        # 1. Fit PLS on each modality independently
        text_scaler = StandardScaler()
        X_text = text_scaler.fit_transform(text_states)
        W_text = self._extract_pls_basis(X_text, text_labels, "text")  # (D, k)

        image_scaler = StandardScaler()
        X_image = image_scaler.fit_transform(image_states)
        W_image = self._extract_pls_basis(X_image, image_labels, "image")  # (D, k)

        # 2. Orthogonal Procrustes:  min ||W_image @ R - W_text||_F  s.t. R'R=I
        #    Solution: M = W_image^T @ W_text;  SVD(M) = U Σ V^T;  R = U V^T
        M = W_image.T @ W_text    # (k, k)
        U, _, Vt = np.linalg.svd(M)
        R = U @ Vt                # (k, k) orthogonal

        W_image_aligned = W_image @ R   # (D, k)

        # 3. Losses
        loss_orig = float(np.sum((W_image - W_text) ** 2))
        loss_aligned = float(np.sum((W_image_aligned - W_text) ** 2))
        loss_reduction = (loss_orig - loss_aligned) / (loss_orig + 1e-12)

        # 4. Per-component cosine similarities
        cosine_before = np.array([
            self._cosine(W_text[:, i], W_image[:, i]) for i in range(self.k)
        ])
        cosine_after = np.array([
            self._cosine(W_text[:, i], W_image_aligned[:, i]) for i in range(self.k)
        ])

        # 5. Separation metrics
        #    Use the FIRST direction (primary safety component) for separation
        text_dir = W_text[:, 0]
        image_dir_orig = W_image[:, 0]
        image_dir_aligned = W_image_aligned[:, 0]

        sep_text = self._separation(X_text @ text_dir, text_labels)
        sep_img_before = self._separation(X_image @ image_dir_orig, image_labels)
        sep_img_aligned = self._separation(X_image @ image_dir_aligned, image_labels)

        result = LayerAlignmentResult(
            layer_idx=layer_idx,
            W_text=W_text,
            W_image=W_image,
            R=R,
            W_image_aligned=W_image_aligned,
            loss_orig=loss_orig,
            loss_aligned=loss_aligned,
            loss_reduction=loss_reduction,
            cosine_before=cosine_before,
            cosine_after=cosine_after,
            separation_text=sep_text,
            separation_image_before=sep_img_before,
            separation_image_aligned=sep_img_aligned,
            text_scaler=text_scaler,
            image_scaler=image_scaler,
        )

        print(
            f"  Layer {layer_idx:3d} | "
            f"loss: {loss_orig:.4f} → {loss_aligned:.4f} "
            f"(↓{loss_reduction*100:.1f}%)  "
            f"cos[0]: {cosine_before[0]:+.4f} → {cosine_after[0]:+.4f}  "
            f"sep: text={sep_text:.3f} img_orig={sep_img_before:.3f} img_aligned={sep_img_aligned:.3f}"
        )

        return result

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _extract_pls_basis(
        self,
        X: np.ndarray,       # (N, D) already scaled
        labels: np.ndarray,
        name: str,
    ) -> np.ndarray:
        """Fit PLS and return (D, k) orthonormal basis of top-k directions."""
        Y = labels.reshape(-1, 1).astype(float)
        pls = PLSRegression(n_components=self.n_pls)
        pls.fit(X, Y)

        W = pls.x_weights_[:, :self.k]  # (D, k)

        # Sign convention: positive = safe for each component
        for i in range(self.k):
            scores = X @ W[:, i]
            if np.mean(scores[labels == 1]) < np.mean(scores[labels == 0]):
                W[:, i] = -W[:, i]

        # Orthonormalise via QR for numerical stability
        W, _ = np.linalg.qr(W)

        return W  # (D, k)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
        return float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))

    @staticmethod
    def _separation(scores: np.ndarray, labels: np.ndarray) -> float:
        safe = scores[labels == 1]
        harm = scores[labels == 0]
        if len(safe) == 0 or len(harm) == 0:
            return 0.0
        num = abs(safe.mean() - harm.mean())
        denom = (safe.std() + harm.std()) / 2 + 1e-12
        return float(num / denom)

    # ------------------------------------------------------------------ #
    # Report
    # ------------------------------------------------------------------ #

    def print_report(self) -> None:
        if not self.results_:
            print("[Procrustes] No results.")
            return

        print("\n" + "=" * 80)
        print(f"LOW-RANK ORTHOGONAL PROCRUSTES ALIGNMENT  (k={self.k})")
        print("=" * 80)

        # Header
        print(
            f"{'Layer':>6}  "
            f"{'Loss_orig':>10} {'Loss_rot':>10} {'↓%':>6}  "
            f"{'cos_before':>11} {'cos_after':>11}  "
            f"{'d_text':>7} {'d_img_orig':>11} {'d_img_align':>12}"
        )
        print("-" * 80)

        for layer_idx in sorted(self.results_.keys()):
            r = self.results_[layer_idx]
            print(
                f"{layer_idx:>6}  "
                f"{r.loss_orig:>10.4f} {r.loss_aligned:>10.4f} {r.loss_reduction*100:>5.1f}%  "
                f"{r.cosine_before[0]:>+11.4f} {r.cosine_after[0]:>+11.4f}  "
                f"{r.separation_text:>7.3f} {r.separation_image_before:>11.3f} {r.separation_image_aligned:>12.3f}"
            )

        print("-" * 80)

        # Aggregate stats
        losses_orig = [r.loss_orig for r in self.results_.values()]
        losses_aligned = [r.loss_aligned for r in self.results_.values()]
        cos_before_1st = [r.cosine_before[0] for r in self.results_.values()]
        cos_after_1st = [r.cosine_after[0] for r in self.results_.values()]
        seps_aligned = [r.separation_image_aligned for r in self.results_.values()]

        print(f"\n  Mean loss reduction     : {np.mean([(o-a)/o for o,a in zip(losses_orig, losses_aligned)])*100:.1f}%")
        print(f"  Mean cos[0] before      : {np.mean(cos_before_1st):+.4f}")
        print(f"  Mean cos[0] after       : {np.mean(cos_after_1st):+.4f}")
        print(f"  Mean d(img) after align : {np.mean(seps_aligned):.3f}")

        print("\n  Interpretation:")
        mean_cos_after = np.mean([abs(c) for c in cos_after_1st])
        mean_reduction = np.mean([(o-a)/o for o,a in zip(losses_orig, losses_aligned)])

        if mean_cos_after > 0.8 and mean_reduction > 0.5:
            print("  >> STRONG alignment achieved.")
            print("     The low-rank rotation successfully maps image safety directions")
            print("     onto text safety directions.  This R can be used in soft-prompt")
            print("     training as an alignment loss term.")
        elif mean_cos_after > 0.5:
            print("  >> MODERATE alignment.  Rotation helps but directions remain partially")
            print("     misaligned.  A higher k or non-orthogonal transform may be needed.")
        else:
            print("  >> WEAK alignment.  The subspaces may differ in more than just rotation")
            print("     (e.g. different dimensionality or non-linear structure).")
            print("     Consider: (1) increasing k, (2) using CCA instead of PLS,")
            print("     (3) a learned linear (non-orthogonal) projection.")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _load_cache(prefix: str, cache_dir: Path) -> tuple[dict[int, np.ndarray], np.ndarray]:
    labels = np.load(cache_dir / f"{prefix}_labels.npy")
    states = {}
    for f in sorted(cache_dir.glob(f"{prefix}_layer*.npy")):
        layer_idx = int(f.stem.replace(f"{prefix}_layer", ""))
        states[layer_idx] = np.load(f)
    print(f"[Cache] Loaded {prefix}: {len(states)} layers, {labels.shape[0]} samples")
    return states, labels


def main():
    parser = argparse.ArgumentParser(
        description="Low-rank Orthogonal Procrustes alignment of safety subspaces."
    )
    parser.add_argument("--text-cache", type=str, default="outputs/cache",
                        help="Cache dir for text states (SaferLHF)")
    parser.add_argument("--image-cache", type=str, default="outputs/cache_merged",
                        help="Cache dir for image states (COCO+HoD)")
    parser.add_argument("--k", type=int, default=5,
                        help="Number of PLS directions to align")
    parser.add_argument("--layers", nargs="+", type=int, default=None,
                        help="Subset of layers (default: all overlapping)")
    args = parser.parse_args()

    text_cache = Path(args.text_cache)
    image_cache = Path(args.image_cache)

    print(f"[Procrustes] Loading text from {text_cache}")
    text_states, text_labels = _load_cache("text", text_cache)
    print(f"[Procrustes] Loading image from {image_cache}")
    image_states, image_labels = _load_cache("image", image_cache)

    aligner = LowRankProcrustes(k=args.k)
    aligner.fit(text_states, text_labels, image_states, image_labels, layers=args.layers)
    aligner.print_report()


if __name__ == "__main__":
    main()
