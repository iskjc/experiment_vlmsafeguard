"""
PLS-based safe/harmful direction finder.

Why PLS over PCA:
  PCA finds directions of maximum variance in X.
  PLS finds directions of maximum covariance between X (hidden states)
  and Y (safe/harmful label) — directly optimizing for the safety-relevant axis.

Pipeline per layer:
  1. Fit PLSRegression on TEXT hidden states + safety labels.
  2. Extract the first PLS component as the "safety direction" vector w.
  3. Project both text and image hidden states onto w: score = X @ w
  4. The sign convention: positive score → safe side.

Key design choice:
  The direction is always fit on TEXT data only.
  Image hidden states are projected onto this text-derived direction — this is
  intentional, because we want to measure where images land in the text safety space.
"""

from __future__ import annotations
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PLSResult:
    layer_idx: int
    direction: np.ndarray            # (D,) unit vector
    scaler: StandardScaler
    pls_model: PLSRegression

    # Projections (set after calling project_*)
    text_scores: Optional[np.ndarray] = None   # (N_text,)
    text_labels: Optional[np.ndarray] = None
    image_scores: Optional[np.ndarray] = None  # (N_img,)
    image_labels: Optional[np.ndarray] = None

    # Derived stats
    stats: dict = field(default_factory=dict)


class PLSDirectionFinder:
    def __init__(self, n_components: int = 5, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.results_: dict[int, PLSResult] = {}

    # ------------------------------------------------------------------
    # Step 1: Fit on text data
    # ------------------------------------------------------------------

    def fit_text(
        self,
        text_states: dict[int, np.ndarray],
        text_labels: np.ndarray,
    ) -> "PLSDirectionFinder":
        """
        Fit PLS for each layer using text hidden states + safety labels.
        Labels: 1=safe, 0=harmful.
        """
        for layer_idx, states in text_states.items():
            scaler = StandardScaler()
            X = scaler.fit_transform(states)
            Y = text_labels.reshape(-1, 1).astype(float)

            pls = PLSRegression(n_components=self.n_components)
            pls.fit(X, Y)

            # First PLS component in original (scaled) space
            direction = pls.x_weights_[:, 0]           # (D,)
            direction = direction / (np.linalg.norm(direction) + 1e-12)

            result = PLSResult(
                layer_idx=layer_idx,
                direction=direction,
                scaler=scaler,
                pls_model=pls,
            )

            # Project text states immediately
            scores = X @ direction
            # Ensure positive = safe (flip if needed)
            if np.mean(scores[text_labels == 1]) < np.mean(scores[text_labels == 0]):
                direction = -direction
                scores = -scores
                result.direction = direction

            result.text_scores = scores
            result.text_labels = text_labels
            result.stats = self._compute_stats(scores, text_labels)
            self.results_[layer_idx] = result
            print(
                f"[PLS] Layer {layer_idx:3d} | "
                f"safe_mean={result.stats['safe_mean']:+.3f} "
                f"harm_mean={result.stats['harm_mean']:+.3f} "
                f"separation={result.stats['separation']:.3f}"
            )

        return self

    # ------------------------------------------------------------------
    # Step 2: Project image states onto text-derived direction
    # ------------------------------------------------------------------

    def project_images(
        self,
        image_states: dict[int, np.ndarray],
        image_labels: np.ndarray,
    ) -> "PLSDirectionFinder":
        """
        Project image hidden states onto the text-derived PLS direction.
        Uses the same scaler fitted on text data.
        """
        for layer_idx, states in image_states.items():
            if layer_idx not in self.results_:
                print(f"[WARN] Layer {layer_idx} has no fitted PLS — skipping")
                continue

            result = self.results_[layer_idx]
            X_scaled = result.scaler.transform(states)
            scores = X_scaled @ result.direction

            result.image_scores = scores
            result.image_labels = image_labels

            img_stats = self._compute_stats(scores, image_labels)
            result.stats["img_harm_mean"] = img_stats.get("harm_mean", float("nan"))
            result.stats["img_safe_mean"] = img_stats.get("safe_mean", float("nan"))
            print(
                f"[PLS] Layer {layer_idx:3d} | "
                f"img_harm_mean={result.stats['img_harm_mean']:+.3f} "
                f"img_safe_mean={result.stats.get('img_safe_mean', float('nan')):+.3f} "
                f"  (text harm_mean={result.stats['harm_mean']:+.3f})"
            )

        return self

    # ------------------------------------------------------------------
    # Step 3: Identify boundary samples
    # ------------------------------------------------------------------

    def get_boundary_indices(
        self, layer_idx: int, threshold: float = 0.35
    ) -> dict[str, np.ndarray]:
        """
        Return indices of samples near the decision boundary (|score| < threshold).
        Boundary samples are most informative for understanding model ambiguity.
        """
        result = self.results_[layer_idx]
        out = {}
        if result.text_scores is not None:
            mask = np.abs(result.text_scores) < threshold
            out["text_boundary"] = np.where(mask)[0]
        if result.image_scores is not None:
            mask = np.abs(result.image_scores) < threshold
            out["image_boundary"] = np.where(mask)[0]
        return out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_stats(scores: np.ndarray, labels: np.ndarray) -> dict:
        stats = {}
        safe_mask = labels == 1
        harm_mask = labels == 0
        if safe_mask.any():
            stats["safe_mean"] = float(scores[safe_mask].mean())
            stats["safe_std"] = float(scores[safe_mask].std())
        if harm_mask.any():
            stats["harm_mean"] = float(scores[harm_mask].mean())
            stats["harm_std"] = float(scores[harm_mask].std())
        if safe_mask.any() and harm_mask.any():
            num = abs(stats["safe_mean"] - stats["harm_mean"])
            denom = (stats["safe_std"] + stats["harm_std"]) / 2 + 1e-12
            stats["separation"] = float(num / denom)  # Cohen's d proxy
        return stats
