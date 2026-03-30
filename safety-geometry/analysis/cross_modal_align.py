"""
Cross-modal safety subspace alignment — comprehensive analysis.

Implements multiple alignment methods beyond Orthogonal Procrustes:

  1. Difference-in-Means (DiM) cosine similarity  — simplest baseline
  2. Non-orthogonal Procrustes                     — relax R^T R = I
  3. Principal Angles (Grassmann distance)          — subspace geometry
  4. Cross-projection test                          — project images onto text direction

Why these matter:
  - If Orthogonal Procrustes gives WEAK alignment (cos ≈ 0.06, loss ↓5-10%),
    it means a pure rotation can't bridge the gap.
  - DiM tells you: are the *simplest* safety directions already aligned?
  - Non-orthogonal tells you: can scaling + rotation help?
  - Principal angles tell you: what is the geometric relationship between
    the two k-dim subspaces?
  - Cross-projection tells you: does the text safety direction actually
    separate safe/harmful images?

Usage:
    python -m analysis.cross_modal_align \\
        --text-cache outputs/cache \\
        --image-cache outputs/cache \\
        --k 5
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler


# =========================================================================== #
# Data classes
# =========================================================================== #

@dataclass
class DiMResult:
    """Difference-in-Means comparison per layer."""
    layer_idx: int
    cos_sim: float          # cosine(d_text, d_img)
    abs_cos: float
    sep_text: float         # separation on d_text
    sep_img_on_text: float  # image separation on TEXT direction (cross-modal transfer)
    sep_img_on_img: float   # image separation on IMAGE direction (within-modal)


@dataclass
class NonOrthogonalResult:
    """Non-orthogonal (affine) Procrustes per layer."""
    layer_idx: int
    loss_ortho: float       # ||W_img @ R_ortho - W_text||_F^2
    loss_affine: float      # ||W_img @ A - W_text||_F^2
    loss_reduction: float   # relative improvement over orthogonal
    condition_number: float # cond(A) — how much distortion
    cos_after: float        # cosine of first component after affine


@dataclass
class PrincipalAngleResult:
    """Principal angles between text and image PLS subspaces."""
    layer_idx: int
    angles_deg: np.ndarray       # (k,) principal angles in degrees
    cosines: np.ndarray          # (k,) cos of principal angles
    grassmann_dist: float        # Grassmann distance = sqrt(sum(θ_i^2))
    chordal_dist: float          # chordal distance = sqrt(sum(sin^2(θ_i)))


@dataclass
class CrossProjectionResult:
    """Cross-projection: project images onto text safety direction."""
    layer_idx: int
    sep_text_on_text: float      # text separation on text PLS dir
    sep_img_on_text_pls: float   # image separation on text PLS dir
    sep_img_on_text_dim: float   # image separation on text DiM dir
    sep_img_on_img_pls: float    # image separation on image PLS dir
    sep_img_on_img_dim: float    # image separation on image DiM dir
    # cross-modal transfer ratio: how much text direction helps images
    transfer_ratio_pls: float    # sep_img_on_text / sep_img_on_img (PLS)
    transfer_ratio_dim: float    # sep_img_on_text / sep_img_on_img (DiM)


# =========================================================================== #
# Main analyzer
# =========================================================================== #

class CrossModalAnalyzer:
    """Comprehensive cross-modal alignment analysis."""

    def __init__(self, k: int = 5, n_pls: int = 5):
        self.k = k
        self.n_pls = max(k, n_pls)

        self.dim_results_: dict[int, DiMResult] = {}
        self.nonortho_results_: dict[int, NonOrthogonalResult] = {}
        self.angle_results_: dict[int, PrincipalAngleResult] = {}
        self.crossproj_results_: dict[int, CrossProjectionResult] = {}

    def fit(
        self,
        text_states: dict[int, np.ndarray],
        text_labels: np.ndarray,
        image_states: dict[int, np.ndarray],
        image_labels: np.ndarray,
        layers: Optional[list[int]] = None,
    ) -> "CrossModalAnalyzer":

        shared = sorted(set(text_states.keys()) & set(image_states.keys()))
        if layers:
            shared = sorted(set(shared) & set(layers))
        if not shared:
            raise ValueError("No overlapping layers.")

        print(f"\n{'='*70}")
        print(f"CROSS-MODAL ALIGNMENT ANALYSIS  (k={self.k}, layers={shared})")
        print(f"{'='*70}")

        for layer_idx in shared:
            self._analyze_layer(
                layer_idx,
                text_states[layer_idx], text_labels,
                image_states[layer_idx], image_labels,
            )

        return self

    def _analyze_layer(
        self,
        layer_idx: int,
        text_raw: np.ndarray,
        text_labels: np.ndarray,
        image_raw: np.ndarray,
        image_labels: np.ndarray,
    ):
        # Standardize
        text_scaler = StandardScaler()
        X_text = text_scaler.fit_transform(text_raw)
        image_scaler = StandardScaler()
        X_image = image_scaler.fit_transform(image_raw)

        # ---- 1. Difference-in-Means ----
        self.dim_results_[layer_idx] = self._dim_analysis(
            layer_idx, X_text, text_labels, X_image, image_labels
        )

        # ---- 2 & 3. PLS-based analyses ----
        W_text = self._extract_pls_basis(X_text, text_labels)
        W_image = self._extract_pls_basis(X_image, image_labels)

        self.nonortho_results_[layer_idx] = self._nonortho_procrustes(
            layer_idx, W_text, W_image
        )
        self.angle_results_[layer_idx] = self._principal_angles(
            layer_idx, W_text, W_image
        )

        # ---- 4. Cross-projection ----
        d_text_dim = self._dim_direction(X_text, text_labels)
        d_img_dim = self._dim_direction(X_image, image_labels)
        pls_text_dir = W_text[:, 0]
        pls_img_dir = W_image[:, 0]

        self.crossproj_results_[layer_idx] = self._cross_projection(
            layer_idx,
            X_text, text_labels,
            X_image, image_labels,
            pls_text_dir, pls_img_dir,
            d_text_dim, d_img_dim,
        )

    # =================================================================== #
    # Method 1: Difference-in-Means
    # =================================================================== #

    def _dim_analysis(self, layer_idx, X_text, text_labels, X_image, image_labels):
        d_text = self._dim_direction(X_text, text_labels)
        d_img = self._dim_direction(X_image, image_labels)

        cos = self._cosine(d_text, d_img)

        sep_text = self._separation(X_text @ d_text, text_labels)
        sep_img_on_text = self._separation(X_image @ d_text, image_labels)
        sep_img_on_img = self._separation(X_image @ d_img, image_labels)

        return DiMResult(
            layer_idx=layer_idx,
            cos_sim=cos,
            abs_cos=abs(cos),
            sep_text=sep_text,
            sep_img_on_text=sep_img_on_text,
            sep_img_on_img=sep_img_on_img,
        )

    @staticmethod
    def _dim_direction(X, labels):
        """Difference-in-means direction, normalized."""
        safe = X[labels == 1].mean(axis=0)
        harm = X[labels == 0].mean(axis=0)
        d = safe - harm
        return d / (np.linalg.norm(d) + 1e-12)

    # =================================================================== #
    # Method 2: Non-orthogonal Procrustes
    # =================================================================== #

    def _nonortho_procrustes(self, layer_idx, W_text, W_image):
        # Orthogonal Procrustes baseline
        M = W_image.T @ W_text
        U, _, Vt = np.linalg.svd(M)
        R_ortho = U @ Vt
        loss_ortho = float(np.sum((W_image @ R_ortho - W_text) ** 2))

        # Non-orthogonal: min ||W_image @ A - W_text||_F
        # Solution: A = pinv(W_image) @ W_text
        # Since W_image is (D, k) with D >> k, use: A = (W_img^T W_img)^{-1} W_img^T W_text
        gram = W_image.T @ W_image  # (k, k)
        # Regularize for stability
        gram_reg = gram + 1e-6 * np.eye(self.k)
        A = np.linalg.solve(gram_reg, W_image.T @ W_text)  # (k, k)

        W_image_affine = W_image @ A
        loss_affine = float(np.sum((W_image_affine - W_text) ** 2))

        cond = float(np.linalg.cond(A))
        loss_reduction = (loss_ortho - loss_affine) / (loss_ortho + 1e-12)

        # Cosine of first component after affine transform
        cos_after = self._cosine(W_text[:, 0], W_image_affine[:, 0])

        return NonOrthogonalResult(
            layer_idx=layer_idx,
            loss_ortho=loss_ortho,
            loss_affine=loss_affine,
            loss_reduction=loss_reduction,
            condition_number=cond,
            cos_after=cos_after,
        )

    # =================================================================== #
    # Method 3: Principal Angles between subspaces
    # =================================================================== #

    def _principal_angles(self, layer_idx, W_text, W_image):
        """
        Principal angles between two k-dim subspaces.

        Given orthonormal bases W_text (D,k) and W_image (D,k):
          cos(θ_i) = σ_i(W_text^T @ W_image)
        where σ_i are singular values.

        θ = 0° means perfectly aligned along that component.
        θ = 90° means orthogonal.
        """
        # W_text and W_image are already QR-orthonormalized from PLS extraction
        S = np.linalg.svd(W_text.T @ W_image, compute_uv=False)  # (k,)
        S = np.clip(S, 0.0, 1.0)  # numerical safety

        angles_rad = np.arccos(S)
        angles_deg = np.degrees(angles_rad)

        grassmann = float(np.sqrt(np.sum(angles_rad ** 2)))
        chordal = float(np.sqrt(np.sum(np.sin(angles_rad) ** 2)))

        return PrincipalAngleResult(
            layer_idx=layer_idx,
            angles_deg=angles_deg,
            cosines=S,
            grassmann_dist=grassmann,
            chordal_dist=chordal,
        )

    # =================================================================== #
    # Method 4: Cross-projection test
    # =================================================================== #

    def _cross_projection(
        self,
        layer_idx,
        X_text, text_labels,
        X_image, image_labels,
        pls_text_dir, pls_img_dir,
        dim_text_dir, dim_img_dir,
    ):
        sep_text_text = self._separation(X_text @ pls_text_dir, text_labels)
        sep_img_text_pls = self._separation(X_image @ pls_text_dir, image_labels)
        sep_img_text_dim = self._separation(X_image @ dim_text_dir, image_labels)
        sep_img_img_pls = self._separation(X_image @ pls_img_dir, image_labels)
        sep_img_img_dim = self._separation(X_image @ dim_img_dir, image_labels)

        transfer_pls = sep_img_text_pls / (sep_img_img_pls + 1e-12)
        transfer_dim = sep_img_text_dim / (sep_img_img_dim + 1e-12)

        return CrossProjectionResult(
            layer_idx=layer_idx,
            sep_text_on_text=sep_text_text,
            sep_img_on_text_pls=sep_img_text_pls,
            sep_img_on_text_dim=sep_img_text_dim,
            sep_img_on_img_pls=sep_img_img_pls,
            sep_img_on_img_dim=sep_img_img_dim,
            transfer_ratio_pls=transfer_pls,
            transfer_ratio_dim=transfer_dim,
        )

    # =================================================================== #
    # Helpers
    # =================================================================== #

    def _extract_pls_basis(self, X, labels):
        Y = labels.reshape(-1, 1).astype(float)
        pls = PLSRegression(n_components=self.n_pls)
        pls.fit(X, Y)
        W = pls.x_weights_[:, :self.k]
        for i in range(self.k):
            scores = X @ W[:, i]
            if np.mean(scores[labels == 1]) < np.mean(scores[labels == 0]):
                W[:, i] = -W[:, i]
        W, _ = np.linalg.qr(W)
        return W

    @staticmethod
    def _cosine(a, b):
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
        return float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))

    @staticmethod
    def _separation(scores, labels):
        safe = scores[labels == 1]
        harm = scores[labels == 0]
        if len(safe) == 0 or len(harm) == 0:
            return 0.0
        num = abs(safe.mean() - harm.mean())
        denom = (safe.std() + harm.std()) / 2 + 1e-12
        return float(num / denom)

    # =================================================================== #
    # Report
    # =================================================================== #

    def print_report(self):
        layers = sorted(self.dim_results_.keys())
        if not layers:
            print("No results.")
            return

        # ---- Section 1: Difference-in-Means ----
        print(f"\n{'='*75}")
        print("1. DIFFERENCE-IN-MEANS (DiM) — 最简单的安全方向对比")
        print(f"{'='*75}")
        print(f"  d = mean(safe) - mean(harmful), normalized")
        print(f"  cos_sim = cosine(d_text, d_image)\n")
        print(f"{'Layer':>6}  {'cos_sim':>8}  {'|cos|':>6}  "
              f"{'d_text':>7}  {'d_img→text':>11}  {'d_img→img':>10}")
        print("-" * 75)
        for l in layers:
            r = self.dim_results_[l]
            print(f"{l:>6}  {r.cos_sim:>+8.4f}  {r.abs_cos:>6.4f}  "
                  f"{r.sep_text:>7.3f}  {r.sep_img_on_text:>11.3f}  {r.sep_img_on_img:>10.3f}")
        print("-" * 75)
        mean_cos = np.mean([r.cos_sim for r in self.dim_results_.values()])
        mean_abs = np.mean([r.abs_cos for r in self.dim_results_.values()])
        print(f"  Mean cos_sim: {mean_cos:+.4f}   Mean |cos|: {mean_abs:.4f}")

        if mean_abs < 0.2:
            print("  >> DiM方向几乎正交 — 文本和图像的safe/harm分界面在不同方向")
        elif mean_abs > 0.5:
            print("  >> DiM方向有显著对齐 — 两个模态的安全方向有重合")
        else:
            print("  >> DiM方向部分对齐 — 有一些共享结构但不一致")

        # ---- Section 2: Non-orthogonal Procrustes ----
        print(f"\n{'='*75}")
        print("2. NON-ORTHOGONAL PROCRUSTES — 放松 R^T R = I 约束")
        print(f"{'='*75}")
        print(f"  如果非正交显著好于正交，说明子空间需要缩放+旋转\n")
        print(f"{'Layer':>6}  {'L_ortho':>8}  {'L_affine':>9}  "
              f"{'↓%':>6}  {'cond(A)':>8}  {'cos_after':>10}")
        print("-" * 75)
        for l in layers:
            r = self.nonortho_results_[l]
            print(f"{l:>6}  {r.loss_ortho:>8.4f}  {r.loss_affine:>9.4f}  "
                  f"{r.loss_reduction*100:>5.1f}%  {r.condition_number:>8.1f}  "
                  f"{r.cos_after:>+10.4f}")
        print("-" * 75)
        mean_red = np.mean([r.loss_reduction for r in self.nonortho_results_.values()])
        mean_cond = np.mean([r.condition_number for r in self.nonortho_results_.values()])
        print(f"  Mean affine improvement over orthogonal: {mean_red*100:.1f}%")
        print(f"  Mean cond(A): {mean_cond:.1f}")
        if mean_red < 0.05:
            print("  >> 放松约束也没用 — 问题不在正交性，而是子空间本身不同")
        elif mean_cond > 100:
            print("  >> 需要极端缩放才能对齐 — 子空间结构差异大")
        else:
            print("  >> 适度的非正交变换有帮助")

        # ---- Section 3: Principal Angles ----
        print(f"\n{'='*75}")
        print("3. PRINCIPAL ANGLES — 子空间之间的精确几何关系")
        print(f"{'='*75}")
        print(f"  θ=0° 完全对齐, θ=90° 完全正交\n")
        for l in layers:
            r = self.angle_results_[l]
            angles_str = ", ".join([f"{a:.1f}°" for a in r.angles_deg])
            print(f"  Layer {l:3d} | angles: [{angles_str}]")
            print(f"           | Grassmann dist: {r.grassmann_dist:.3f}  "
                  f"Chordal dist: {r.chordal_dist:.3f}")
        print("-" * 75)
        mean_first_angle = np.mean([r.angles_deg[0] for r in self.angle_results_.values()])
        mean_grassmann = np.mean([r.grassmann_dist for r in self.angle_results_.values()])
        print(f"  Mean 1st principal angle: {mean_first_angle:.1f}°")
        print(f"  Mean Grassmann distance: {mean_grassmann:.3f}")
        if mean_first_angle > 70:
            print("  >> 即使最对齐的维度也接近正交 — 两个子空间完全不重合")
        elif mean_first_angle < 30:
            print("  >> 最对齐的维度有显著重合 — 存在共享安全结构")
        else:
            print("  >> 部分维度有重合，但整体子空间不同")

        # ---- Section 4: Cross-projection ----
        print(f"\n{'='*75}")
        print("4. CROSS-PROJECTION — 文本安全方向能否分离图像?")
        print(f"{'='*75}")
        print(f"  transfer_ratio = sep(img→text_dir) / sep(img→img_dir)")
        print(f"  ratio=1.0 完美迁移, ratio=0 完全不迁移\n")
        print(f"{'Layer':>6}  {'sep_txt':>8}  "
              f"{'img→txt(P)':>11}  {'img→img(P)':>11}  {'ratio(P)':>9}  "
              f"{'img→txt(D)':>11}  {'img→img(D)':>11}  {'ratio(D)':>9}")
        print("-" * 75)
        for l in layers:
            r = self.crossproj_results_[l]
            print(f"{l:>6}  {r.sep_text_on_text:>8.3f}  "
                  f"{r.sep_img_on_text_pls:>11.3f}  {r.sep_img_on_img_pls:>11.3f}  "
                  f"{r.transfer_ratio_pls:>9.3f}  "
                  f"{r.sep_img_on_text_dim:>11.3f}  {r.sep_img_on_img_dim:>11.3f}  "
                  f"{r.transfer_ratio_dim:>9.3f}")
        print("-" * 75)
        mean_tr_pls = np.mean([r.transfer_ratio_pls for r in self.crossproj_results_.values()])
        mean_tr_dim = np.mean([r.transfer_ratio_dim for r in self.crossproj_results_.values()])
        print(f"  Mean transfer ratio (PLS): {mean_tr_pls:.3f}")
        print(f"  Mean transfer ratio (DiM): {mean_tr_dim:.3f}")

        # ---- Overall verdict ----
        print(f"\n{'='*75}")
        print("OVERALL VERDICT")
        print(f"{'='*75}")
        print(f"  DiM cosine:            {mean_abs:.4f}  ({'正交' if mean_abs < 0.2 else '部分对齐' if mean_abs < 0.5 else '对齐'})")
        print(f"  1st principal angle:   {mean_first_angle:.1f}°  ({'正交' if mean_first_angle > 70 else '部分' if mean_first_angle > 30 else '对齐'})")
        print(f"  Affine improvement:    {mean_red*100:.1f}%  ({'无效' if mean_red < 0.05 else '有效'})")
        print(f"  Cross-modal transfer:  {mean_tr_pls:.3f}  ({'无迁移' if mean_tr_pls < 0.3 else '部分迁移' if mean_tr_pls < 0.7 else '强迁移'})")
        print()

        if mean_abs < 0.2 and mean_first_angle > 70:
            print("  结论: 文本和图像的安全子空间在LLaVA中几乎完全正交。")
            print("        这意味着文本safety alignment无法直接保护图像输入。")
            print("        对齐这两个子空间需要的不是简单旋转，而是更根本的干预。")
            print()
            print("  可能的干预方案:")
            print("    (a) 在训练时加入跨模态安全对齐损失")
            print("    (b) 用adapter/LoRA在safety-critical层注入共享安全方向")
            print("    (c) 推理时在隐层空间做投影: h_img → h_img + α·(d_text - d_img)")
        elif mean_tr_pls > 0.5:
            print("  结论: 虽然子空间不完全对齐，但文本安全方向可以部分分离图像。")
            print("        这说明存在一些共享的安全结构，只是不够强。")
        else:
            print("  结论: 子空间部分正交。需要非线性方法（如MLP adapter）来桥接差距。")


# =========================================================================== #
# CLI
# =========================================================================== #

def _load_cache(prefix, cache_dir):
    labels = np.load(cache_dir / f"{prefix}_labels.npy")
    states = {}
    for f in sorted(cache_dir.glob(f"{prefix}_layer*.npy")):
        layer_idx = int(f.stem.replace(f"{prefix}_layer", ""))
        states[layer_idx] = np.load(f)
    print(f"[Cache] Loaded {prefix}: {len(states)} layers, {labels.shape[0]} samples")
    return states, labels


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive cross-modal safety subspace alignment analysis."
    )
    parser.add_argument("--text-cache", default="outputs/cache")
    parser.add_argument("--image-cache", default="outputs/cache")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--layers", nargs="+", type=int, default=None)
    args = parser.parse_args()

    text_states, text_labels = _load_cache("text", Path(args.text_cache))
    image_states, image_labels = _load_cache("image", Path(args.image_cache))

    analyzer = CrossModalAnalyzer(k=args.k)
    analyzer.fit(text_states, text_labels, image_states, image_labels, layers=args.layers)
    analyzer.print_report()


if __name__ == "__main__":
    main()
