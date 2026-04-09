"""
Direction computation utilities.

Key functions:
  mean_diff_direction(harmful_embs, safe_embs)
      → unit vector: mean(harmful) - mean(safe)

  cosine_similarity_matrix(directions)
      → (K, K) symmetric matrix

  principal_angles(A, B, k)
      → principal angles (radians) between two subspaces spanned by A, B

  explained_variance_knee(singular_values)
      → auto-select subspace dimension k via knee of explained variance curve
"""

from __future__ import annotations
import numpy as np
from scipy.linalg import svd


# ---------------------------------------------------------------------------
# Single direction
# ---------------------------------------------------------------------------

def mean_diff_direction(harmful: np.ndarray, safe: np.ndarray) -> np.ndarray:
    """
    Compute L2-normalised mean-difference direction.
    harmful, safe: (N, D) L2-normalised embeddings
    Returns: (D,) unit vector
    """
    diff = harmful.mean(axis=0) - safe.mean(axis=0)
    norm = np.linalg.norm(diff)
    if norm < 1e-8:
        # harmful and safe means are identical; return zeros to signal no direction
        return np.zeros_like(diff)
    return diff / norm


def cosine_similarity_matrix(directions: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    """
    directions: {label: (D,) unit vector}
    Returns:
        cos_sim: (K, K) float matrix
        labels : list of K label strings (same order)
    """
    labels = list(directions.keys())
    vecs   = np.stack([directions[l] for l in labels], axis=0)   # (K, D)
    cos_sim = vecs @ vecs.T                                        # (K, K)
    return cos_sim, labels


# ---------------------------------------------------------------------------
# Safety subspace (Idea 1)
# ---------------------------------------------------------------------------

def safety_subspace(harmful: np.ndarray, safe: np.ndarray, k: int | None = None
                    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the k-dimensional safety subspace via SVD of the difference matrix.
    harmful, safe: (N, D) — need not have same N (will use min)
    Returns:
        U  : (D, k) orthonormal basis of safety subspace
        sv : (k,)   singular values
    """
    n = min(len(harmful), len(safe))
    diff_matrix = harmful[:n] - safe[:n]           # (N, D)
    diff_matrix -= diff_matrix.mean(axis=0)        # center

    U, sv, _ = svd(diff_matrix, full_matrices=False)
    # U is (N, N), sv is (min(N,D),), Vt is (min(N,D), D)
    # We want the right singular vectors (Vt rows) as the subspace basis
    _, sv, Vt = svd(diff_matrix, full_matrices=False)

    if k is None:
        k = explained_variance_knee(sv)

    return Vt[:k].T, sv   # (D, k) basis, (min(N,D),) singular values


def explained_variance_knee(singular_values: np.ndarray, threshold: float = 0.90) -> int:
    """
    Auto-select k as the smallest integer s.t.
    cumulative explained variance >= threshold.
    Falls back to knee-point detection if threshold gives k=1.
    """
    var = singular_values ** 2
    cum_var = np.cumsum(var) / var.sum()
    k = int(np.searchsorted(cum_var, threshold)) + 1
    return max(k, 2)


def principal_angles(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Compute principal angles (radians) between subspaces spanned by columns of U and V.
    U: (D, p), V: (D, q) — orthonormal bases
    Returns: min(p, q) principal angles in [0, pi/2]
    """
    # QR to ensure orthonormality
    Qu, _ = np.linalg.qr(U)
    Qv, _ = np.linalg.qr(V)
    M  = Qu.T @ Qv                          # (p, q)
    sv = np.linalg.svd(M, compute_uv=False) # singular values in [0, 1]
    sv = np.clip(sv, 0, 1)
    return np.arccos(sv)                    # (min(p,q),) in radians


def subspace_overlap(U: np.ndarray, V: np.ndarray) -> float:
    """
    Scalar overlap measure: mean cos(principal_angles).
    1.0 = identical subspaces, 0.0 = fully orthogonal.
    """
    angles = principal_angles(U, V)
    return float(np.cos(angles).mean())


# ---------------------------------------------------------------------------
# Orthogonal complement projection (Idea 3)
# ---------------------------------------------------------------------------

def project_orthogonal(x: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """
    Remove the component of x along `direction`.
    x         : (D,) or (N, D)
    direction : (D,) unit vector
    Returns   : projected vector(s), same shape as x
    """
    d = direction / (np.linalg.norm(direction) + 1e-8)
    if x.ndim == 1:
        return x - np.dot(x, d) * d
    return x - (x @ d)[:, None] * d   # (N, D)


def residual_harmfulness(
    harmful_projected: np.ndarray,
    safe_projected:    np.ndarray,
) -> float:
    """
    After projecting out the refusal direction, how much of the
    harmful signal remains?
    Returns cosine similarity of the projected mean-diff to itself
    (i.e., L2 norm ratio vs. original). If close to 1, harmful signal persists.
    """
    proj_diff  = harmful_projected.mean(0) - safe_projected.mean(0)
    return float(np.linalg.norm(proj_diff))
