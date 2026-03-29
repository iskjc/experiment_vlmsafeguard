"""
Cross-modal safety direction cosine similarity analysis.

This computes the cosine similarity between:
  - text safety direction  (PLS fitted on SaferLHF text hidden states)
  - image safety direction (PLS fitted on HoD image hidden states)

at each layer, to replicate the analysis in:
  "Cross-Modal Safety Mechanism Transfer in Large Vision-Language Models"
  which reports LOW cosine similarity (inconsistent directions).

If your results show HIGH similarity, possible reasons are documented at the
bottom of this file.

Usage (standalone, loads cached .npy files):
    python -m analysis.cosine_similarity --cache-dir outputs/cache --layers 14 15 16 17

Usage (as a module):
    from analysis.cosine_similarity import CrossModalCosineAnalyzer
    analyzer = CrossModalCosineAnalyzer()
    results = analyzer.fit_and_compare(text_states, text_labels, image_states, image_labels)
    analyzer.print_report(results)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------------------------------- #
# Data class for per-layer result
# --------------------------------------------------------------------------- #

@dataclass
class LayerCosineSimilarity:
    layer_idx: int
    text_direction: np.ndarray      # (D,) unit vector — PLS on text
    image_direction: np.ndarray     # (D,) unit vector — PLS on image
    cosine_similarity: float        # [-1, 1]; 1 = identical, 0 = orthogonal, -1 = opposite
    abs_cosine: float               # |cos|; sign-invariant measure


# --------------------------------------------------------------------------- #
# Core analyzer
# --------------------------------------------------------------------------- #

class CrossModalCosineAnalyzer:
    """
    Fits independent PLS directions on text and image hidden states,
    then computes cosine similarity between the two directions per layer.

    Key design:
      - Text direction : PLS(text_states, text_labels)  — from SaferLHF
      - Image direction: PLS(image_states, image_labels) — from HoD
      - Cosine sim     : dot(text_dir, image_dir) after both are L2-normalised

    Why independent PLS?
      The paper derives separate "safety subspaces" for text and image modalities,
      then asks: are they the same direction?  Projecting images onto a text-derived
      direction (as in pls_direction.py) answers a DIFFERENT question — it asks
      "where do images land in the text safety space", not "are the spaces aligned".
      This file answers the alignment question directly.
    """

    def __init__(self, n_components: int = 5):
        self.n_components = n_components
        self.results_: dict[int, LayerCosineSimilarity] = {}

    # ------------------------------------------------------------------ #

    def fit_and_compare(
        self,
        text_states: dict[int, np.ndarray],    # {layer: (N_text, D)}
        text_labels: np.ndarray,               # (N_text,) 1=safe 0=harmful
        image_states: dict[int, np.ndarray],   # {layer: (N_img,  D)}
        image_labels: np.ndarray,              # (N_img,)  1=safe 0=harmful
        layers: Optional[list[int]] = None,
    ) -> dict[int, LayerCosineSimilarity]:
        """
        Main entry point. Fits PLS on text and image separately per layer,
        then computes cosine similarity.

        Parameters
        ----------
        text_states  : hidden states from SaferLHF text-only inputs
        text_labels  : safety labels for text (1=safe, 0=harmful)
        image_states : hidden states from HoD image inputs
        image_labels : safety labels for images (1=safe, 0=harmful)
        layers       : if given, only compute for these layer indices

        Returns
        -------
        dict mapping layer_idx -> LayerCosineSimilarity
        """
        shared_layers = set(text_states.keys()) & set(image_states.keys())
        if layers is not None:
            shared_layers = shared_layers & set(layers)
        shared_layers = sorted(shared_layers)

        if not shared_layers:
            raise ValueError(
                "No overlapping layers found between text_states and image_states. "
                f"Text layers: {sorted(text_states.keys())}, "
                f"Image layers: {sorted(image_states.keys())}"
            )

        print(f"\n[CosineAnalyzer] Computing on {len(shared_layers)} layers: {shared_layers}")

        for layer_idx in shared_layers:
            text_dir = self._fit_pls_direction(
                text_states[layer_idx], text_labels, label_name="text"
            )
            image_dir = self._fit_pls_direction(
                image_states[layer_idx], image_labels, label_name="image"
            )

            cos_sim = float(np.dot(text_dir, image_dir))
            cos_sim = max(-1.0, min(1.0, cos_sim))   # numerical clamp

            result = LayerCosineSimilarity(
                layer_idx=layer_idx,
                text_direction=text_dir,
                image_direction=image_dir,
                cosine_similarity=cos_sim,
                abs_cosine=abs(cos_sim),
            )
            self.results_[layer_idx] = result

            print(
                f"  Layer {layer_idx:3d} | "
                f"cos_sim={cos_sim:+.4f}  |cos|={abs(cos_sim):.4f}  "
                f"→ {'ALIGNED' if abs(cos_sim) > 0.5 else 'ORTHOGONAL' if abs(cos_sim) < 0.2 else 'PARTIAL'}"
            )

        return self.results_
    

    # ------------------------------------------------------------------ #

    def _fit_pls_direction(
        self,
        states: np.ndarray,   # (N, D)
        labels: np.ndarray,   # (N,)
        label_name: str = "",
    ) -> np.ndarray:
        """
        Fit PLS on (states, labels) and return the first component as a unit vector.
        Same convention as PLSDirectionFinder in pls_direction.py:
          positive direction = safe side.
        """
        scaler = StandardScaler()
        X = scaler.fit_transform(states)
        Y = labels.reshape(-1, 1).astype(float)

        pls = PLSRegression(n_components=self.n_components)
        pls.fit(X, Y)

        direction = pls.x_weights_[:, 0]          # first component (D,)
        direction = direction / (np.linalg.norm(direction) + 1e-12)

        # Sign convention: positive projection = safe
        scores = X @ direction
        safe_mask = labels == 1
        harm_mask = labels == 0
        if safe_mask.any() and harm_mask.any():
            if np.mean(scores[safe_mask]) < np.mean(scores[harm_mask]):
                direction = -direction

        return direction

    # ------------------------------------------------------------------ #

    def print_report(self, results: Optional[dict] = None) -> None:
        """Print a human-readable summary of cosine similarities."""
        if results is None:
            results = self.results_
        if not results:
            print("[CosineAnalyzer] No results to report.")
            return

        cos_vals = [r.cosine_similarity for r in results.values()]
        abs_vals = [r.abs_cosine for r in results.values()]

        print("\n" + "=" * 60)
        print("CROSS-MODAL COSINE SIMILARITY  (SaferLHF text vs HoD image)")
        print("=" * 60)
        print(f"{'Layer':>6}  {'cos_sim':>8}  {'|cos|':>6}  {'Interpretation':}")
        print("-" * 60)
        for layer_idx in sorted(results.keys()):
            r = results[layer_idx]
            if r.abs_cosine > 0.5:
                interp = "ALIGNED — directions are similar"
            elif r.abs_cosine < 0.2:
                interp = "ORTHOGONAL — directions are unrelated"
            else:
                interp = "PARTIAL alignment"
            print(f"{layer_idx:>6}  {r.cosine_similarity:>+8.4f}  {r.abs_cosine:>6.4f}  {interp}")
        print("-" * 60)
        print(f"  Mean cos_sim : {np.mean(cos_vals):+.4f}")
        print(f"  Mean |cos|   : {np.mean(abs_vals):.4f}")
        print(f"  Std  |cos|   : {np.std(abs_vals):.4f}")

        mean_abs = np.mean(abs_vals)
        print("\n  Verdict:")
        if mean_abs > 0.5:
            print("  >> HIGH similarity — text and image safety directions are ALIGNED.")
            print("     This CONTRADICTS the paper's finding of low cosine similarity.")
            print("     See 'Why your results may differ' section below.")
        elif mean_abs < 0.2:
            print("  >> LOW similarity — directions are nearly ORTHOGONAL.")
            print("     This REPLICATES the paper's finding.")
        else:
            print("  >> MODERATE similarity — partial cross-modal transfer.")

        print("\n" + "=" * 60)
        print("WHY YOUR RESULTS MAY DIFFER FROM THE PAPER")
        print("=" * 60)
        print("""
  1. Dataset gap
       The paper uses their own curated image/text pairs.
       SaferLHF + HoD may have very different harm distributions or
       vocabulary overlap, making PLS find a shared linguistic axis
       rather than a modality-specific safety axis.

  2. Pooling strategy
       'last token' vs 'mean pooling' changes which part of the
       representation PLS extracts.  Mean pooling smooths out
       modality-specific variation, which can inflate cosine sim.

  3. Prompt template effect
       If image inputs use the same text prompt ("Is this safe?"),
       the language model sees identical text tokens for both modalities.
       PLS may latch onto the prompt embedding (shared) rather than
       the visual content embedding (modality-specific).
       → Try extracting hidden states from the IMAGE tokens only
         (before the prompt tokens) to isolate visual representations.

  4. Layer selection
       Only top safety layers are compared here.  If those layers
       happen to be dominated by text processing (common in LLaVA),
       both text and image directions reflect text-space geometry.

  5. PLS vs the paper's method
       The paper may use a different direction-finding method
       (e.g. difference-in-means, linear SVM normal vector, or
       probing-classifier weight).  PLS can produce smoother,
       more generalizable directions.
        """)


# --------------------------------------------------------------------------- #
# Standalone CLI entry point (loads cached .npy files)
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
        description="Compute cosine similarity between text and image safety directions."
    )
    parser.add_argument(
        "--cache-dir", type=str, default="outputs/cache",
        help="Directory containing cached text_layer*.npy and image_layer*.npy files."
    )
    parser.add_argument(
        "--text-cache", type=str, default=None,
        help="Optional: separate directory for text cache. Defaults to --cache-dir."
    )
    parser.add_argument(
        "--image-cache", type=str, default=None,
        help="Optional: separate directory for image cache. Defaults to --cache-dir."
    )
    parser.add_argument(
        "--layers", nargs="+", type=int, default=None,
        help="Subset of layers to compare. Defaults to all cached layers."
    )
    parser.add_argument(
        "--n-components", type=int, default=5,
        help="Number of PLS components (first one used as direction)."
    )
    args = parser.parse_args()

    text_cache_dir = Path(args.text_cache) if args.text_cache else Path(args.cache_dir)
    image_cache_dir = Path(args.image_cache) if args.image_cache else Path(args.cache_dir)

    if not text_cache_dir.exists():
        raise FileNotFoundError(f"Text cache directory not found: {text_cache_dir}")
    if not image_cache_dir.exists():
        raise FileNotFoundError(f"Image cache directory not found: {image_cache_dir}")

    print(f"[CosineAnalyzer] Loading text from {text_cache_dir}")
    text_states, text_labels = _load_cache("text", text_cache_dir)
    print(f"[CosineAnalyzer] Loading image from {image_cache_dir}")
    image_states, image_labels = _load_cache("image", image_cache_dir)

    analyzer = CrossModalCosineAnalyzer(n_components=args.n_components)
    results = analyzer.fit_and_compare(
        text_states, text_labels,
        image_states, image_labels,
        layers=args.layers,
    )
    analyzer.print_report(results)


if __name__ == "__main__":
    main()
