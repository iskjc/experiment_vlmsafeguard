"""
Safety layer localization via linear probing.

For each layer, fit a logistic regression (safe vs harmful) on text hidden states.
Layers with highest accuracy are the "safety-active" layers — these are where
the model geometrically separates safe from harmful content in text.

Why this matters: We only run PLS on the informative layers, not all ~32 layers,
reducing noise and computation.
"""

from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from typing import Tuple
import warnings


class LayerLocator:
    def __init__(self, n_safety_layers: int = 5, cv: int = 5, random_state: int = 42):
        self.n_safety_layers = n_safety_layers
        self.cv = cv
        self.random_state = random_state
        self.layer_accuracy_: dict[int, float] = {}
        self.top_layers_: list[int] = []

    def fit(self, states_by_layer: dict[int, np.ndarray], labels: np.ndarray) -> "LayerLocator":
        """
        Probe each layer with logistic regression.

        Parameters
        ----------
        states_by_layer : {layer_idx: (N, D)}
        labels          : (N,) int, 1=safe / 0=harmful
        """
        print("[LayerLocator] Probing layers...")
        for layer_idx, states in states_by_layer.items():
            scaler = StandardScaler()
            X = scaler.fit_transform(states)
            clf = LogisticRegression(max_iter=1000, random_state=self.random_state)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores = cross_val_score(clf, X, labels, cv=self.cv, scoring="accuracy")
            acc = scores.mean()
            self.layer_accuracy_[layer_idx] = acc
            print(f"  Layer {layer_idx:3d}: accuracy = {acc:.4f}")

        # Top-k by accuracy
        sorted_layers = sorted(self.layer_accuracy_, key=self.layer_accuracy_.get, reverse=True)
        self.top_layers_ = sorted(sorted_layers[: self.n_safety_layers])
        print(f"\n[LayerLocator] Top-{self.n_safety_layers} safety layers: {self.top_layers_}")
        return self

    def get_top_layer_states(
        self,
        states_by_layer: dict[int, np.ndarray],
    ) -> dict[int, np.ndarray]:
        """Return only the states for the top safety layers."""
        return {k: states_by_layer[k] for k in self.top_layers_ if k in states_by_layer}
