from typing import Any, Iterable, List

import numpy as np


class LinearFeatureAdapter:
    """Minimal adapter for linear NumPy-style models with a `weights` vector."""

    def __init__(self, model: Any) -> None:
        self.model = model

    @classmethod
    def supports(cls, model: Any) -> bool:
        return hasattr(model, "weights")

    def snapshot(self) -> np.ndarray:
        return np.asarray(self.model.weights, dtype=float).copy()

    def restore(self, state: np.ndarray) -> None:
        self.model.weights = np.asarray(state, dtype=float).copy()

    def predict_scores(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(features, dtype=float) @ np.asarray(self.model.weights, dtype=float)

    def ablate_components(self, component_indices: Iterable[int]) -> None:
        weights = np.asarray(self.model.weights, dtype=float).copy()
        for index in component_indices:
            if 0 <= index < len(weights):
                weights[index] = 0.0
        self.model.weights = weights

    def random_control_indices(self, component_indices: List[int]) -> List[int]:
        all_indices = list(range(len(np.asarray(self.model.weights, dtype=float))))
        excluded = set(component_indices)
        controls = [index for index in all_indices if index not in excluded]
        return controls[: max(1, len(component_indices))]


class VectorReadoutAdapter:
    """Adapter for models with a vector readout over supplied feature bundles."""

    def __init__(self, model: Any) -> None:
        self.model = model
        if hasattr(model, "readout"):
            self.weight_attr = "readout"
            self.bias_attr = "bias"
        else:
            self.weight_attr = "output_weights"
            self.bias_attr = "output_bias"

    @classmethod
    def supports(cls, model: Any) -> bool:
        return hasattr(model, "readout") or hasattr(model, "output_weights")

    def snapshot(self) -> dict:
        return {
            "weights": np.asarray(getattr(self.model, self.weight_attr), dtype=float).copy(),
            "bias": float(getattr(self.model, self.bias_attr, 0.0)),
        }

    def restore(self, state: dict) -> None:
        setattr(self.model, self.weight_attr, np.asarray(state["weights"], dtype=float).copy())
        setattr(self.model, self.bias_attr, float(state["bias"]))

    def predict_scores(self, features: np.ndarray) -> np.ndarray:
        weights = np.asarray(getattr(self.model, self.weight_attr), dtype=float)
        bias = float(getattr(self.model, self.bias_attr, 0.0))
        return np.asarray(features, dtype=float) @ weights + bias

    def ablate_components(self, component_indices: Iterable[int]) -> None:
        weights = np.asarray(getattr(self.model, self.weight_attr), dtype=float).copy()
        for index in component_indices:
            if 0 <= index < len(weights):
                weights[index] = 0.0
        setattr(self.model, self.weight_attr, weights)

    def random_control_indices(self, component_indices: List[int]) -> List[int]:
        all_indices = list(range(len(np.asarray(getattr(self.model, self.weight_attr), dtype=float))))
        excluded = set(component_indices)
        controls = [index for index in all_indices if index not in excluded]
        return controls[: max(1, len(component_indices))]
