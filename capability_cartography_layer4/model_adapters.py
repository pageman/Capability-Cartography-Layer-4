from typing import Any, Iterable, List

import numpy as np


def _as_scalar(value: Any) -> float:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return float(array)
    return float(array.reshape(-1)[0])


class LinearFeatureAdapter:
    """Minimal adapter for linear NumPy-style models with a vector or single-output weight."""

    def __init__(self, model: Any) -> None:
        self.model = model
        if hasattr(model, "weights"):
            self.weight_attr = "weights"
        else:
            self.weight_attr = "weight"
        self.bias_attr = "bias"

    @classmethod
    def supports(cls, model: Any) -> bool:
        if hasattr(model, "weights"):
            return True
        if hasattr(model, "weight"):
            weight = np.asarray(model.weight, dtype=float)
            return weight.ndim in (1, 2)
        return False

    def _get_weights(self) -> np.ndarray:
        weights = np.asarray(getattr(self.model, self.weight_attr), dtype=float)
        if weights.ndim == 2:
            if weights.shape[0] == 1:
                return weights[0].copy()
            return np.mean(weights, axis=0)
        return weights.copy()

    def snapshot(self) -> np.ndarray:
        return self._get_weights()

    def restore(self, state: np.ndarray) -> None:
        restored = np.asarray(state, dtype=float).copy()
        current = np.asarray(getattr(self.model, self.weight_attr), dtype=float)
        if current.ndim == 2:
            tiled = np.tile(restored[None, :], (current.shape[0], 1))
            setattr(self.model, self.weight_attr, tiled)
        else:
            setattr(self.model, self.weight_attr, restored)

    def predict_scores(self, features: np.ndarray) -> np.ndarray:
        bias = _as_scalar(getattr(self.model, self.bias_attr, 0.0))
        return np.asarray(features, dtype=float) @ self._get_weights() + bias

    def ablate_components(self, component_indices: Iterable[int]) -> None:
        weights = self._get_weights()
        for index in component_indices:
            if 0 <= index < len(weights):
                weights[index] = 0.0
        self.restore(weights)

    def random_control_indices(self, component_indices: List[int]) -> List[int]:
        all_indices = list(range(len(self._get_weights())))
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
            "bias": _as_scalar(getattr(self.model, self.bias_attr, 0.0)),
        }

    def restore(self, state: dict) -> None:
        setattr(self.model, self.weight_attr, np.asarray(state["weights"], dtype=float).copy())
        setattr(self.model, self.bias_attr, _as_scalar(state["bias"]))

    def predict_scores(self, features: np.ndarray) -> np.ndarray:
        weights = np.asarray(getattr(self.model, self.weight_attr), dtype=float)
        bias = _as_scalar(getattr(self.model, self.bias_attr, 0.0))
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
