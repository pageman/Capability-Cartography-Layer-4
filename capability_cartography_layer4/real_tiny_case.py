import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .case_study import classify_metric_curve, classify_rmse_curve
from .circuit_discovery import CircuitDiscovery
from .mechanism_feedback import feedback_adjust_forecast
from .orchestration import CCL4Pipeline
from .parameter_extractor import extract_case_parameters, flatten_parameters


class TinyPeriodicNet:
    def __init__(self) -> None:
        self.weights = np.zeros(4, dtype=float)

    def predict_matrix(self, features: np.ndarray) -> np.ndarray:
        return features @ self.weights

    def train_step(self, features: np.ndarray, scores: np.ndarray, learning_rate: float) -> None:
        predictions = self.predict_matrix(features)
        error = predictions - scores
        gradient = (2.0 / max(1, len(features))) * (features.T @ error)
        self.weights -= learning_rate * gradient

    def snapshot(self) -> np.ndarray:
        return self.weights.copy()

    def restore(self, state: np.ndarray) -> None:
        self.weights = state.copy()


class TinySparseRelationalNet:
    def __init__(self) -> None:
        self.hidden_weights = np.zeros((3, 4), dtype=float)
        self.hidden_bias = np.zeros(4, dtype=float)
        self.output_weights = np.zeros(4, dtype=float)
        self.output_bias = 0.0

    def _forward_components(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        hidden_linear = features @ self.hidden_weights + self.hidden_bias
        hidden = np.maximum(hidden_linear, 0.0)
        predictions = hidden @ self.output_weights + self.output_bias
        return hidden_linear, hidden, predictions

    def predict_matrix(self, features: np.ndarray) -> np.ndarray:
        return self._forward_components(features)[2]

    def hidden_features(self, features: np.ndarray) -> np.ndarray:
        return self._forward_components(features)[1]

    def train_step(self, features: np.ndarray, scores: np.ndarray, learning_rate: float) -> None:
        hidden_linear, hidden, predictions = self._forward_components(features)
        grad_out = (2.0 / max(1, len(features))) * (predictions - scores)
        grad_output_weights = hidden.T @ grad_out
        grad_output_bias = float(np.sum(grad_out))
        grad_hidden = grad_out[:, None] * self.output_weights[None, :]
        grad_hidden[hidden_linear <= 0.0] = 0.0
        grad_hidden_weights = features.T @ grad_hidden
        grad_hidden_bias = np.sum(grad_hidden, axis=0)

        self.output_weights -= learning_rate * grad_output_weights
        self.output_bias -= learning_rate * grad_output_bias
        self.hidden_weights -= learning_rate * grad_hidden_weights
        self.hidden_bias -= learning_rate * grad_hidden_bias

    def snapshot(self) -> Dict[str, np.ndarray]:
        return {
            "hidden_weights": self.hidden_weights.copy(),
            "hidden_bias": self.hidden_bias.copy(),
            "output_weights": self.output_weights.copy(),
            "output_bias": np.array([self.output_bias], dtype=float),
        }

    def restore(self, state: Dict[str, np.ndarray]) -> None:
        self.hidden_weights = state["hidden_weights"].copy()
        self.hidden_bias = state["hidden_bias"].copy()
        self.output_weights = state["output_weights"].copy()
        self.output_bias = float(state["output_bias"][0])


def _periodic_rows() -> List[Tuple[int, List[float], float, int]]:
    rows = []
    modulus = 17
    for x in range(modulus):
        angle = 2.0 * math.pi * x / modulus
        features = [
            math.sin(angle),
            math.cos(angle),
            math.sin(2.0 * angle),
            math.cos(2.0 * angle),
        ]
        score = 1.15 * features[0] + 0.9 * features[3] - 0.2
        label = 1 if score >= 0.0 else 0
        rows.append((x, features, score, label))
    return rows


def _non_periodic_rows() -> List[Tuple[int, List[float], float, int]]:
    rows = []
    for x in range(17):
        normalized = x / 16.0
        centered = normalized - 0.5
        features = [normalized, normalized ** 2, centered, 1.0]
        score = 1.1 * normalized + 0.2 * (normalized ** 2) - 0.55
        label = 1 if score >= 0.0 else 0
        rows.append((x, features, score, label))
    return rows


def _sparse_relational_rows() -> List[Tuple[int, List[float], float, int]]:
    rows = []
    for x in range(24):
        a = float(x % 2)
        b = float((x // 2) % 2)
        c = float((x // 4) % 2)
        label = 1 if (a + b + c) >= 2 else 0
        score = 0.8 * a + 0.8 * b + 0.8 * c - 1.1
        rows.append((x, [a, b, c], score, label))
    return rows


def _split_rows(rows: Sequence[Tuple[int, List[float], float, int]]) -> Tuple[List, List]:
    train = [row for row in rows if row[0] % 3 != 0]
    heldout = [row for row in rows if row[0] % 3 == 0]
    return train, heldout


def _split_non_periodic_rows(rows: Sequence[Tuple[int, List[float], float, int]]) -> Tuple[List, List]:
    heldout_ids = {0, 4, 8, 12, 16}
    train = [row for row in rows if row[0] not in heldout_ids]
    heldout = [row for row in rows if row[0] in heldout_ids]
    return train, heldout


def _arrayize(rows: Sequence[Tuple[int, List[float], float, int]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.asarray([row[1] for row in rows], dtype=float)
    scores = np.asarray([row[2] for row in rows], dtype=float)
    labels = np.asarray([row[3] for row in rows], dtype=float)
    return features, scores, labels


def _classification_accuracy(model, features: np.ndarray, labels: np.ndarray) -> float:
    predictions = (model.predict_matrix(features) >= 0.0).astype(float)
    return float(np.mean(predictions == labels))


def _thresholded_pass_rate(model, features: np.ndarray, scores: np.ndarray, labels: np.ndarray, tolerance: float = 0.28) -> float:
    prediction_scores = model.predict_matrix(features)
    prediction_labels = (prediction_scores >= 0.0).astype(float)
    passes = (prediction_labels == labels) & (np.abs(prediction_scores - scores) <= tolerance)
    return float(np.mean(passes))


def _rmse(model, features: np.ndarray, scores: np.ndarray) -> float:
    return float(np.sqrt(np.mean((model.predict_matrix(features) - scores) ** 2)))


def _score_correlation(model, features: np.ndarray, scores: np.ndarray) -> float:
    predictions = np.asarray(model.predict_matrix(features), dtype=float)
    if len(predictions) < 2 or np.std(predictions) <= 1e-9 or np.std(scores) <= 1e-9:
        return 0.0
    return float(np.corrcoef(predictions, scores)[0, 1])


def _monotonic_agreement(values: np.ndarray) -> float:
    diffs = np.diff(np.asarray(values, dtype=float))
    non_zero = diffs[np.abs(diffs) > 1e-9]
    if non_zero.size == 0:
        return 0.0
    positive = float(np.mean(non_zero > 0.0))
    negative = float(np.mean(non_zero < 0.0))
    return max(positive, negative)


def _score_monotonic_agreement(model, features: np.ndarray, scores: np.ndarray) -> float:
    predictions = np.asarray(model.predict_matrix(features), dtype=float)
    order = np.argsort(scores)
    return _monotonic_agreement(predictions[order])


def _periodic_signal(model) -> float:
    weights = model.weights
    numerator = float(np.linalg.norm(weights[2:]))
    denominator = float(np.linalg.norm(weights))
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def _sparse_completeness(model, periodic: bool) -> float:
    if periodic:
        target = np.asarray([1.15, 0.0, 0.0, 0.9], dtype=float)
        distance = float(np.abs(model.weights - target).sum())
        max_distance = float(np.abs(target).sum()) + 1e-9
        return max(0.0, 1.0 - (distance / max_distance))
    if hasattr(model, "weights"):
        return min(1.0, float(np.mean(np.abs(model.weights))))
    return min(1.0, float(np.mean(np.abs(model.hidden_weights))))


def _train_model(model, train_rows, heldout_rows, steps: int, learning_rate: float, periodic: bool):
    train_features, train_scores, train_labels = _arrayize(train_rows)
    heldout_features, heldout_scores, heldout_labels = _arrayize(heldout_rows)
    checkpoints: List[Dict[str, float]] = []

    def record(step: int) -> None:
        checkpoints.append(
            {
                "step": step,
                "thresholded_pass_rate": round(_thresholded_pass_rate(model, train_features, train_scores, train_labels), 4),
                "classification_accuracy": round(_classification_accuracy(model, train_features, train_labels), 4),
                "score_rmse": round(_rmse(model, train_features, train_scores), 4),
                "heldout_thresholded_pass_rate": round(
                    _thresholded_pass_rate(model, heldout_features, heldout_scores, heldout_labels), 4
                ),
                "heldout_classification_accuracy": round(
                    _classification_accuracy(model, heldout_features, heldout_labels), 4
                ),
                "heldout_score_rmse": round(_rmse(model, heldout_features, heldout_scores), 4),
                "score_correlation": round(_score_correlation(model, train_features, train_scores), 4),
                "heldout_score_correlation": round(_score_correlation(model, heldout_features, heldout_scores), 4),
                "score_monotonic_agreement": round(_score_monotonic_agreement(model, train_features, train_scores), 4),
                "heldout_score_monotonic_agreement": round(
                    _score_monotonic_agreement(model, heldout_features, heldout_scores), 4
                ),
                "fourier_signal": round(_periodic_signal(model), 4) if periodic else 0.0,
                "circuit_completeness": round(_sparse_completeness(model, periodic), 4),
            }
        )

    record(0)
    for step in range(1, steps + 1):
        model.train_step(train_features, train_scores, learning_rate)
        if step % 6 == 0:
            record(step)

    return checkpoints, train_features, train_scores, train_labels, heldout_features, heldout_scores, heldout_labels


def _causal_validation(model, heldout_features: np.ndarray, heldout_scores: np.ndarray, heldout_labels: np.ndarray, periodic: bool) -> Dict[str, float]:
    baseline = _classification_accuracy(model, heldout_features, heldout_labels)
    saved_state = model.snapshot()

    if periodic:
        model.weights[2:] = 0.0
    elif hasattr(model, "weights"):
        model.weights[:] = 0.0
    else:
        model.hidden_weights[:, :] = 0.0

    ablated = _classification_accuracy(model, heldout_features, heldout_labels)
    model.restore(saved_state)
    restored = _classification_accuracy(model, heldout_features, heldout_labels)

    return {
        "baseline_accuracy": round(baseline, 4),
        "ablated_accuracy": round(ablated, 4),
        "restored_accuracy": round(restored, 4),
        "accuracy_drop_from_ablation": round(baseline - ablated, 4),
        "accuracy_recovery_after_restoration": round(restored - ablated, 4),
    }


def _forecast_for_case(case_id: str, kind: str) -> Dict[str, object]:
    pipeline = CCL4Pipeline()
    case_spec = {"case_id": case_id, "kind": kind}
    extracted = extract_case_parameters(case_spec)
    regime_params = flatten_parameters(extracted)
    if kind == "periodic":
        record = pipeline.run_record(
            paper_id=case_id,
            capability_id=case_id,
            regime_params=regime_params,
            task_metadata={
                "domain": "mathematics",
                "relational_depth": 6,
                "compressibility_gap": 24.0,
                "intrinsic_dimension": 18.0,
                "param_count": 100.0,
            },
        )
    elif kind == "non_periodic":
        record = pipeline.run_record(
            paper_id=case_id,
            capability_id=case_id,
            regime_params=regime_params,
            task_metadata={
                "domain": "regression",
                "relational_depth": 1,
                "compressibility_gap": 3.0,
                "intrinsic_dimension": 4.0,
                "param_count": 100.0,
            },
        )
    else:
        record = pipeline.run_record(
            paper_id=case_id,
            capability_id=case_id,
            regime_params=regime_params,
            task_metadata={
                "domain": "reasoning",
                "relational_depth": 5,
                "novel_architecture": True,
                "compressibility_gap": 10.0,
                "intrinsic_dimension": 14.0,
                "param_count": 80.0,
            },
        )
    return {
        "forecast_type": record.forecast.type.value,
        "confidence": round(record.forecast.confidence, 4),
        "regime_label": record.regime.regime_label,
        "parameter_extraction": extracted,
    }


def _structured_bundle(features: np.ndarray, scores: np.ndarray, labels: np.ndarray, component_names: List[str]) -> Dict[str, object]:
    return {
        "features": features.tolist(),
        "scores": scores.tolist(),
        "labels": labels.tolist(),
        "component_names": component_names,
    }


def _circuit_report(circuit) -> Dict[str, object]:
    return {
        "circuit_type": circuit.type.value,
        "components": list(circuit.components),
        "fourier_signature": circuit.fourier_signature,
        "mechanism_description": circuit.mechanism_description,
        "component_scores": circuit.analysis_metadata.get("component_scores", {}),
        "targeted_drop": circuit.analysis_metadata.get("targeted_drop"),
        "random_drop": circuit.analysis_metadata.get("random_drop"),
        "monotonic_signal": circuit.analysis_metadata.get("monotonic_signal"),
    }


def _case_evidence_rubric(
    case_id: str,
    forecast: Dict[str, object],
    observed_curve_classes: Dict[str, str],
    checkpoints: List[Dict[str, float]],
    causal_validation: Dict[str, float],
    targeted_vs_random: Optional[Dict[str, float]] = None,
    control_case: bool = False,
) -> Dict[str, object]:
    early = checkpoints[min(2, len(checkpoints) - 1)]
    first = checkpoints[0]
    late = checkpoints[-1]
    rmse_drop = round(first["score_rmse"] - early["score_rmse"], 4)
    fourier_gain = round(early.get("fourier_signal", 0.0) - first.get("fourier_signal", 0.0), 4)
    completeness_gain = round(early.get("circuit_completeness", 0.0) - first.get("circuit_completeness", 0.0), 4)
    late_fourier_strength = round(late.get("fourier_signal", 0.0), 4)
    heldout_score_correlation = round(late.get("heldout_score_correlation", 0.0), 4)
    heldout_monotonic_agreement = round(late.get("heldout_score_monotonic_agreement", 0.0), 4)

    supports_forecast = str(forecast["forecast_type"]) in {
        observed_curve_classes["thresholded_pass_rate"],
        observed_curve_classes["classification_accuracy"],
        observed_curve_classes["score_rmse"],
    }
    supports_masking = (
        observed_curve_classes["thresholded_pass_rate"] == "step-function"
        or observed_curve_classes["classification_accuracy"] == "step-function"
    ) and observed_curve_classes["score_rmse"] == "power-law" and rmse_drop > 0.15
    supports_mechanism = (
        late_fourier_strength > 0.45
        and completeness_gain > 0.2
        and targeted_vs_random is not None
        and targeted_vs_random["targeted_drop"] > targeted_vs_random["random_drop"] + 0.15
        and causal_validation["accuracy_recovery_after_restoration"] >= 0.15
    )
    if control_case:
        supports_masking = False
        supports_forecast = (
            observed_curve_classes["score_rmse"] == "power-law"
            and heldout_score_correlation >= 0.98
            and heldout_monotonic_agreement >= 0.95
            and late_fourier_strength <= 0.05
        )
        supports_mechanism = (
            abs(fourier_gain) <= 0.05
            and heldout_score_correlation >= 0.98
            and heldout_monotonic_agreement >= 0.95
            and (
                targeted_vs_random is None
                or abs(targeted_vs_random["targeted_drop"] - targeted_vs_random["random_drop"]) <= 0.05
            )
        )

    grade = "weak"
    if supports_forecast and supports_mechanism and (supports_masking or control_case):
        grade = "strong"
    elif supports_forecast and (supports_masking or supports_mechanism):
        grade = "moderate"

    return {
        "case_id": case_id,
        "supports_masking_claim": supports_masking,
        "supports_mechanism_claim": supports_mechanism,
        "supports_forecast_claim": supports_forecast,
        "overall_evidence_grade": grade,
        "control_case": control_case,
        "supports_control_claim": control_case and supports_forecast and supports_mechanism,
        "early_rmse_drop": rmse_drop,
        "early_fourier_gain": fourier_gain,
        "early_completeness_gain": completeness_gain,
        "late_fourier_strength": late_fourier_strength,
        "heldout_score_correlation": heldout_score_correlation,
        "heldout_score_monotonic_agreement": heldout_monotonic_agreement,
    }


def run_real_tiny_suite(output_dir: Optional[Path] = None) -> Dict[str, object]:
    output_root = output_dir or Path(__file__).resolve().parents[1] / "artifacts" / "minimal_suite"
    output_root.mkdir(parents=True, exist_ok=True)
    np.random.seed(0)
    cases: Dict[str, object] = {}

    periodic_rows = _periodic_rows()
    periodic_train, periodic_heldout = _split_rows(periodic_rows)
    periodic_model = TinyPeriodicNet()
    periodic_discovery = CircuitDiscovery(periodic_model)
    periodic_checkpoints, _, _, _, periodic_hf, periodic_hs, periodic_hl = _train_model(
        periodic_model, periodic_train, periodic_heldout, steps=72, learning_rate=0.08, periodic=True
    )
    periodic_bundle = _structured_bundle(
        periodic_hf,
        periodic_hs,
        periodic_hl,
        ["sin", "cos", "sin2", "cos2"],
    )
    periodic_circuit = periodic_discovery.identify_circuit("tiny_periodic_modular", periodic_bundle)
    periodic_payload = {
        "forecast": _forecast_for_case("tiny_periodic_modular", "periodic"),
        "observed_curve_classes": {
            "thresholded_pass_rate": classify_metric_curve([checkpoint["thresholded_pass_rate"] for checkpoint in periodic_checkpoints]),
            "classification_accuracy": classify_metric_curve([checkpoint["classification_accuracy"] for checkpoint in periodic_checkpoints]),
            "score_rmse": classify_rmse_curve([checkpoint["score_rmse"] for checkpoint in periodic_checkpoints]),
        },
        "checkpoints": periodic_checkpoints,
        "causal_validation": _causal_validation(periodic_model, periodic_hf, periodic_hs, periodic_hl, periodic=True),
        "mechanism_discovery": _circuit_report(periodic_circuit),
        "targeted_vs_random_ablation": {
            "baseline_accuracy": round(_classification_accuracy(periodic_model, periodic_hf, periodic_hl), 4),
            "targeted_drop": periodic_circuit.analysis_metadata.get("targeted_drop", 0.0) or 0.0,
            "random_drop": periodic_circuit.analysis_metadata.get("random_drop", 0.0) or 0.0,
        },
    }
    periodic_payload["feedback_adjusted_forecast"] = feedback_adjust_forecast(periodic_payload["forecast"], periodic_checkpoints)
    periodic_payload["evidence_rubric"] = _case_evidence_rubric(
        case_id="tiny_periodic_modular",
        forecast=periodic_payload["forecast"],
        observed_curve_classes=periodic_payload["observed_curve_classes"],
        checkpoints=periodic_checkpoints,
        causal_validation=periodic_payload["causal_validation"],
        targeted_vs_random=periodic_payload["targeted_vs_random_ablation"],
    )
    cases["tiny_periodic_modular"] = periodic_payload

    non_periodic_rows = _non_periodic_rows()
    non_periodic_train, non_periodic_heldout = _split_non_periodic_rows(non_periodic_rows)
    non_periodic_model = TinyPeriodicNet()
    non_periodic_discovery = CircuitDiscovery(non_periodic_model)
    non_periodic_checkpoints, _, _, _, non_hf, non_hs, non_hl = _train_model(
        non_periodic_model, non_periodic_train, non_periodic_heldout, steps=72, learning_rate=0.09, periodic=False
    )
    non_periodic_bundle = _structured_bundle(
        non_hf,
        non_hs,
        non_hl,
        ["normalized_input", "quadratic_trend", "centered_input", "bias_like"],
    )
    non_periodic_circuit = non_periodic_discovery.identify_circuit("tiny_nonperiodic_linear", non_periodic_bundle)
    non_periodic_payload = {
        "forecast": _forecast_for_case("tiny_nonperiodic_linear", "non_periodic"),
        "observed_curve_classes": {
            "thresholded_pass_rate": classify_metric_curve([checkpoint["thresholded_pass_rate"] for checkpoint in non_periodic_checkpoints]),
            "classification_accuracy": classify_metric_curve([checkpoint["classification_accuracy"] for checkpoint in non_periodic_checkpoints]),
            "score_rmse": classify_rmse_curve([checkpoint["score_rmse"] for checkpoint in non_periodic_checkpoints]),
        },
        "checkpoints": non_periodic_checkpoints,
        "causal_validation": _causal_validation(non_periodic_model, non_hf, non_hs, non_hl, periodic=False),
        "mechanism_discovery": _circuit_report(non_periodic_circuit),
    }
    non_periodic_payload["feedback_adjusted_forecast"] = feedback_adjust_forecast(non_periodic_payload["forecast"], non_periodic_checkpoints)
    non_periodic_payload["targeted_vs_random_ablation"] = {
        "baseline_accuracy": round(_classification_accuracy(non_periodic_model, non_hf, non_hl), 4),
        "targeted_drop": non_periodic_circuit.analysis_metadata.get("targeted_drop", 0.0) or 0.0,
        "random_drop": non_periodic_circuit.analysis_metadata.get("random_drop", 0.0) or 0.0,
    }
    non_periodic_payload["evidence_rubric"] = _case_evidence_rubric(
        case_id="tiny_nonperiodic_linear",
        forecast=non_periodic_payload["forecast"],
        observed_curve_classes=non_periodic_payload["observed_curve_classes"],
        checkpoints=non_periodic_checkpoints,
        causal_validation=non_periodic_payload["causal_validation"],
        targeted_vs_random=non_periodic_payload["targeted_vs_random_ablation"],
        control_case=True,
    )
    cases["tiny_nonperiodic_linear"] = non_periodic_payload

    sparse_rows = _sparse_relational_rows()
    sparse_train, sparse_heldout = _split_rows(sparse_rows)
    sparse_model = TinySparseRelationalNet()
    sparse_discovery = CircuitDiscovery(sparse_model)
    sparse_checkpoints, _, _, _, sparse_hf, sparse_hs, sparse_hl = _train_model(
        sparse_model, sparse_train, sparse_heldout, steps=96, learning_rate=0.05, periodic=False
    )
    sparse_hidden = sparse_model.hidden_features(sparse_hf)
    sparse_bundle = _structured_bundle(
        sparse_hidden,
        sparse_hs,
        sparse_hl,
        [f"hidden_unit_{index}" for index in range(sparse_hidden.shape[1])],
    )
    sparse_circuit = sparse_discovery.identify_circuit("tiny_sparse_relational", sparse_bundle)
    sparse_payload = {
        "forecast": _forecast_for_case("tiny_sparse_relational", "sparse_relational"),
        "observed_curve_classes": {
            "thresholded_pass_rate": classify_metric_curve([checkpoint["thresholded_pass_rate"] for checkpoint in sparse_checkpoints]),
            "classification_accuracy": classify_metric_curve([checkpoint["classification_accuracy"] for checkpoint in sparse_checkpoints]),
            "score_rmse": classify_rmse_curve([checkpoint["score_rmse"] for checkpoint in sparse_checkpoints]),
        },
        "checkpoints": sparse_checkpoints,
        "causal_validation": _causal_validation(sparse_model, sparse_hf, sparse_hs, sparse_hl, periodic=False),
        "mechanism_discovery": _circuit_report(sparse_circuit),
        "targeted_vs_random_ablation": {
            "baseline_accuracy": round(_classification_accuracy(sparse_model, sparse_hf, sparse_hl), 4),
            "targeted_drop": sparse_circuit.analysis_metadata.get("targeted_drop", 0.0) or 0.0,
            "random_drop": sparse_circuit.analysis_metadata.get("random_drop", 0.0) or 0.0,
        },
    }
    sparse_payload["feedback_adjusted_forecast"] = feedback_adjust_forecast(sparse_payload["forecast"], sparse_checkpoints)
    sparse_payload["evidence_rubric"] = _case_evidence_rubric(
        case_id="tiny_sparse_relational",
        forecast=sparse_payload["forecast"],
        observed_curve_classes=sparse_payload["observed_curve_classes"],
        checkpoints=sparse_checkpoints,
        causal_validation=sparse_payload["causal_validation"],
        targeted_vs_random=sparse_payload["targeted_vs_random_ablation"],
        control_case=False,
    )
    cases["tiny_sparse_relational"] = sparse_payload

    summary = {
        "suite_id": "ccl4-minimum-viable-pass-v1",
        "cases": cases,
    }
    (output_root / "real_tiny_results.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run_real_tiny_suite(), indent=2))
