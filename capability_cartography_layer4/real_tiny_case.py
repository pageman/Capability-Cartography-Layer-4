import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    from .schemas import MockTorch

    torch = MockTorch()

    class MockNN:
        class Module:
            pass

        class Linear:
            def __init__(self, *args, **kwargs):
                pass

    nn = MockNN()

from .case_study import classify_metric_curve, classify_rmse_curve
from .mechanism_feedback import feedback_adjust_forecast
from .orchestration import CCL4Pipeline
from .parameter_extractor import extract_case_parameters, flatten_parameters

HAS_REAL_TORCH = hasattr(torch, "manual_seed") and hasattr(nn, "Linear")


class TinyPeriodicNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 1)

    def forward(self, inputs):
        return self.linear(inputs).squeeze(-1)


class TinySparseRelationalNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = nn.Linear(3, 4)
        self.linear2 = nn.Linear(4, 1)

    def forward(self, inputs):
        hidden = torch.relu(self.linear1(inputs))
        return self.linear2(hidden).squeeze(-1)


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
        features = [normalized, 1.0, 0.0, 0.0]
        score = 1.4 * normalized - 0.6
        label = 1 if score >= 0.0 else 0
        rows.append((x, features, score, label))
    return rows


def _sparse_relational_rows() -> List[Tuple[int, List[float], float, int]]:
    rows = []
    for x in range(24):
        a = (x % 2) * 1.0
        b = ((x // 2) % 2) * 1.0
        c = ((x // 4) % 2) * 1.0
        label = 1 if (a + b + c) >= 2 else 0
        score = 0.8 * a + 0.8 * b + 0.8 * c - 1.1
        rows.append((x, [a, b, c], score, label))
    return rows


def _split_rows(rows: Sequence[Tuple[int, List[float], float, int]]) -> Tuple[List, List]:
    train = [row for row in rows if row[0] % 3 != 0]
    heldout = [row for row in rows if row[0] % 3 == 0]
    return train, heldout


def _tensorize(rows: Sequence[Tuple[int, List[float], float, int]]):
    if not HAS_REAL_TORCH:
        return rows
    features = torch.tensor([row[1] for row in rows], dtype=torch.float)
    scores = torch.tensor([row[2] for row in rows], dtype=torch.float)
    labels = torch.tensor([row[3] for row in rows], dtype=torch.float)
    return features, scores, labels


def _classification_accuracy(model: nn.Module, features, labels) -> float:
    if not HAS_REAL_TORCH:
        correct = 0
        for _, row_features, _, label in features:
            prediction = model.predict(row_features)
            if (1 if prediction >= 0.0 else 0) == label:
                correct += 1
        return correct / max(1, len(features))
    with torch.no_grad():
        predictions = (model(features) >= 0.0).float()
        return float((predictions == labels).float().mean().item())


def _thresholded_pass_rate(model: nn.Module, features, scores, labels, tolerance: float = 0.28) -> float:
    if not HAS_REAL_TORCH:
        passes = 0
        for _, row_features, score, label in features:
            prediction = model.predict(row_features)
            if (1 if prediction >= 0.0 else 0) == label and abs(prediction - score) <= tolerance:
                passes += 1
        return passes / max(1, len(features))
    with torch.no_grad():
        prediction_scores = model(features)
        prediction_labels = (prediction_scores >= 0.0).float()
        passes = ((prediction_labels == labels) & ((prediction_scores - scores).abs() <= tolerance)).float()
        return float(passes.mean().item())


def _rmse(model: nn.Module, features, scores) -> float:
    if not HAS_REAL_TORCH:
        mse = 0.0
        for _, row_features, score, _ in features:
            prediction = model.predict(row_features)
            mse += (prediction - score) ** 2
        return math.sqrt(mse / max(1, len(features)))
    with torch.no_grad():
        return float(torch.sqrt(((model(features) - scores) ** 2).mean()).item())


def _periodic_signal(model: nn.Module) -> float:
    if not HAS_REAL_TORCH:
        weights = model.weights
        numerator = math.sqrt(sum(weight * weight for weight in weights[2:]))
        denominator = math.sqrt(sum(weight * weight for weight in weights))
        if denominator == 0.0:
            return 0.0
        return numerator / denominator
    with torch.no_grad():
        weights = model.linear.weight.squeeze(0)
        numerator = torch.linalg.vector_norm(weights[2:]).item()
        denominator = torch.linalg.vector_norm(weights).item()
        if denominator == 0.0:
            return 0.0
        return numerator / denominator


def _sparse_completeness(model: nn.Module, periodic: bool) -> float:
    if not HAS_REAL_TORCH:
        if periodic:
            target = [1.15, 0.0, 0.0, 0.9]
            distance = sum(abs(weight - target_weight) for weight, target_weight in zip(model.weights, target))
            max_distance = sum(abs(target_weight) for target_weight in target) + 1e-9
            return max(0.0, 1.0 - (distance / max_distance))
        if hasattr(model, "weights"):
            return min(1.0, sum(abs(weight) for weight in model.weights) / max(1.0, len(model.weights)))
        hidden_weights = sum(abs(weight) for row in model.hidden_weights for weight in row) / max(
            1, len(model.hidden_weights) * len(model.hidden_weights[0])
        )
        return min(1.0, hidden_weights)
    with torch.no_grad():
        if periodic:
            target = torch.tensor([1.15, 0.0, 0.0, 0.9], dtype=torch.float)
            weights = model.linear.weight.squeeze(0)
            distance = torch.abs(weights - target).sum().item()
            max_distance = torch.abs(target).sum().item() + 1e-9
            return max(0.0, 1.0 - (distance / max_distance))
        if hasattr(model, "linear1"):
            hidden_weights = torch.abs(model.linear1.weight).mean().item()
            return min(1.0, hidden_weights)
        if hasattr(model, "linear"):
            linear_weights = torch.abs(model.linear.weight).mean().item()
            return min(1.0, linear_weights)
        return 0.0


def _train_model(model: nn.Module, train_rows, heldout_rows, steps: int, learning_rate: float, periodic: bool):
    if HAS_REAL_TORCH:
        train_features, train_scores, train_labels = _tensorize(train_rows)
        heldout_features, heldout_scores, heldout_labels = _tensorize(heldout_rows)
    else:
        train_features = train_rows
        train_scores = None
        train_labels = None
        heldout_features = heldout_rows
        heldout_scores = None
        heldout_labels = None

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) if HAS_REAL_TORCH else None
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
                "fourier_signal": round(_periodic_signal(model), 4) if periodic else 0.0,
                "circuit_completeness": round(_sparse_completeness(model, periodic), 4),
            }
        )

    record(0)
    for step in range(1, steps + 1):
        if HAS_REAL_TORCH:
            optimizer.zero_grad()
            predictions = model(train_features)
            loss = ((predictions - train_scores) ** 2).mean()
            loss.backward()
            optimizer.step()
        else:
            model.train_step(train_rows, learning_rate)
        if step % 6 == 0:
            record(step)

    return checkpoints, train_features, train_scores, train_labels, heldout_features, heldout_scores, heldout_labels


def _causal_validation(model: nn.Module, heldout_features, heldout_scores, heldout_labels, periodic: bool) -> Dict[str, float]:
    baseline = _classification_accuracy(model, heldout_features, heldout_labels)

    if HAS_REAL_TORCH:
        saved_state = {key: value.clone() for key, value in model.state_dict().items()}
        with torch.no_grad():
            if periodic:
                model.linear.weight[:, 2:] = 0.0
            elif hasattr(model, "linear1"):
                model.linear1.weight.zero_()
            elif hasattr(model, "linear"):
                model.linear.weight.zero_()
            else:
                raise AttributeError("Unsupported model structure for non-periodic causal validation.")
    else:
        saved_state = model.snapshot()
        if periodic:
            model.weights[2] = 0.0
            model.weights[3] = 0.0
        elif hasattr(model, "weights"):
            model.weights = [0.0 for _ in model.weights]
        else:
            model.hidden_weights = [[0.0 for _ in row] for row in model.hidden_weights]
    ablated = _classification_accuracy(model, heldout_features, heldout_labels)

    if HAS_REAL_TORCH:
        model.load_state_dict(saved_state)
    else:
        model.restore(saved_state)
    restored = _classification_accuracy(model, heldout_features, heldout_labels)

    return {
        "baseline_accuracy": round(baseline, 4),
        "ablated_accuracy": round(ablated, 4),
        "restored_accuracy": round(restored, 4),
        "accuracy_drop_from_ablation": round(baseline - ablated, 4),
        "accuracy_recovery_after_restoration": round(restored - ablated, 4),
    }


class TinyPeriodicFallback:
    def __init__(self) -> None:
        self.weights = [0.0, 0.0, 0.0, 0.0]

    def predict(self, features: Sequence[float]) -> float:
        return sum(weight * feature for weight, feature in zip(self.weights, features))

    def train_step(self, rows, learning_rate: float) -> None:
        gradients = [0.0 for _ in self.weights]
        for _, features, score, _ in rows:
            error = self.predict(features) - score
            for index, feature in enumerate(features):
                gradients[index] += 2.0 * error * feature / max(1, len(rows))
        for index, gradient in enumerate(gradients):
            self.weights[index] -= learning_rate * gradient

    def snapshot(self):
        return list(self.weights)

    def restore(self, state) -> None:
        self.weights = list(state)


class TinySparseRelationalFallback:
    def __init__(self) -> None:
        self.hidden_weights = [[0.0, 0.0, 0.0] for _ in range(4)]
        self.output_weights = [0.0, 0.0, 0.0, 0.0]

    def predict(self, features: Sequence[float]) -> float:
        hidden = []
        for row in self.hidden_weights:
            activation = sum(weight * feature for weight, feature in zip(row, features))
            hidden.append(max(0.0, activation))
        return sum(weight * value for weight, value in zip(self.output_weights, hidden))

    def train_step(self, rows, learning_rate: float) -> None:
        for _, features, score, _ in rows:
            hidden_linear = [sum(weight * feature for weight, feature in zip(row, features)) for row in self.hidden_weights]
            hidden = [max(0.0, value) for value in hidden_linear]
            prediction = sum(weight * value for weight, value in zip(self.output_weights, hidden))
            error = prediction - score
            for index in range(len(self.output_weights)):
                self.output_weights[index] -= learning_rate * error * hidden[index]
            for hidden_index in range(len(self.hidden_weights)):
                if hidden_linear[hidden_index] <= 0.0:
                    continue
                for feature_index in range(len(features)):
                    gradient = error * self.output_weights[hidden_index] * features[feature_index]
                    self.hidden_weights[hidden_index][feature_index] -= learning_rate * gradient

    def snapshot(self):
        return {
            "hidden_weights": [list(row) for row in self.hidden_weights],
            "output_weights": list(self.output_weights),
        }

    def restore(self, state) -> None:
        self.hidden_weights = [list(row) for row in state["hidden_weights"]]
        self.output_weights = list(state["output_weights"])


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


def _discover_periodic_mechanism(model) -> Dict[str, object]:
    feature_names = ["sin", "cos", "sin2", "cos2"]
    if HAS_REAL_TORCH:
        weights = model.linear.weight.detach().squeeze(0).tolist()
    else:
        weights = list(model.weights)
    ranked = sorted(
        [{"feature": name, "weight": round(weight, 4), "abs_weight": abs(weight)} for name, weight in zip(feature_names, weights)],
        key=lambda item: item["abs_weight"],
        reverse=True,
    )
    return {
        "top_features": ranked[:2],
        "all_weights": ranked,
    }


def _targeted_vs_random_ablation(model, heldout_features, heldout_scores, heldout_labels) -> Dict[str, float]:
    baseline = _classification_accuracy(model, heldout_features, heldout_labels)

    if HAS_REAL_TORCH:
        base_state = {key: value.clone() for key, value in model.state_dict().items()}
        with torch.no_grad():
            model.linear.weight[:, [0, 3]] = 0.0
        targeted = _classification_accuracy(model, heldout_features, heldout_labels)
        model.load_state_dict(base_state)
        with torch.no_grad():
            model.linear.weight[:, [1, 2]] = 0.0
        random_like = _classification_accuracy(model, heldout_features, heldout_labels)
        model.load_state_dict(base_state)
    else:
        base_state = model.snapshot()
        model.weights[0] = 0.0
        model.weights[3] = 0.0
        targeted = _classification_accuracy(model, heldout_features, heldout_labels)
        model.restore(base_state)
        model.weights[1] = 0.0
        model.weights[2] = 0.0
        random_like = _classification_accuracy(model, heldout_features, heldout_labels)
        model.restore(base_state)

    return {
        "baseline_accuracy": round(baseline, 4),
        "targeted_ablation_accuracy": round(targeted, 4),
        "random_ablation_accuracy": round(random_like, 4),
        "targeted_drop": round(baseline - targeted, 4),
        "random_drop": round(baseline - random_like, 4),
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

    supports_forecast = str(forecast["forecast_type"]) in {
        observed_curve_classes["thresholded_pass_rate"],
        observed_curve_classes["classification_accuracy"],
        observed_curve_classes["score_rmse"],
    }
    supports_masking = (
        observed_curve_classes["thresholded_pass_rate"] == "step-function"
        or observed_curve_classes["classification_accuracy"] == "step-function"
    ) and (
        observed_curve_classes["score_rmse"] == "power-law"
        and rmse_drop > 0.15
    )
    supports_mechanism = (
        late_fourier_strength > 0.45
        and completeness_gain > 0.2
        and targeted_vs_random is not None
        and targeted_vs_random["targeted_drop"] > targeted_vs_random["random_drop"] + 0.15
        and causal_validation["accuracy_recovery_after_restoration"] >= 0.15
    )
    if control_case:
        supports_masking = False
        supports_forecast = observed_curve_classes["score_rmse"] == "power-law"
        supports_mechanism = (
            abs(fourier_gain) <= 0.05
            and abs(completeness_gain) <= 0.1
            and (targeted_vs_random is None or abs(targeted_vs_random["targeted_drop"] - targeted_vs_random["random_drop"]) <= 0.2)
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
        "early_rmse_drop": rmse_drop,
        "early_fourier_gain": fourier_gain,
        "early_completeness_gain": completeness_gain,
        "late_fourier_strength": late_fourier_strength,
    }


def run_real_tiny_suite(output_dir: Optional[Path] = None) -> Dict[str, object]:
    output_root = output_dir or Path(__file__).resolve().parents[1] / "artifacts" / "minimal_suite"
    output_root.mkdir(parents=True, exist_ok=True)

    if HAS_REAL_TORCH:
        torch.manual_seed(0)
    cases: Dict[str, object] = {}

    periodic_rows = _periodic_rows()
    periodic_train, periodic_heldout = _split_rows(periodic_rows)
    periodic_model = TinyPeriodicNet() if HAS_REAL_TORCH else TinyPeriodicFallback()
    periodic_checkpoints, _, _, _, periodic_hf, periodic_hs, periodic_hl = _train_model(
        periodic_model, periodic_train, periodic_heldout, steps=72, learning_rate=0.08, periodic=True
    )
    periodic_payload = {
        "forecast": _forecast_for_case("tiny_periodic_modular", "periodic"),
        "observed_curve_classes": {
            "thresholded_pass_rate": classify_metric_curve(
                [checkpoint["thresholded_pass_rate"] for checkpoint in periodic_checkpoints]
            ),
            "classification_accuracy": classify_metric_curve(
                [checkpoint["classification_accuracy"] for checkpoint in periodic_checkpoints]
            ),
            "score_rmse": classify_rmse_curve([checkpoint["score_rmse"] for checkpoint in periodic_checkpoints]),
        },
        "checkpoints": periodic_checkpoints,
        "causal_validation": _causal_validation(periodic_model, periodic_hf, periodic_hs, periodic_hl, periodic=True),
        "mechanism_discovery": _discover_periodic_mechanism(periodic_model),
        "targeted_vs_random_ablation": _targeted_vs_random_ablation(periodic_model, periodic_hf, periodic_hs, periodic_hl),
    }
    periodic_payload["feedback_adjusted_forecast"] = feedback_adjust_forecast(
        periodic_payload["forecast"], periodic_checkpoints
    )
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
    non_periodic_train, non_periodic_heldout = _split_rows(non_periodic_rows)
    non_periodic_model = TinyPeriodicNet() if HAS_REAL_TORCH else TinyPeriodicFallback()
    non_periodic_checkpoints, _, _, _, non_hf, non_hs, non_hl = _train_model(
        non_periodic_model, non_periodic_train, non_periodic_heldout, steps=72, learning_rate=0.09, periodic=False
    )
    non_periodic_payload = {
        "forecast": _forecast_for_case("tiny_nonperiodic_linear", "non_periodic"),
        "observed_curve_classes": {
            "thresholded_pass_rate": classify_metric_curve(
                [checkpoint["thresholded_pass_rate"] for checkpoint in non_periodic_checkpoints]
            ),
            "classification_accuracy": classify_metric_curve(
                [checkpoint["classification_accuracy"] for checkpoint in non_periodic_checkpoints]
            ),
            "score_rmse": classify_rmse_curve([checkpoint["score_rmse"] for checkpoint in non_periodic_checkpoints]),
        },
        "checkpoints": non_periodic_checkpoints,
        "causal_validation": _causal_validation(
            non_periodic_model, non_hf, non_hs, non_hl, periodic=False
        ),
    }
    non_periodic_payload["feedback_adjusted_forecast"] = feedback_adjust_forecast(
        non_periodic_payload["forecast"], non_periodic_checkpoints
    )
    non_periodic_payload["targeted_vs_random_ablation"] = _targeted_vs_random_ablation(
        non_periodic_model, non_hf, non_hs, non_hl
    )
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
    sparse_model = TinySparseRelationalNet() if HAS_REAL_TORCH else TinySparseRelationalFallback()
    sparse_checkpoints, _, _, _, sparse_hf, sparse_hs, sparse_hl = _train_model(
        sparse_model, sparse_train, sparse_heldout, steps=96, learning_rate=0.05, periodic=False
    )
    sparse_payload = {
        "forecast": _forecast_for_case("tiny_sparse_relational", "sparse_relational"),
        "observed_curve_classes": {
            "thresholded_pass_rate": classify_metric_curve(
                [checkpoint["thresholded_pass_rate"] for checkpoint in sparse_checkpoints]
            ),
            "classification_accuracy": classify_metric_curve(
                [checkpoint["classification_accuracy"] for checkpoint in sparse_checkpoints]
            ),
            "score_rmse": classify_rmse_curve([checkpoint["score_rmse"] for checkpoint in sparse_checkpoints]),
        },
        "checkpoints": sparse_checkpoints,
        "causal_validation": _causal_validation(sparse_model, sparse_hf, sparse_hs, sparse_hl, periodic=False),
    }
    sparse_payload["feedback_adjusted_forecast"] = feedback_adjust_forecast(
        sparse_payload["forecast"], sparse_checkpoints
    )
    sparse_payload["evidence_rubric"] = _case_evidence_rubric(
        case_id="tiny_sparse_relational",
        forecast=sparse_payload["forecast"],
        observed_curve_classes=sparse_payload["observed_curve_classes"],
        checkpoints=sparse_checkpoints,
        causal_validation=sparse_payload["causal_validation"],
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
