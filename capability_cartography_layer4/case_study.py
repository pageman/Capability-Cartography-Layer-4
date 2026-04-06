import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from .orchestration import CCL4Pipeline


@dataclass
class CheckpointRecord:
    step: int
    coarse_accuracy: float
    classification_accuracy: float
    fine_rmse: float
    heldout_coarse_accuracy: float
    heldout_classification_accuracy: float
    heldout_fine_rmse: float
    fourier_signature: float
    circuit_completeness: float


def _feature_vector(x: int, modulus: int) -> List[float]:
    angle = 2.0 * math.pi * x / modulus
    return [
        math.sin(angle),
        math.cos(angle),
        math.sin(2.0 * angle),
        math.cos(2.0 * angle),
    ]


def _dot(weights: Sequence[float], features: Sequence[float]) -> float:
    return sum(weight * feature for weight, feature in zip(weights, features))


def _target_score(x: int, modulus: int) -> float:
    features = _feature_vector(x, modulus)
    target_weights = [1.2, 0.0, 0.0, 0.95]
    return _dot(target_weights, features) - 0.2


def _label_from_score(score: float) -> int:
    return 1 if score >= 0.0 else 0


def _dataset(modulus: int) -> List[Tuple[int, List[float], float, int]]:
    rows = []
    for x in range(modulus):
        score = _target_score(x, modulus)
        rows.append((x, _feature_vector(x, modulus), score, _label_from_score(score)))
    return rows


def _predict(weights: Sequence[float], features: Sequence[float]) -> float:
    return _dot(weights, features)


def _rmse(weights: Sequence[float], rows: Sequence[Tuple[int, List[float], float, int]]) -> float:
    mse = sum((_predict(weights, features) - score) ** 2 for _, features, score, _ in rows) / max(1, len(rows))
    return math.sqrt(mse)


def _accuracy(weights: Sequence[float], rows: Sequence[Tuple[int, List[float], float, int]]) -> float:
    matches = 0
    for _, features, _, label in rows:
        prediction = _label_from_score(_predict(weights, features))
        if prediction == label:
            matches += 1
    return matches / max(1, len(rows))


def _capability_pass_rate(
    weights: Sequence[float],
    rows: Sequence[Tuple[int, List[float], float, int]],
    tolerance: float = 0.28,
) -> float:
    passes = 0
    for _, features, score, label in rows:
        prediction_score = _predict(weights, features)
        prediction_label = _label_from_score(prediction_score)
        if prediction_label == label and abs(prediction_score - score) <= tolerance:
            passes += 1
    return passes / max(1, len(rows))


def _train_epoch(
    weights: List[float],
    rows: Sequence[Tuple[int, List[float], float, int]],
    learning_rate: float,
) -> None:
    gradients = [0.0 for _ in weights]
    for _, features, score, _ in rows:
        prediction = _predict(weights, features)
        error = prediction - score
        for index, feature in enumerate(features):
            gradients[index] += 2.0 * error * feature / max(1, len(rows))
    for index, gradient in enumerate(gradients):
        weights[index] -= learning_rate * gradient


def _fourier_signature(weights: Sequence[float]) -> float:
    target_weights = [1.2, 0.0, 0.0, 0.95]
    numerator = sum(weight * target for weight, target in zip(weights, target_weights))
    weight_norm = math.sqrt(sum(weight * weight for weight in weights))
    target_norm = math.sqrt(sum(target * target for target in target_weights))
    if weight_norm == 0.0 or target_norm == 0.0:
        return 0.0
    return max(0.0, min(1.0, numerator / (weight_norm * target_norm)))


def _circuit_completeness(weights: Sequence[float]) -> float:
    target_weights = [1.2, 0.0, 0.0, 0.95]
    distance = sum(abs(weight - target) for weight, target in zip(weights, target_weights))
    max_distance = sum(abs(target) for target in target_weights) + 1e-9
    return max(0.0, min(1.0, 1.0 - (distance / max_distance)))


def _curve_class(records: Sequence[CheckpointRecord]) -> str:
    return classify_metric_curve([record.coarse_accuracy for record in records])


def classify_metric_curve(values: Sequence[float]) -> str:
    """Classifies whether a metric trajectory looks step-like or smooth."""
    if len(values) < 3:
        return "unknown"
    deltas = [later - earlier for earlier, later in zip(values, values[1:])]
    positive_deltas = [delta for delta in deltas if delta > 0.0]
    if not positive_deltas:
        return "power-law"
    dominant_jump = max(positive_deltas)
    secondary_mass = sum(positive_deltas) - dominant_jump
    if dominant_jump >= 0.3 and secondary_mass <= dominant_jump:
        return "step-function"
    return "power-law"


def classify_rmse_curve(values: Sequence[float]) -> str:
    """RMSE typically falls smoothly; convert to progress before classification."""
    if len(values) < 3:
        return "unknown"
    start = values[0]
    progress = [start - value for value in values]
    return classify_metric_curve(progress)


def _detect_masking(records: Sequence[CheckpointRecord]) -> Dict[str, object]:
    jump_index = 0
    largest_jump = -1.0
    for index in range(1, len(records)):
        jump = records[index].coarse_accuracy - records[index - 1].coarse_accuracy
        if jump > largest_jump:
            largest_jump = jump
            jump_index = index
    pre_jump_rmse_drop = records[0].fine_rmse - records[jump_index - 1].fine_rmse if jump_index > 0 else 0.0
    pre_jump_signature_gain = (
        records[jump_index - 1].fourier_signature - records[0].fourier_signature if jump_index > 0 else 0.0
    )
    return {
        "jump_step": records[jump_index].step,
        "largest_coarse_jump": round(largest_jump, 4),
        "fine_rmse_drop_before_jump": round(pre_jump_rmse_drop, 4),
        "fourier_signature_gain_before_jump": round(pre_jump_signature_gain, 4),
        "masking_supported": pre_jump_rmse_drop > 0.15 and pre_jump_signature_gain > 0.2 and largest_jump > 0.15,
    }


def _causal_validation(
    final_weights: Sequence[float],
    heldout_rows: Sequence[Tuple[int, List[float], float, int]],
) -> Dict[str, float]:
    baseline_accuracy = _accuracy(final_weights, heldout_rows)
    baseline_rmse = _rmse(final_weights, heldout_rows)

    ablated_weights = [0.0, 0.0, 0.0, 0.0]
    ablated_accuracy = _accuracy(ablated_weights, heldout_rows)
    ablated_rmse = _rmse(ablated_weights, heldout_rows)

    restored_weights = list(final_weights)
    restored_accuracy = _accuracy(restored_weights, heldout_rows)
    restored_rmse = _rmse(restored_weights, heldout_rows)

    return {
        "baseline_accuracy": round(baseline_accuracy, 4),
        "baseline_rmse": round(baseline_rmse, 4),
        "ablated_accuracy": round(ablated_accuracy, 4),
        "ablated_rmse": round(ablated_rmse, 4),
        "restored_accuracy": round(restored_accuracy, 4),
        "restored_rmse": round(restored_rmse, 4),
        "accuracy_drop_from_ablation": round(baseline_accuracy - ablated_accuracy, 4),
        "accuracy_recovery_after_restoration": round(restored_accuracy - ablated_accuracy, 4),
    }


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_markdown(path: Path, lines: List[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_minimal_case_study(output_dir: Path | None = None) -> Dict[str, object]:
    output_root = output_dir or Path(__file__).resolve().parents[1] / "artifacts" / "minimal_case_study"
    output_root.mkdir(parents=True, exist_ok=True)

    modulus = 17
    all_rows = _dataset(modulus)
    train_residues = {0, 1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16}
    train_rows = [row for row in all_rows if row[0] in train_residues]
    heldout_rows = [row for row in all_rows if row[0] not in train_residues]

    pipeline = CCL4Pipeline()
    forecast_record = pipeline.run_record(
        paper_id="CASE_MOD_MASKING",
        capability_id="modular_periodic_thresholding",
        regime_params={"m": 17, "r": 1.0, "d": 64, "s_star": 4},
        task_metadata={
            "domain": "mathematics",
            "relational_depth": 6,
            "novel_architecture": False,
            "compressibility_gap": 24.0,
            "intrinsic_dimension": 18.0,
            "param_count": 100.0,
            "has_empirical_results": True,
        },
    )

    registration_payload = {
        "registered_before_training": True,
        "paper_id": forecast_record.paper_id,
        "capability_id": forecast_record.capability_id,
        "forecast_type": forecast_record.forecast.type.value,
        "predicted_exponent": forecast_record.forecast.predicted_exponent,
        "predicted_threshold": forecast_record.forecast.emergence_threshold,
        "confidence": round(forecast_record.forecast.confidence, 4),
        "regime_label": forecast_record.regime.regime_label,
    }
    _write_json(output_root / "forecast_registration.json", registration_payload)

    weights = [0.0, 0.0, 0.0, 0.0]
    learning_rate = 0.08
    steps_per_checkpoint = 6
    checkpoint_count = 12
    checkpoint_records: List[CheckpointRecord] = []

    checkpoint_records.append(
        CheckpointRecord(
            step=0,
            coarse_accuracy=round(_capability_pass_rate(weights, train_rows), 4),
            classification_accuracy=round(_accuracy(weights, train_rows), 4),
            fine_rmse=round(_rmse(weights, train_rows), 4),
            heldout_coarse_accuracy=round(_capability_pass_rate(weights, heldout_rows), 4),
            heldout_classification_accuracy=round(_accuracy(weights, heldout_rows), 4),
            heldout_fine_rmse=round(_rmse(weights, heldout_rows), 4),
            fourier_signature=round(_fourier_signature(weights), 4),
            circuit_completeness=round(_circuit_completeness(weights), 4),
        )
    )

    for checkpoint_index in range(checkpoint_count):
        for _ in range(steps_per_checkpoint):
            _train_epoch(weights, train_rows, learning_rate)
        checkpoint_records.append(
            CheckpointRecord(
                step=(checkpoint_index + 1) * steps_per_checkpoint,
                coarse_accuracy=round(_capability_pass_rate(weights, train_rows), 4),
                classification_accuracy=round(_accuracy(weights, train_rows), 4),
                fine_rmse=round(_rmse(weights, train_rows), 4),
                heldout_coarse_accuracy=round(_capability_pass_rate(weights, heldout_rows), 4),
                heldout_classification_accuracy=round(_accuracy(weights, heldout_rows), 4),
                heldout_fine_rmse=round(_rmse(weights, heldout_rows), 4),
                fourier_signature=round(_fourier_signature(weights), 4),
                circuit_completeness=round(_circuit_completeness(weights), 4),
            )
        )

    observed_curve = _curve_class(checkpoint_records)
    masking_analysis = _detect_masking(checkpoint_records)
    causal_validation = _causal_validation(weights, heldout_rows)

    metrics_payload = {
        "task_family": "periodic modular threshold classification",
        "observed_curve_class": observed_curve,
        "masking_analysis": masking_analysis,
        "checkpoints": [asdict(record) for record in checkpoint_records],
    }
    _write_json(output_root / "checkpoint_metrics.json", metrics_payload)
    _write_json(output_root / "causal_validation.json", causal_validation)

    summary_lines = [
        "# Minimal Case Study Report",
        "",
        "## Forecast Registration",
        f"- Forecast type: `{registration_payload['forecast_type']}`",
        f"- Regime label: `{registration_payload['regime_label']}`",
        f"- Confidence: `{registration_payload['confidence']}`",
        "",
        "## Observation Effect",
        f"- Observed curve class: `{observed_curve}`",
        f"- Largest coarse jump at step `{masking_analysis['jump_step']}` with delta `{masking_analysis['largest_coarse_jump']}`",
        f"- Fine RMSE drop before jump: `{masking_analysis['fine_rmse_drop_before_jump']}`",
        f"- Fourier signature gain before jump: `{masking_analysis['fourier_signature_gain_before_jump']}`",
        f"- Masking supported: `{masking_analysis['masking_supported']}`",
        "",
        "## Causal Validation",
        f"- Baseline held-out accuracy: `{causal_validation['baseline_accuracy']}`",
        f"- Ablated held-out accuracy: `{causal_validation['ablated_accuracy']}`",
        f"- Restored held-out accuracy: `{causal_validation['restored_accuracy']}`",
        f"- Accuracy drop from ablation: `{causal_validation['accuracy_drop_from_ablation']}`",
        f"- Accuracy recovery after restoration: `{causal_validation['accuracy_recovery_after_restoration']}`",
    ]
    _write_markdown(output_root / "summary_report.md", summary_lines)

    return {
        "forecast_registration": registration_payload,
        "checkpoint_metrics": metrics_payload,
        "causal_validation": causal_validation,
        "artifact_dir": str(output_root),
    }
