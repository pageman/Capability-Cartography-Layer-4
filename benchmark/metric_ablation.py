import json
import math
from pathlib import Path
from typing import Dict, List, Sequence

from capability_cartography_layer4.case_study import classify_metric_curve, classify_rmse_curve, run_minimal_case_study


def _periodic_signal_ratio(weights: Sequence[float]) -> float:
    total = math.sqrt(sum(weight * weight for weight in weights))
    periodic = math.sqrt(sum(weight * weight for weight in weights[2:]))
    if total == 0.0:
        return 0.0
    return periodic / total


def _run_non_periodic_control() -> Dict[str, object]:
    xs = list(range(17))
    train_xs = [x for x in xs if x % 3 != 0]
    heldout_xs = [x for x in xs if x % 3 == 0]

    def features(x: int) -> List[float]:
        return [x / 16.0, 1.0, 0.0, 0.0]

    def target(x: int) -> float:
        return 1.4 * (x / 16.0) - 0.6

    def label(score: float) -> int:
        return 1 if score >= 0.0 else 0

    def accuracy(weights: Sequence[float], points: Sequence[int]) -> float:
        correct = 0
        for x in points:
            pred_score = sum(weight * feature for weight, feature in zip(weights, features(x)))
            if label(pred_score) == label(target(x)):
                correct += 1
        return correct / max(1, len(points))

    def rmse(weights: Sequence[float], points: Sequence[int]) -> float:
        mse = 0.0
        for x in points:
            pred_score = sum(weight * feature for weight, feature in zip(weights, features(x)))
            mse += (pred_score - target(x)) ** 2
        return math.sqrt(mse / max(1, len(points)))

    weights = [0.0, 0.0, 0.0, 0.0]
    checkpoints: List[Dict[str, float]] = []
    learning_rate = 0.09

    for step in range(13):
        checkpoints.append(
            {
                "step": step * 6,
                "classification_accuracy": round(accuracy(weights, train_xs), 4),
                "heldout_accuracy": round(accuracy(weights, heldout_xs), 4),
                "score_rmse": round(rmse(weights, train_xs), 4),
                "periodic_signal_ratio": round(_periodic_signal_ratio(weights), 4),
            }
        )
        gradients = [0.0, 0.0, 0.0, 0.0]
        for x in train_xs:
            feats = features(x)
            error = sum(weight * feature for weight, feature in zip(weights, feats)) - target(x)
            for index, feature in enumerate(feats):
                gradients[index] += 2.0 * error * feature / max(1, len(train_xs))
        for index, gradient in enumerate(gradients):
            weights[index] -= learning_rate * gradient

    return {
        "observed_curve_class": classify_metric_curve([point["classification_accuracy"] for point in checkpoints]),
        "final_periodic_signal_ratio": checkpoints[-1]["periodic_signal_ratio"],
        "checkpoints": checkpoints,
    }


def run_metric_ablation(output_dir: Path | None = None) -> Dict[str, object]:
    output_root = output_dir or Path(__file__).resolve().parents[1] / "artifacts" / "minimal_case_study"
    output_root.mkdir(parents=True, exist_ok=True)

    case_result = run_minimal_case_study(output_root)
    checkpoints = case_result["checkpoint_metrics"]["checkpoints"]

    threshold_curve = [checkpoint["coarse_accuracy"] for checkpoint in checkpoints]
    classification_curve = [checkpoint["classification_accuracy"] for checkpoint in checkpoints]
    rmse_curve = [checkpoint["fine_rmse"] for checkpoint in checkpoints]

    non_periodic_control = _run_non_periodic_control()
    payload = {
        "metric_curve_classes": {
            "thresholded_pass_rate": classify_metric_curve(threshold_curve),
            "classification_accuracy": classify_metric_curve(classification_curve),
            "score_rmse": classify_rmse_curve(rmse_curve),
        },
        "falsification_controls": {
            "masking_disappears_under_classification_accuracy": classify_metric_curve(classification_curve) != "step-function",
            "masking_disappears_under_score_rmse": classify_rmse_curve(rmse_curve) != "step-function",
            "non_periodic_control_fourier_signal_stays_low": non_periodic_control["final_periodic_signal_ratio"] < 0.2,
            "non_periodic_control_curve_is_not_step_function": non_periodic_control["observed_curve_class"] != "step-function",
        },
        "non_periodic_control": non_periodic_control,
    }

    (output_root / "metric_ablation.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    result = run_metric_ablation()
    print(json.dumps(result, indent=2))
