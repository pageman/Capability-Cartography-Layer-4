import json
from pathlib import Path
from typing import Dict, List, Tuple

from capability_cartography_layer4.orchestration import CCL4Pipeline


BENCHMARK_ROWS: List[Dict[str, object]] = [
    {
        "case_id": "modular_periodic_thresholding",
        "split": "train",
        "label": "step-function",
        "regime_params": {"m": 17, "r": 1.0, "d": 64, "s_star": 4},
        "task_metadata": {
            "domain": "mathematics",
            "relational_depth": 6,
            "novel_architecture": False,
            "compressibility_gap": 24.0,
            "param_count": 100.0,
            "intrinsic_dimension": 18.0,
            "periodic_task": 1.0,
        },
    },
    {
        "case_id": "smooth_language_scaling",
        "split": "train",
        "label": "power-law",
        "regime_params": {"m": 64, "r": 50.0, "d": 512, "s_star": 32},
        "task_metadata": {
            "domain": "language",
            "relational_depth": 2,
            "novel_architecture": False,
            "compressibility_gap": 2.0,
            "param_count": 8000.0,
            "intrinsic_dimension": 12.0,
            "periodic_task": 0.0,
        },
    },
    {
        "case_id": "theoretical_mdl_with_empirics",
        "split": "train",
        "label": "power-law",
        "regime_params": {"m": 0, "r": 0.0, "d": 128, "s_star": 12},
        "task_metadata": {
            "domain": "vision",
            "relational_depth": 1,
            "novel_architecture": False,
            "compressibility_gap": 3.0,
            "param_count": 300.0,
            "intrinsic_dimension": 8.0,
            "has_empirical_results": True,
            "periodic_task": 0.0,
        },
    },
    {
        "case_id": "novel_relational_sparse_task",
        "split": "train",
        "label": "emergent",
        "regime_params": {"m": 32, "r": 2.0, "d": 256, "s_star": 6},
        "task_metadata": {
            "domain": "reasoning",
            "relational_depth": 7,
            "novel_architecture": True,
            "compressibility_gap": 12.0,
            "param_count": 1200.0,
            "intrinsic_dimension": 90.0,
            "periodic_task": 0.0,
        },
    },
    {
        "case_id": "hybrid_moe_retrieval",
        "split": "test",
        "label": "hybrid",
        "regime_params": {"m": 100, "r": 50.0, "d": 768, "s_star": 10},
        "task_metadata": {
            "domain": "retrieval",
            "relational_depth": 3,
            "novel_architecture": False,
            "compressibility_gap": 8.0,
            "param_count": 6000.0,
            "intrinsic_dimension": 70.0,
            "moe_gate": True,
            "periodic_task": 0.0,
        },
    },
    {
        "case_id": "hardware_threshold_vision",
        "split": "test",
        "label": "step-function",
        "regime_params": {"m": 24, "r": 4.0, "d": 96, "s_star": 10},
        "task_metadata": {
            "domain": "vision",
            "relational_depth": 4,
            "novel_architecture": False,
            "compressibility_gap": 28.0,
            "hardware_constrained": True,
            "param_count": 900.0,
            "intrinsic_dimension": 120.0,
            "periodic_task": 0.0,
        },
    },
]

LABELS = ["power-law", "emergent", "step-function", "hybrid"]


def _majority_label(train_rows: List[Dict[str, object]]) -> str:
    counts: Dict[str, int] = {}
    for row in train_rows:
        label = str(row["label"])
        counts[label] = counts.get(label, 0) + 1
    return max(counts.items(), key=lambda item: item[1])[0]


def _param_count_only_predict(train_rows: List[Dict[str, object]], row: Dict[str, object]) -> str:
    class_means: Dict[str, float] = {}
    class_counts: Dict[str, int] = {}
    for train_row in train_rows:
        label = str(train_row["label"])
        param_count = float(train_row["task_metadata"]["param_count"])
        class_means[label] = class_means.get(label, 0.0) + param_count
        class_counts[label] = class_counts.get(label, 0) + 1
    for label in class_means:
        class_means[label] /= class_counts[label]
    current = float(row["task_metadata"]["param_count"])
    return min(class_means.items(), key=lambda item: abs(current - item[1]))[0]


def _features(row: Dict[str, object]) -> List[float]:
    metadata = row["task_metadata"]
    return [
        float(metadata.get("param_count", 0.0)) / 1000.0,
        float(metadata.get("relational_depth", 0.0)) / 10.0,
        float(metadata.get("compressibility_gap", 0.0)) / 30.0,
        1.0 if bool(metadata.get("novel_architecture", False)) else 0.0,
        1.0 if bool(metadata.get("moe_gate", False)) else 0.0,
        float(metadata.get("periodic_task", 0.0)),
        1.0,
    ]


def _metadata_linear_fit(train_rows: List[Dict[str, object]], epochs: int = 40) -> Dict[str, List[float]]:
    weights = {label: [0.0 for _ in _features(train_rows[0])] for label in LABELS}
    for _ in range(epochs):
        for row in train_rows:
            features = _features(row)
            truth = str(row["label"])
            predicted = _metadata_linear_predict(weights, row)
            if predicted == truth:
                continue
            for index, feature in enumerate(features):
                weights[truth][index] += feature
                weights[predicted][index] -= feature
    return weights


def _metadata_linear_predict(weights: Dict[str, List[float]], row: Dict[str, object]) -> str:
    features = _features(row)
    scores: Dict[str, float] = {}
    for label, label_weights in weights.items():
        scores[label] = sum(weight * feature for weight, feature in zip(label_weights, features))
    return max(scores.items(), key=lambda item: item[1])[0]


def _evaluate(rows: List[Dict[str, object]], predictor) -> Tuple[float, List[Dict[str, str]]]:
    correct = 0
    outputs: List[Dict[str, str]] = []
    for row in rows:
        prediction = predictor(row)
        truth = str(row["label"])
        outputs.append({"case_id": str(row["case_id"]), "truth": truth, "prediction": prediction})
        if prediction == truth:
            correct += 1
    return correct / max(1, len(rows)), outputs


def run_baseline_benchmark(output_dir: Path | None = None) -> Dict[str, object]:
    output_root = output_dir or Path(__file__).resolve().parents[1] / "artifacts" / "minimal_benchmark"
    output_root.mkdir(parents=True, exist_ok=True)

    train_rows = [row for row in BENCHMARK_ROWS if row["split"] == "train"]
    test_rows = [row for row in BENCHMARK_ROWS if row["split"] == "test"]

    majority = _majority_label(train_rows)
    majority_accuracy, majority_outputs = _evaluate(test_rows, lambda row: majority)
    param_accuracy, param_outputs = _evaluate(test_rows, lambda row: _param_count_only_predict(train_rows, row))

    linear_weights = _metadata_linear_fit(train_rows)
    linear_accuracy, linear_outputs = _evaluate(test_rows, lambda row: _metadata_linear_predict(linear_weights, row))

    pipeline = CCL4Pipeline()
    ccl4_accuracy, ccl4_outputs = _evaluate(
        test_rows,
        lambda row: pipeline.run_record(
            paper_id=str(row["case_id"]),
            capability_id=str(row["case_id"]),
            regime_params=row["regime_params"],
            task_metadata=row["task_metadata"],
        ).forecast.type.value,
    )

    payload = {
        "benchmark_rows": len(BENCHMARK_ROWS),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "results": {
            "majority": {"accuracy": round(majority_accuracy, 4), "predictions": majority_outputs},
            "param_count_only": {"accuracy": round(param_accuracy, 4), "predictions": param_outputs},
            "metadata_linear": {"accuracy": round(linear_accuracy, 4), "predictions": linear_outputs},
            "ccl4_rule_set": {"accuracy": round(ccl4_accuracy, 4), "predictions": ccl4_outputs},
        },
    }

    (output_root / "baseline_results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    result = run_baseline_benchmark()
    for name, values in result["results"].items():
        print(f"{name}: {values['accuracy']}")
