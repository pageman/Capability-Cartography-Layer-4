import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:  # pragma: no cover - import fallback for script execution
    from .case_study import classify_metric_curve, classify_rmse_curve
    from .checkpointed_attention_discovery import RealCircuitDiscoveryResult, discover_stable_attention_circuit
except ImportError:  # pragma: no cover
    from capability_cartography_layer4.case_study import classify_metric_curve, classify_rmse_curve
    from capability_cartography_layer4.checkpointed_attention_discovery import (
        RealCircuitDiscoveryResult,
        discover_stable_attention_circuit,
    )


@dataclass
class SmallTransformerCheckpointPlan:
    """Narrow plan for a checkpointed small-transformer benchmark case."""

    model_name: str
    task_family: str
    checkpoint_dir: Path
    heldout_split_name: str
    metric_names: List[str] = field(default_factory=list)
    mechanism_signal_names: List[str] = field(default_factory=list)


@dataclass
class SmallTransformerCaseResult:
    """Result schema for the checkpointed small-transformer benchmark case."""

    forecast: Dict[str, object]
    checkpoints_loaded: int
    checkpoint_metrics_path: Optional[str] = None
    mechanism_discovery_path: Optional[str] = None
    causal_validation_path: Optional[str] = None
    deliverable_manifest_path: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class TinyAttentionSequenceModel:
    """
    Small attention-style sequence model implemented in pure Python.

    This is intentionally narrow: one query route attends over a length-3 token
    sequence and regresses a periodic target score.
    """

    def __init__(self, vocab_size: int = 17, dim: int = 4) -> None:
        self.vocab_size = vocab_size
        self.dim = dim
        self.embeddings = []
        for token in range(vocab_size):
            angle = 2.0 * math.pi * token / max(1, vocab_size)
            self.embeddings.append(
                [
                    math.sin(angle),
                    math.cos(angle),
                    math.sin(2.0 * angle),
                    math.cos(2.0 * angle),
                ]
            )
        self.query = [0.12, -0.03, 0.07, 0.05]
        self.readout = [0.11, 0.02, -0.04, 0.09]
        self.bias = 0.0
        self.position_encoding = [
            [0.35, 0.0, 0.0, 0.0],
            [0.0, 0.25, 0.0, 0.0],
            [0.0, 0.0, 0.18, 0.0],
        ]

    def snapshot(self) -> Dict[str, object]:
        return {
            "embeddings": [list(row) for row in self.embeddings],
            "query": list(self.query),
            "readout": list(self.readout),
            "bias": self.bias,
        }

    def restore(self, state: Dict[str, object]) -> None:
        self.embeddings = [list(row) for row in state["embeddings"]]
        self.query = list(state["query"])
        self.readout = list(state["readout"])
        self.bias = float(state["bias"])

    @staticmethod
    def _dot(left: Sequence[float], right: Sequence[float]) -> float:
        return sum(a * b for a, b in zip(left, right))

    @staticmethod
    def _add(left: Sequence[float], right: Sequence[float]) -> List[float]:
        return [a + b for a, b in zip(left, right)]

    def _encode(self, sequence: Sequence[int]) -> List[List[float]]:
        encoded: List[List[float]] = []
        for position, token in enumerate(sequence):
            encoded.append(self._add(self.embeddings[token], self.position_encoding[position]))
        return encoded

    def forward(self, sequence: Sequence[int], masked_positions: Optional[Sequence[int]] = None) -> Dict[str, object]:
        encoded = self._encode(sequence)
        scores: List[float] = []
        mask = set(masked_positions or [])
        for position, vector in enumerate(encoded):
            if position in mask:
                scores.append(-1e9)
            else:
                scores.append(self._dot(self.query, vector))
        max_score = max(scores)
        exps = [math.exp(score - max_score) if score > -1e8 else 0.0 for score in scores]
        denom = sum(exps) or 1.0
        attention = [value / denom for value in exps]
        context = [
            sum(attention[position] * encoded[position][axis] for position in range(len(encoded)))
            for axis in range(self.dim)
        ]
        output = self._dot(self.readout, context) + self.bias
        return {
            "score": output,
            "attention": attention,
            "context": context,
            "encoded": encoded,
        }

    def train_step(self, sequence: Sequence[int], target_score: float, learning_rate: float) -> None:
        result = self.forward(sequence)
        prediction = float(result["score"])
        attention = list(result["attention"])
        context = list(result["context"])
        encoded = list(result["encoded"])
        error = prediction - target_score
        output_grad = 2.0 * error
        context_grad = [output_grad * weight for weight in self.readout]
        readout_grad = [output_grad * value for value in context]
        bias_grad = output_grad

        attention_value_grads = [sum(context_grad[axis] * encoded[position][axis] for axis in range(self.dim)) for position in range(3)]
        average_attention_grad = sum(attention[position] * attention_value_grads[position] for position in range(3))
        score_grads = [
            attention[position] * (attention_value_grads[position] - average_attention_grad) for position in range(3)
        ]

        query_grad = [0.0 for _ in range(self.dim)]
        for position in range(3):
            for axis in range(self.dim):
                query_grad[axis] += score_grads[position] * encoded[position][axis]

        for axis in range(self.dim):
            self.query[axis] -= learning_rate * query_grad[axis]
            self.readout[axis] -= learning_rate * readout_grad[axis]
        self.bias -= learning_rate * bias_grad


def default_small_transformer_plan(base_dir: Path) -> SmallTransformerCheckpointPlan:
    checkpoint_root = base_dir if base_dir.name == "small_transformer_case" else base_dir / "small_transformer_case"
    return SmallTransformerCheckpointPlan(
        model_name="tiny-attention-sequence-model",
        task_family="small_transformer_circuit",
        checkpoint_dir=checkpoint_root,
        heldout_split_name="residue_mod_3_holdout",
        metric_names=[
            "thresholded_pass_rate",
            "classification_accuracy",
            "score_rmse",
        ],
        mechanism_signal_names=[
            "stable_edge_overlap",
            "targeted_ablation_drop",
            "random_ablation_drop",
            "restoration_recovery",
        ],
    )


def _periodic_rows() -> List[Tuple[int, List[int], float, int]]:
    rows: List[Tuple[int, List[int], float, int]] = []
    modulus = 17
    for x in range(modulus):
        sequence = [x, (2 * x) % modulus, (4 * x) % modulus]
        angle_pos_1 = 2.0 * math.pi * sequence[1] / modulus
        angle_pos_2 = 2.0 * math.pi * sequence[2] / modulus
        score = 0.9 * math.sin(angle_pos_2) + 0.35 * math.cos(angle_pos_1) - 0.05
        label = 1 if score >= 0.0 else 0
        rows.append((x, sequence, score, label))
    return rows


def _smooth_rows() -> List[Tuple[int, List[int], float, int]]:
    rows: List[Tuple[int, List[int], float, int]] = []
    modulus = 17
    for x in range(modulus):
        sequence = [x, min(modulus - 1, x + 1), min(modulus - 1, x + 2)]
        normalized = x / (modulus - 1)
        score = 1.1 * normalized - 0.45
        label = 1 if score >= 0.0 else 0
        rows.append((x, sequence, score, label))
    return rows


def _split_rows(rows: Sequence[Tuple[int, List[int], float, int]]) -> Tuple[List, List]:
    train = [row for row in rows if row[0] % 3 != 0]
    heldout = [row for row in rows if row[0] % 3 == 0]
    return train, heldout


def _classification_accuracy(model: TinyAttentionSequenceModel, rows: Sequence[Tuple[int, List[int], float, int]], masked_positions=None) -> float:
    correct = 0
    for _, sequence, _, label in rows:
        prediction = model.forward(sequence, masked_positions=masked_positions)["score"]
        if (1 if prediction >= 0.0 else 0) == label:
            correct += 1
    return correct / max(1, len(rows))


def _thresholded_pass_rate(
    model: TinyAttentionSequenceModel,
    rows: Sequence[Tuple[int, List[int], float, int]],
    tolerance: float = 0.25,
    masked_positions=None,
) -> float:
    passes = 0
    for _, sequence, score, label in rows:
        prediction = model.forward(sequence, masked_positions=masked_positions)["score"]
        if (1 if prediction >= 0.0 else 0) == label and abs(prediction - score) <= tolerance:
            passes += 1
    return passes / max(1, len(rows))


def _rmse(model: TinyAttentionSequenceModel, rows: Sequence[Tuple[int, List[int], float, int]], masked_positions=None) -> float:
    mse = 0.0
    for _, sequence, score, _ in rows:
        prediction = model.forward(sequence, masked_positions=masked_positions)["score"]
        mse += (prediction - score) ** 2
    return math.sqrt(mse / max(1, len(rows)))


def _average_attention(model: TinyAttentionSequenceModel, rows: Sequence[Tuple[int, List[int], float, int]]) -> List[float]:
    totals = [0.0, 0.0, 0.0]
    for _, sequence, _, _ in rows:
        attention = model.forward(sequence)["attention"]
        for position in range(3):
            totals[position] += attention[position]
    return [total / max(1, len(rows)) for total in totals]


def _checkpoint_record(model: TinyAttentionSequenceModel, step: int, train_rows, heldout_rows) -> Dict[str, object]:
    return {
        "step": step,
        "thresholded_pass_rate": round(_thresholded_pass_rate(model, train_rows), 4),
        "classification_accuracy": round(_classification_accuracy(model, train_rows), 4),
        "score_rmse": round(_rmse(model, train_rows), 4),
        "heldout_thresholded_pass_rate": round(_thresholded_pass_rate(model, heldout_rows), 4),
        "heldout_classification_accuracy": round(_classification_accuracy(model, heldout_rows), 4),
        "heldout_score_rmse": round(_rmse(model, heldout_rows), 4),
        "avg_attention_by_position": [round(value, 4) for value in _average_attention(model, train_rows)],
    }


def _train_small_transformer_case(
    rows: Sequence[Tuple[int, List[int], float, int]],
    steps: int = 180,
    learning_rate: float = 0.06,
) -> Tuple[TinyAttentionSequenceModel, List[Dict[str, object]], List, List]:
    train_rows, heldout_rows = _split_rows(rows)
    model = TinyAttentionSequenceModel()
    checkpoints = [_checkpoint_record(model, 0, train_rows, heldout_rows)]
    for step in range(1, steps + 1):
        for _, sequence, target_score, _ in train_rows:
            model.train_step(sequence, target_score, learning_rate)
        if step % 12 == 0:
            checkpoints.append(_checkpoint_record(model, step, train_rows, heldout_rows))
    return model, checkpoints, train_rows, heldout_rows


def _evaluate_discovery(
    model: TinyAttentionSequenceModel,
    discovery: RealCircuitDiscoveryResult,
    heldout_rows: Sequence[Tuple[int, List[int], float, int]],
) -> Dict[str, float]:
    baseline = _classification_accuracy(model, heldout_rows)
    targeted_positions = [int(discovery.discovered_operations[0].metadata["position"])] if discovery.discovered_operations else [0]
    all_positions = [0, 1, 2]
    random_positions = [position for position in all_positions if position not in targeted_positions]
    if not random_positions:
        random_positions = [all_positions[-1]]

    saved_state = model.snapshot()
    targeted_accuracy = _classification_accuracy(model, heldout_rows, masked_positions=targeted_positions)
    model.restore(saved_state)
    random_accuracy = _classification_accuracy(model, heldout_rows, masked_positions=[random_positions[0]])
    model.restore(saved_state)
    restored_accuracy = _classification_accuracy(model, heldout_rows)

    targeted_drop = baseline - targeted_accuracy
    random_drop = baseline - random_accuracy
    restoration = restored_accuracy - targeted_accuracy

    discovery.targeted_ablation_drop = targeted_drop
    discovery.random_ablation_drop = random_drop
    discovery.restoration_recovery = restoration

    return {
        "baseline_accuracy": round(baseline, 4),
        "targeted_ablation_accuracy": round(targeted_accuracy, 4),
        "random_ablation_accuracy": round(random_accuracy, 4),
        "restored_accuracy": round(restored_accuracy, 4),
        "targeted_drop": round(targeted_drop, 4),
        "random_drop": round(random_drop, 4),
        "restoration_recovery": round(restoration, 4),
        "claim_coverage": "low-to-medium",
        "failure_modes": [
            "matched random ablation can be as destructive as targeted ablation on some runs",
            "restoration recovery can be weak even when route stability is high",
            "held-out degradation is diagnostic only for this synthetic family",
        ],
    }


def _write_summary_report(
    output_dir: Path,
    plan: SmallTransformerCheckpointPlan,
    family_reports: Sequence[Dict[str, object]],
) -> None:
    report = [
        "# Small-Transformer Checkpointed Discovery Report",
        "",
        f"- model: `{plan.model_name}`",
        f"- task family collection: `{plan.task_family}`",
        f"- held-out split: `{plan.heldout_split_name}`",
        "",
        "This artifact supports narrow claims only: checkpointed small-transformer discovery on synthetic benchmark families.",
        "",
    ]
    for family in family_reports:
        report.extend(
            [
                f"## {family['family_id']}",
                f"- forecast: `{family['forecast']['forecast_type']}`",
                f"- observed thresholded curve: `{family['observed_curve_classes']['thresholded_pass_rate']}`",
                f"- observed rmse curve: `{family['observed_curve_classes']['score_rmse']}`",
                f"- final held-out accuracy: `{family['checkpoints'][-1]['heldout_classification_accuracy']:.4f}`",
                f"- stable overlap score: `{family['mechanism_discovery']['stable_overlap_score']:.2f}`",
                f"- targeted ablation drop: `{family['causal_validation']['targeted_drop']:.4f}`",
                f"- random ablation drop: `{family['causal_validation']['random_drop']:.4f}`",
                f"- restoration recovery: `{family['causal_validation']['restoration_recovery']:.4f}`",
                "",
                f"- claim coverage: `{family['claim_coverage']}`",
                f"- failure modes: {', '.join(family['failure_modes'])}",
                "",
            ]
        )
    (output_dir / "summary_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def _family_payload(
    family_id: str,
    forecast_type: str,
    confidence: float,
    rows: Sequence[Tuple[int, List[int], float, int]],
) -> Dict[str, object]:
    model, checkpoints, _, heldout_rows = _train_small_transformer_case(rows)
    discovery = discover_stable_attention_circuit(checkpoints, family_id=family_id)
    causal_validation = _evaluate_discovery(model, discovery, heldout_rows)
    return {
        "family_id": family_id,
        "forecast": {
            "forecast_type": forecast_type,
            "confidence": confidence,
            "scope": "small-transformer synthetic benchmark family",
        },
        "observed_curve_classes": {
            "thresholded_pass_rate": classify_metric_curve([checkpoint["thresholded_pass_rate"] for checkpoint in checkpoints]),
            "classification_accuracy": classify_metric_curve([checkpoint["classification_accuracy"] for checkpoint in checkpoints]),
            "score_rmse": classify_rmse_curve([checkpoint["score_rmse"] for checkpoint in checkpoints]),
        },
        "checkpoints": checkpoints,
        "mechanism_discovery": discovery.to_dict(),
        "causal_validation": causal_validation,
        "claim_coverage": "low-to-medium",
        "failure_modes": [
            "two-family synthetic results still do not establish general transformer mechanism laws",
            "synthetic sequences remain much simpler than realistic model workloads",
            "attention-route extraction is a narrow proxy for circuit discovery",
        ],
    }


def _write_deliverable_manifest(output_dir: Path, family_reports: Sequence[Dict[str, object]]) -> Path:
    manifest = {
        "deliverable_id": "ccl4-small-transformer-observability-v1",
        "artifact_root": str(output_dir),
        "families": [
            {
                "family_id": family["family_id"],
                "forecast_type": family["forecast"]["forecast_type"],
                "claim_coverage": family["claim_coverage"],
                "files": [
                    "checkpoint_metrics.json",
                    "circuit_discovery.json",
                    "causal_validation.json",
                ],
            }
            for family in family_reports
        ],
        "files": [
            "checkpoint_metrics.json",
            "circuit_discovery.json",
            "causal_validation.json",
            "deliverable_manifest.json",
            "summary_report.md",
        ],
        "notes": [
            "This manifest freezes provenance for the small-transformer evidence package.",
            "The package is observational and benchmark-scoped, not a correctness certificate.",
        ],
    }
    path = output_dir / "deliverable_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def run_small_transformer_case(base_dir: Path | None = None) -> SmallTransformerCaseResult:
    root = base_dir or Path(__file__).resolve().parents[1] / "artifacts" / "small_transformer_case"
    plan = default_small_transformer_plan(root)
    plan.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    family_reports = [
        _family_payload(
            family_id="periodic_modular_attention",
            forecast_type="step-function",
            confidence=0.9,
            rows=_periodic_rows(),
        ),
        _family_payload(
            family_id="smooth_monotonic_attention",
            forecast_type="power-law",
            confidence=0.93,
            rows=_smooth_rows(),
        ),
    ]

    checkpoint_metrics = {
        "claim_coverage": "low-to-medium",
        "failure_modes": [
            "two synthetic families are still too small for broad generalization claims",
            "curve classes depend on handcrafted metrics and synthetic targets",
            "family comparison is useful for scope control, not for universal theory",
        ],
        "families": family_reports,
    }

    mechanism_discovery = {
        "claim_coverage": "low-to-medium",
        "failure_modes": [
            "stable attention-route extraction does not imply full circuit recovery",
            "discovered routes are family-local and may not transfer",
        ],
        "families": [family["mechanism_discovery"] for family in family_reports],
    }

    causal_validation = {
        "claim_coverage": "low",
        "failure_modes": [
            "ablation comparisons are narrow diagnostics only",
            "restoration recovery can be weak or mixed even when routes are stable",
        ],
        "families": [family["causal_validation"] for family in family_reports],
    }

    checkpoint_metrics_path = plan.checkpoint_dir / "checkpoint_metrics.json"
    mechanism_discovery_path = plan.checkpoint_dir / "circuit_discovery.json"
    causal_validation_path = plan.checkpoint_dir / "causal_validation.json"

    checkpoint_metrics_path.write_text(json.dumps(checkpoint_metrics, indent=2) + "\n", encoding="utf-8")
    mechanism_discovery_path.write_text(json.dumps(mechanism_discovery, indent=2) + "\n", encoding="utf-8")
    causal_validation_path.write_text(json.dumps(causal_validation, indent=2) + "\n", encoding="utf-8")
    _write_summary_report(plan.checkpoint_dir, plan, family_reports)
    deliverable_manifest_path = _write_deliverable_manifest(plan.checkpoint_dir, family_reports)

    return SmallTransformerCaseResult(
        forecast={
            "forecast_type": "family_bundle",
            "confidence": 0.92,
            "scope": "two synthetic small-transformer benchmark families",
        },
        checkpoints_loaded=sum(len(family["checkpoints"]) for family in family_reports),
        checkpoint_metrics_path=str(checkpoint_metrics_path),
        mechanism_discovery_path=str(mechanism_discovery_path),
        causal_validation_path=str(causal_validation_path),
        deliverable_manifest_path=str(deliverable_manifest_path),
        notes=[
            "This is a narrow implemented small-transformer evidence bundle, not a frontier-model experiment.",
            "Circuit discovery is position-level attention-route extraction with explicit family-level scope and failure modes.",
        ],
    )


if __name__ == "__main__":
    print(json.dumps(run_small_transformer_case().to_dict(), indent=2))
