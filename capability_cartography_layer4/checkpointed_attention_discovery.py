from dataclasses import asdict, dataclass, field
from typing import Dict, List, Sequence


@dataclass
class StableOperation:
    """Discovered stable operation for a small checkpointed attention model."""

    op_id: str
    layer: int
    component: str
    score: float
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class RealCircuitDiscoveryResult:
    """Artifact schema for narrow checkpointed circuit discovery."""

    family_id: str = ""
    discovered_operations: List[StableOperation] = field(default_factory=list)
    stable_overlap_score: float = 0.0
    targeted_ablation_drop: float = 0.0
    random_ablation_drop: float = 0.0
    restoration_recovery: float = 0.0
    claim_coverage: str = "low"
    failure_modes: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "family_id": self.family_id,
            "discovered_operations": [asdict(op) for op in self.discovered_operations],
            "stable_overlap_score": round(self.stable_overlap_score, 4),
            "targeted_ablation_drop": round(self.targeted_ablation_drop, 4),
            "random_ablation_drop": round(self.random_ablation_drop, 4),
            "restoration_recovery": round(self.restoration_recovery, 4),
            "claim_coverage": self.claim_coverage,
            "failure_modes": list(self.failure_modes),
            "notes": list(self.notes),
        }


def _top_positions(checkpoint: Dict[str, object], top_k: int) -> List[int]:
    attention = checkpoint.get("avg_attention_by_position", [])
    ranked = sorted(range(len(attention)), key=lambda index: attention[index], reverse=True)
    return ranked[:top_k]


def discover_stable_attention_circuit(
    checkpoints: Sequence[Dict[str, object]],
    top_k: int = 2,
    family_id: str = "",
) -> RealCircuitDiscoveryResult:
    """
    Extract a narrow checkpointed circuit summary from position-level attention traces.

    The discovery claim is deliberately scoped:
    - stable operations are position-focused attention routes, not full ACDC/SAE graphs
    - stability is measured by top-k overlap across the last few checkpoints
    """

    if not checkpoints:
        return RealCircuitDiscoveryResult(
            family_id=family_id,
            failure_modes=["no checkpoints were provided"],
            notes=["No checkpoints were provided."],
        )

    tail = list(checkpoints[-3:]) if len(checkpoints) >= 3 else list(checkpoints)
    top_sets = [set(_top_positions(checkpoint, top_k)) for checkpoint in tail]
    intersection = set.intersection(*top_sets) if top_sets else set()
    union = set.union(*top_sets) if top_sets else set()
    overlap = len(intersection) / max(1, len(union))

    final_checkpoint = checkpoints[-1]
    final_attention = final_checkpoint.get("avg_attention_by_position", [])
    operations: List[StableOperation] = []
    for position in _top_positions(final_checkpoint, top_k):
        operations.append(
            StableOperation(
                op_id=f"pos_{position}",
                layer=0,
                component="self_attention_route",
                score=float(final_attention[position]),
                metadata={
                    "position": str(position),
                    "checkpoint_step": str(final_checkpoint.get("step", 0)),
                },
            )
        )

    return RealCircuitDiscoveryResult(
        family_id=family_id,
        discovered_operations=operations,
        stable_overlap_score=overlap,
        claim_coverage="low-to-medium",
        failure_modes=[
            "stable position overlap does not imply full graph recovery",
            "attention-route stability can persist even when held-out generalization is weak",
            "targeted-vs-random ablation remains a narrow diagnostic rather than a causal proof",
        ],
        notes=[
            "Stable operations are the highest-attended sequence positions across late checkpoints.",
            "This is a narrow small-transformer discovery contract, not full ACDC/SAE graph recovery.",
        ],
    )
