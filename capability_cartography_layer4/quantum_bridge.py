try:
    import torch
    import torch.nn as nn
except ImportError:
    from .schemas import MockTorch
    torch = MockTorch()
    class MockNN:
        class Module: pass
    nn = MockNN()

from typing import Dict, List, Optional
from .schemas import MechanismCircuit, MechanismMoment, MechanismOperation, QuantumAnalogyAnnotation

class QuantumAnalogyBridge:
    """
    Cirq-inspired bridge for interpretive quantum analogies.
    Treats transformer analysis stages as moments and discovered components as operations.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

    def build_mechanism_circuit(self, capability_id: str, components: List[str]) -> MechanismCircuit:
        """
        Builds a lightweight staged IR and attaches interpretive quantum analogies.
        These analogies are heuristics for inspection, not verified semantic equivalences.
        """
        moments: List[MechanismMoment] = []
        analogies: List[QuantumAnalogyAnnotation] = []

        for i, comp in enumerate(components):
            operation = MechanismOperation(
                operation_id=f"op_{i}",
                component=comp,
                role=self._infer_role(comp),
                outputs=[f"signal_{i}"],
                metadata={"capability_id": capability_id},
            )
            moments.append(
                MechanismMoment(
                    stage=i,
                    operations=[operation],
                    label=self._infer_stage_label(comp),
                )
            )

            if "attn" in comp:
                analogies.append(QuantumAnalogyAnnotation(
                    analogy_name="conditional_rotation_analogy",
                    register_ids=[i, i + 1],
                    classical_component=comp,
                    confidence=0.68,
                    rationale="Attention routing resembles a conditional phase-style routing motif."
                ))
            elif "mlp" in comp:
                analogies.append(QuantumAnalogyAnnotation(
                    analogy_name="phase_accumulation_analogy",
                    register_ids=[i],
                    classical_component=comp,
                    confidence=0.57,
                    rationale="MLP mixing can be read as a phase-like accumulation motif in feature space."
                ))

        if "modular" in capability_id:
            analogies.append(QuantumAnalogyAnnotation(
                analogy_name="qft_core_analogy",
                register_ids=list(range(len(components))),
                classical_component="unified_circuit",
                confidence=0.79,
                rationale="Periodic structure in modular tasks suggests a Fourier-style computational motif."
            ))

        mechanism_circuit = MechanismCircuit(moments=moments)
        mechanism_circuit.interpretive_quantum_analogies = self.optimize_analogies(analogies)
        mechanism_circuit.analogy_summary = self.summarize_analogies(mechanism_circuit.interpretive_quantum_analogies)
        return mechanism_circuit

    def optimize_analogies(
        self, analogies: List[QuantumAnalogyAnnotation]
    ) -> List[QuantumAnalogyAnnotation]:
        """
        Deduplicates overlapping interpretive annotations.
        This is not a semantic circuit optimization pass.
        """
        optimized: List[QuantumAnalogyAnnotation] = []
        seen_keys = set()
        for analogy in analogies:
            key = (
                analogy.analogy_name,
                tuple(sorted(analogy.register_ids)),
                analogy.classical_component,
            )
            if key not in seen_keys:
                optimized.append(analogy)
                seen_keys.add(key)
        return optimized

    def summarize_analogies(self, analogies: List[QuantumAnalogyAnnotation]) -> Optional[str]:
        """Produces a concise summary of the current heuristic analogy set."""
        if not analogies:
            return None
        names = ", ".join(analogy.analogy_name for analogy in analogies)
        return f"Interpretive quantum analogies detected: {names}."

    def _infer_role(self, component: str) -> str:
        if "attn" in component or "head" in component:
            return "routing"
        if "mlp" in component:
            return "feature_mixing"
        return "unknown"

    def _infer_stage_label(self, component: str) -> str:
        if "attn" in component or "head" in component:
            return "attention"
        if "mlp" in component:
            return "mlp"
        return "mechanism"


# Backward-compatible alias while the repo transitions to the more accurate name.
QuantumBridge = QuantumAnalogyBridge
