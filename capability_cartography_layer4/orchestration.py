try:
    import torch
    import torch.nn as nn
except ImportError:
    from .schemas import MockTorch
    torch = MockTorch()
    class MockNN:
        class Module: pass
    nn = MockNN()

from typing import List, Dict, Optional
from .schemas import CCL4Record, VerdictType
from .regime_forecaster import RegimeForecaster
from .circuit_discovery import CircuitDiscovery
from .quantum_bridge import QuantumAnalogyBridge

class CCL4Pipeline:
    """Orchestrates Predictive Forecasting and Mechanistic Analysis."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.forecaster = RegimeForecaster(config)
        self.discovery = CircuitDiscovery()
        self.bridge = QuantumAnalogyBridge(config)

    def run_record(self, paper_id: str, capability_id: str, 
                   regime_params: Dict, task_metadata: Dict,
                   model: Optional[nn.Module] = None,
                   data: Optional[torch.Tensor] = None) -> CCL4Record:
        """Runs the complete CCL4 process for a single record."""
        
        # 1. Pre-training: Regime Classification and Forecasting
        regime = self.forecaster.classify_regime(
            m=regime_params.get("m", 1),
            r=regime_params.get("r", 1.0),
            d=regime_params.get("d", 1),
            s_star=regime_params.get("s_star", 1)
        )
        
        forecast = self.forecaster.forecast(regime, task_metadata)
        
        # 2. Post-training: Circuit Discovery and Verification
        circuit = None
        verdict = VerdictType.UNDETECTED
        verdict_confidence = 0.0
        
        if model is not None and data is not None:
            circuit = self.discovery.identify_circuit(capability_id, data)

            # Add heuristic, interpretive analogies without changing the core circuit type.
            mechanism_circuit = self.bridge.build_mechanism_circuit(capability_id, circuit.components)
            if circuit.mechanism_circuit is None:
                circuit.mechanism_circuit = mechanism_circuit
            else:
                circuit.mechanism_circuit.interpretive_quantum_analogies = (
                    mechanism_circuit.interpretive_quantum_analogies
                )
                circuit.mechanism_circuit.analogy_summary = mechanism_circuit.analogy_summary

            # Simulated causal verification
            ablation_drop = self.discovery.compute_ablation_impact(circuit, model, data)
            
            # 3. Final Verdict Policy
            if ablation_drop > 0.8:
                verdict = VerdictType.CONFIRMED
                verdict_confidence = ablation_drop
            elif ablation_drop > 0.3:
                verdict = VerdictType.CONDITIONAL
                verdict_confidence = ablation_drop
            else:
                verdict = VerdictType.UNCONFIRMED
                verdict_confidence = ablation_drop

        return CCL4Record(
            capability_id=capability_id,
            paper_id=paper_id,
            regime=regime,
            forecast=forecast,
            circuit=circuit,
            verdict=verdict,
            verdict_confidence=verdict_confidence,
            provenance={
                "forecaster_version": "v1.0-Schur2026",
                "discovery_version": "v1.0-Mechanistic",
                "bridge_version": "v1.1-Quantum-Analogy-Heuristic"
            }
        )

    def run_batch(self, records: List[Dict]) -> List[CCL4Record]:
        """Process a batch of capability records."""
        return [self.run_record(**r) for r in records]
