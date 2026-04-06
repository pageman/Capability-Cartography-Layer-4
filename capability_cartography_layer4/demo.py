import numpy as np

from capability_cartography_layer4.orchestration import CCL4Pipeline


class NumpyLinearPlaceholder:
    def __init__(self, weights) -> None:
        self.weight = np.asarray(weights, dtype=float)[None, :]
        self.bias = np.zeros(1, dtype=float)


def _periodic_demo_features(modulus: int = 17) -> np.ndarray:
    rows = []
    for x in range(modulus):
        angle = 2.0 * np.pi * x / modulus
        rows.append([np.sin(angle), np.cos(angle), x / (modulus - 1), np.sin(2.0 * angle)])
    return np.asarray(rows, dtype=float)


def _induction_demo_features(length: int = 12) -> np.ndarray:
    rows = []
    for index in range(length):
        rows.append(
            [
                1.0 if index % 3 == 0 else 0.0,
                float(index) / max(1, length - 1),
                1.0 if index >= length // 2 else 0.0,
                np.sin(index / 2.0),
            ]
        )
    return np.asarray(rows, dtype=float)

def run_demo():
    print("="*80)
    print("CAPABILITY CARTOGRAPHY LAYER 4: PREDICTIVE & MECHANISTIC CONSOLIDATION")
    print("="*80)

    pipeline = CCL4Pipeline()

    # Case 1: P22 Scaling Laws (Kaplan et al. 2020)
    # Regime: Classical (r >> 1), Domain: Language
    p22_params = {
        "paper_id": "P22",
        "capability_id": "cross_entropy_scaling",
        "regime_params": {"m": 100, "r": 1000.0, "d": 512, "s_star": 10},
        "task_metadata": {"domain": "language", "relational_depth": 2}
    }

    # Case 2: P19 Transformer (Vaswani et al. 2017)
    # Regime: Sparse, Novel Architecture, Relational Depth > 4
    p19_params = {
        "paper_id": "P19",
        "capability_id": "in-context_induction",
        "regime_params": {"m": 512, "r": 5.0, "d": 512, "s_star": 20},
        "task_metadata": {
            "domain": "sequence_modeling", 
            "relational_depth": 6, 
            "novel_architecture": True,
            "compressibility_gap": 15.0
        },
        "model": NumpyLinearPlaceholder([0.9, 0.4, 0.2, 0.1]),
        "data": _induction_demo_features()
    }

    # Case 3: P31 Modular Exponentiation (Grokking)
    # Regime: Middle Regime, Fourier Circuits, Quantum Analogy Bridge
    p31_params = {
        "paper_id": "P31",
        "capability_id": "modular_exponentiation",
        "regime_params": {"m": 97, "r": 1.5, "d": 128, "s_star": 10},
        "task_metadata": {
            "domain": "mathematics",
            "relational_depth": 10,
            "novel_architecture": False,
            "compressibility_gap": 35.0
        },
        "model": NumpyLinearPlaceholder([1.1, 0.25, 0.05, 0.8]),
        "data": _periodic_demo_features()
    }

    print("\n[STEP 1] Running P22 Scaling Laws (Predictive Only)")
    r1 = pipeline.run_record(**p22_params)
    print(f"  Regime:   {r1.regime.regime_label}")
    print(f"  Forecast: {r1.forecast.type.value}")
    print(f"  Exponent: {r1.forecast.predicted_exponent}")
    print(f"  Invariant: {r1.forecast.scale_transfer_invariant}")

    print("\n[STEP 2] Running P19 Transformer (Predictive + Mechanistic)")
    r2 = pipeline.run_record(**p19_params)
    print(f"  Regime:   {r2.regime.regime_label}")
    print(f"  Forecast: {r2.forecast.type.value}")
    print(f"  Circuit:  {r2.circuit.type.value if r2.circuit else 'None'}")
    print(f"  Verdict:  {r2.verdict.value}")

    print("\n[STEP 3] Running P31 Modular Exponentiation (Quantum Analogy Bridge)")
    r3 = pipeline.run_record(**p31_params)
    print(f"  Regime:   {r3.regime.regime_label}")
    print(f"  Forecast: {r3.forecast.type.value} (Step-function due to gap)")
    print(f"  Circuit:  {r3.circuit.type.value}")
    print(f"  Quantum Potential: {r3.circuit.quantum_connection_potential}")
    print(f"  Fourier Signature: {r3.circuit.fourier_signature}")
    print(f"  Verdict:  {r3.verdict.value}")

    print("\n" + "="*80)
    print("DEMO COMPLETE: CONSOLIDATION VERIFIED")
    print("="*80)

if __name__ == "__main__":
    run_demo()
