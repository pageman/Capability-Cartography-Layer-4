import math
import unittest

import numpy as np

from capability_cartography_layer4.circuit_discovery import CircuitDiscovery


class LinearPeriodicModel:
    def __init__(self, weights):
        self.weights = np.asarray(weights, dtype=float)


class CircuitDiscoveryTests(unittest.TestCase):
    def test_periodic_feature_bundle_discovers_fourier_like_components(self) -> None:
        modulus = 17
        features = []
        scores = []
        labels = []
        for x in range(modulus):
            angle = 2.0 * math.pi * x / modulus
            row = [
                math.sin(angle),
                math.cos(angle),
                math.sin(2.0 * angle),
                math.cos(2.0 * angle),
            ]
            score = 1.15 * row[0] + 0.9 * row[3] - 0.2
            label = 1 if score >= 0.0 else 0
            features.append(row)
            scores.append(score)
            labels.append(label)

        model = LinearPeriodicModel([1.15, 0.0, 0.0, 0.9])
        discovery = CircuitDiscovery(model)
        circuit = discovery.identify_circuit(
            "modular_exponentiation",
            {"features": features, "scores": scores, "labels": labels},
        )

        self.assertEqual(circuit.type.value, "fourier_based")
        self.assertGreater(circuit.fourier_signature or 0.0, 0.35)
        self.assertIn("feature_0", circuit.components)
        self.assertIn("feature_3", circuit.components)

        ablation_drop = discovery.compute_ablation_impact(
            circuit, model, {"features": features, "scores": scores, "labels": labels}
        )
        self.assertGreater(ablation_drop, 0.2)


if __name__ == "__main__":
    unittest.main()
