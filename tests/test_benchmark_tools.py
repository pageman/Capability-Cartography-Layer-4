import tempfile
import unittest
from pathlib import Path

from benchmark.baselines import run_baseline_benchmark
from benchmark.metric_ablation import run_metric_ablation
from capability_cartography_layer4.real_tiny_case import run_real_tiny_suite


class BenchmarkToolTests(unittest.TestCase):
    def test_baselines_generate_results(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_baseline_benchmark(Path(temp_dir))
            self.assertIn("results", result)
            self.assertIn("majority", result["results"])
            self.assertIn("ccl4_rule_set", result["results"])
            self.assertGreaterEqual(result["results"]["ccl4_rule_set"]["accuracy"], result["results"]["majority"]["accuracy"])

    def test_metric_ablation_generates_expected_controls(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_metric_ablation(Path(temp_dir))
            self.assertEqual(result["metric_curve_classes"]["thresholded_pass_rate"], "step-function")
            self.assertNotEqual(result["metric_curve_classes"]["score_rmse"], "step-function")
            self.assertTrue(result["falsification_controls"]["non_periodic_control_fourier_signal_stays_low"])

    def test_real_tiny_control_case_is_strong_under_control_rubric(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_real_tiny_suite(Path(temp_dir))
            control = result["cases"]["tiny_nonperiodic_linear"]
            self.assertEqual(control["observed_curve_classes"]["score_rmse"], "power-law")
            self.assertEqual(control["evidence_rubric"]["overall_evidence_grade"], "strong")
            self.assertTrue(control["evidence_rubric"]["supports_control_claim"])

    def test_sparse_case_is_clean_falsifier(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_real_tiny_suite(Path(temp_dir))
            sparse = result["cases"]["tiny_sparse_relational"]
            self.assertEqual(sparse["evidence_rubric"]["overall_evidence_grade"], "weak")
            self.assertEqual(sparse["mechanism_discovery"]["circuit_type"], "unknown")
            self.assertTrue(sparse["falsifier_summary"]["forecast_miss"])
            self.assertTrue(sparse["falsifier_summary"]["mechanism_missing"])
            self.assertTrue(sparse["falsifier_summary"]["causal_effect_missing"])


if __name__ == "__main__":
    unittest.main()
