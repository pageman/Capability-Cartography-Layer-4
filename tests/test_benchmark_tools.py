import tempfile
import unittest
from pathlib import Path

from benchmark.baselines import run_baseline_benchmark
from benchmark.metric_ablation import run_metric_ablation


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


if __name__ == "__main__":
    unittest.main()
