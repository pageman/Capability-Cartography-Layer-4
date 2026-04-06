import json
import tempfile
import unittest
from pathlib import Path

from capability_cartography_layer4.case_study import run_minimal_case_study


class MinimalCaseStudyTests(unittest.TestCase):
    def test_case_study_generates_expected_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_minimal_case_study(Path(temp_dir))

            self.assertEqual(result["forecast_registration"]["forecast_type"], "step-function")
            self.assertEqual(result["checkpoint_metrics"]["observed_curve_class"], "step-function")
            self.assertTrue(result["checkpoint_metrics"]["masking_analysis"]["masking_supported"])
            self.assertGreater(result["causal_validation"]["accuracy_drop_from_ablation"], 0.25)
            self.assertGreater(result["causal_validation"]["accuracy_recovery_after_restoration"], 0.2)

            checkpoints = result["checkpoint_metrics"]["checkpoints"]
            self.assertLess(checkpoints[-1]["fine_rmse"], checkpoints[0]["fine_rmse"])
            self.assertGreater(checkpoints[-1]["fourier_signature"], checkpoints[0]["fourier_signature"])
            self.assertGreater(checkpoints[-1]["heldout_coarse_accuracy"], checkpoints[0]["heldout_coarse_accuracy"])

    def test_case_study_writes_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            run_minimal_case_study(output_dir)

            expected_files = [
                output_dir / "forecast_registration.json",
                output_dir / "checkpoint_metrics.json",
                output_dir / "causal_validation.json",
                output_dir / "summary_report.md",
            ]
            for path in expected_files:
                self.assertTrue(path.exists(), str(path))

            metrics = json.loads((output_dir / "checkpoint_metrics.json").read_text(encoding="utf-8"))
            self.assertIn("masking_analysis", metrics)
            self.assertEqual(len(metrics["checkpoints"]), 13)


if __name__ == "__main__":
    unittest.main()
