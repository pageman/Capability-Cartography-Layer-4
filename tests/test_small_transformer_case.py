import json
import tempfile
import unittest
from pathlib import Path

from capability_cartography_layer4.small_transformer_case import run_small_transformer_case


class SmallTransformerCaseTests(unittest.TestCase):
    def test_small_transformer_case_generates_artifacts_and_signal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_small_transformer_case(Path(temp_dir))

            checkpoint_path = Path(result.checkpoint_metrics_path or "")
            discovery_path = Path(result.mechanism_discovery_path or "")
            causal_path = Path(result.causal_validation_path or "")
            manifest_path = Path(result.deliverable_manifest_path or "")

            for path in [checkpoint_path, discovery_path, causal_path, manifest_path, checkpoint_path.parent / "summary_report.md"]:
                self.assertTrue(path.exists(), str(path))

            metrics = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            discovery = json.loads(discovery_path.read_text(encoding="utf-8"))
            causal = json.loads(causal_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(result.forecast["forecast_type"], "family_bundle")
            self.assertEqual(len(metrics["families"]), 2)
            self.assertEqual(metrics["claim_coverage"], "low-to-medium")
            self.assertEqual(len(discovery["families"]), 2)
            self.assertEqual(len(causal["families"]), 2)
            self.assertEqual(len(manifest["families"]), 2)
            self.assertIn("family_evidence_overview", metrics)
            self.assertIn("evidence_rubric", metrics["families"][0])
            self.assertIn("feature_bundle_discovery", metrics["families"][0])
            self.assertLess(
                metrics["families"][0]["checkpoints"][-1]["score_rmse"],
                metrics["families"][0]["checkpoints"][0]["score_rmse"],
            )
            self.assertGreaterEqual(
                discovery["families"][0]["attention_route_discovery"]["stable_overlap_score"],
                0.5,
            )
            self.assertIn("components", discovery["families"][0]["feature_bundle_discovery"])
            self.assertIn(
                metrics["families"][0]["evidence_rubric"]["overall_evidence_grade"],
                {"strong", "moderate", "weak"},
            )


if __name__ == "__main__":
    unittest.main()
