import json
from pathlib import Path
import unittest


class FrozenSyntheticMetricTests(unittest.TestCase):
    def test_frozen_metrics_remain_below_acceptance_thresholds(self) -> None:
        repository = Path(__file__).resolve().parents[2]
        metrics = json.loads(
            (repository / "runs/20260823-synthetic-motion/metrics.json").read_text(
                encoding="utf-8"
            )
        )
        oracle = metrics["oracle_matrix"]
        tracker = metrics["cotracker_translation"]
        combined = metrics["cotracker_translation_rotation_local"]
        self.assertEqual(oracle["case_count"], 116)
        self.assertLess(oracle["max_common_point_rmse_px"], 1e-4)
        self.assertLess(tracker["tracking_rmse_px"], 0.15)
        self.assertLess(combined["tracking_rmse_px"], 0.6)
        self.assertLess(combined["local_residual_rmse_px"], 0.4)
        self.assertLess(tracker["warm_p95_block_ms"], 266.7)
        self.assertLess(combined["warm_p95_block_ms"], 266.7)


if __name__ == "__main__":
    unittest.main()
