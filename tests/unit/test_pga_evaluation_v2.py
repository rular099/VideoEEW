import csv
from pathlib import Path
import tempfile
import unittest

import yaml

from seismic_motion.pga.evaluation import evaluate_subsets, video_quality_decision


class PGAEvaluationV2Tests(unittest.TestCase):
    def _row(self, index: int, correlation: float = 0.6) -> dict[str, str]:
        return {
            "record_id": str(index),
            "pga_horizontal_vector_gal": str(50 + 10 * index),
            "common_peak_acceleration_px_s2": str(1 + index),
            "common_rms_acceleration_px_s2": str(0.5 + index),
            "common_dominant_frequency_hz": "2",
            "common_energy_0p3_1_hz": "0.1",
            "common_energy_1_3_hz": "0.2",
            "common_energy_3_8_hz": "0.1",
            "residual_rms_px": "0.1",
            "residual_peak_px": "0.2",
            "mean_visibility": "0.9",
            "mean_inlier_ratio": "0.9",
            "mean_fit_rmse_px": "0.2",
            "mean_active_track_count": "20",
            "estimated_missing_frames": "0",
            "timestamp_rms_jitter_s": "0.001",
            "quality_flags": "GOOD",
            "alignment_correlation": str(correlation),
            "effective_fps": "30",
            "window_start": "0",
            "window_end": "10",
            "scale_state": "UNCALIBRATED",
            "causal": "1",
        }

    def test_video_quality_does_not_consult_truth_or_alignment(self) -> None:
        config = yaml.safe_load(
            (Path(__file__).resolve().parents[2] / "configs/pga_eval_v2.yaml").read_text()
        )
        first = self._row(0, correlation=0.0)
        second = dict(first, pga_horizontal_vector_gal="9999", alignment_correlation="1.0")
        self.assertEqual(
            video_quality_decision(first, config["video_quality"]),
            video_quality_decision(second, config["video_quality"]),
        )

    def test_three_subsets_and_bootstrap_are_written(self) -> None:
        config = yaml.safe_load(
            (Path(__file__).resolve().parents[2] / "configs/pga_eval_v2.yaml").read_text()
        )
        config["bootstrap"]["iterations"] = 20
        rows = [self._row(index, correlation=0.2 if index == 0 else 0.6) for index in range(6)]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            result = evaluate_subsets(rows, config, output)
            self.assertEqual(result["causal_pga_status"], "PASS")
            for name in (
                "pga_eval_all.csv",
                "pga_eval_video_quality.csv",
                "pga_eval_posthoc_aligned.csv",
                "pga_metrics.json",
                "pga_bootstrap_ci.json",
                "pga_model_research.json",
            ):
                self.assertTrue((output / name).is_file(), name)
            with (output / "pga_eval_posthoc_aligned.csv").open() as handle:
                table = list(csv.DictReader(handle))
            self.assertEqual(table[0]["included"], "False")
            self.assertIn("RESEARCH_DIAGNOSTIC_ONLY", table[1]["interpretation"])


if __name__ == "__main__":
    unittest.main()
