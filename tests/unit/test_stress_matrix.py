from pathlib import Path
import unittest

import yaml

from benchmarks.synthetic.run_stress_matrix import coverage_cases


class StressMatrixTests(unittest.TestCase):
    def test_coverage_contains_every_required_axis_and_scene(self) -> None:
        spec = yaml.safe_load(
            (Path(__file__).resolve().parents[2] / "benchmarks/synthetic/spec.yaml").read_text()
        )
        cases = coverage_cases(spec)
        self.assertEqual(
            {case["translation_amplitude_px"] for case in cases if case["scene"] == "translation"},
            {0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0},
        )
        self.assertEqual(
            {case["translation_frequency_hz"] for case in cases if case["scene"] == "translation"},
            {0.5, 1.0, 3.0, 5.0, 8.0},
        )
        self.assertEqual({case["fps"] for case in cases}, {25.0, 30.0, 50.0, 60.0})
        self.assertEqual(
            {case["rotation_amplitude_deg"] for case in cases if case["scene"] == "rotation"},
            {0.02, 0.05, 0.1, 0.5, 1.0},
        )
        self.assertTrue(
            {
                "translation_rotation",
                "translation_local",
                "rotation_local",
                "translation_rotation_local",
                "occlusion",
                "motion_blur",
                "illumination_change",
                "low_texture",
            }.issubset({case["scene"] for case in cases})
        )


if __name__ == "__main__":
    unittest.main()
