import sys
from pathlib import Path
import unittest

import numpy as np


BENCHMARK = Path(__file__).resolve().parents[2] / "benchmarks" / "synthetic"
sys.path.insert(0, str(BENCHMARK))
from generator import generate_sequence  # noqa: E402


class SyntheticGeneratorTests(unittest.TestCase):
    def test_subpixel_translation_ground_truth(self) -> None:
        sequence = generate_sequence(
            "translation",
            fps=30,
            duration_s=2,
            translation_amplitude_px=0.1,
            translation_frequency_hz=2,
            image_size=(96, 128),
            point_grid=(3, 4),
        )
        displacement = sequence.tracks_xy_px[:, :, 0] - sequence.reference_xy_px[None, :, 0]
        basis = np.sin(2 * np.pi * 2 * sequence.timestamps)
        estimated_amplitude = np.linalg.lstsq(
            basis[:, None], displacement[:, 0], rcond=None
        )[0][0]
        self.assertAlmostEqual(float(estimated_amplitude), 0.1, places=4)
        self.assertEqual(sequence.frames_rgb.dtype, np.uint8)

    def test_degraded_case_marks_occluded_points(self) -> None:
        sequence = generate_sequence(
            "degraded", fps=25, duration_s=1, image_size=(96, 128), point_grid=(4, 5)
        )
        self.assertFalse(sequence.visibility.all())
        self.assertTrue(np.any(np.linalg.norm(sequence.local_residual_px, axis=-1) > 0))


if __name__ == "__main__":
    unittest.main()
