import unittest

import numpy as np

from seismic_motion.pga.alignment_null import (
    alignment_null_test,
    benjamini_hochberg,
    phase_randomized_surrogate,
)


class AlignmentNullTests(unittest.TestCase):
    def test_null_repeats_complete_search_and_reports_empirical_p(self) -> None:
        visual_t = np.arange(100, dtype=float) * 0.02
        sensor_t = np.arange(300, dtype=float) * 0.02
        visual = np.sin(2 * np.pi * 1.2 * visual_t)[:, None]
        sensor = np.zeros((sensor_t.size, 2))
        sensor[50:150, 1] = -visual[:, 0]
        summary, rows, candidate = alignment_null_test(
            "x",
            visual_t,
            {"acceleration": visual},
            sensor_t,
            {"acceleration": sensor},
            offset_range_s=(0, 4),
            iterations=10,
            seed=3,
        )
        self.assertEqual(len(rows), 10)
        self.assertGreaterEqual(summary.empirical_p_value, 1 / 11)
        self.assertEqual(candidate["sensor_channel"], 1)
        self.assertEqual(candidate["polarity"], -1)

    def test_phase_randomization_preserves_channel_spectrum(self) -> None:
        rng = np.random.default_rng(4)
        values = rng.normal(size=(101, 2))
        surrogate = phase_randomized_surrogate(values, rng)
        np.testing.assert_allclose(
            np.abs(np.fft.rfft(values, axis=0)),
            np.abs(np.fft.rfft(surrogate, axis=0)),
            atol=1e-10,
        )

    def test_bh_is_monotone_in_rank(self) -> None:
        p = np.asarray([0.01, 0.04, 0.03, 0.2])
        q = benjamini_hochberg(p)
        order = np.argsort(p)
        self.assertTrue(np.all(np.diff(q[order]) >= -1e-12))


if __name__ == "__main__":
    unittest.main()
