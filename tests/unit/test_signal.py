import unittest

import numpy as np

from seismic_motion.signal.derivatives import finite_difference, local_polynomial_derivative
from seismic_motion.signal.features import extract_motion_features
from seismic_motion.signal.filtering import bandpass_filter
from seismic_motion.signal.timestamps import diagnose_timebase, resample_uniform


class SignalTests(unittest.TestCase):
    def test_irregular_local_polynomial_derivative(self) -> None:
        rng = np.random.default_rng(3)
        dt = 1 / 30 + rng.normal(0, 0.0004, 180)
        times = np.cumsum(dt)
        values = np.sin(2 * np.pi * 2 * times)
        expected = 2 * np.pi * 2 * np.cos(2 * np.pi * 2 * times)
        estimated = local_polynomial_derivative(times, values, window_length=7)
        rmse = np.sqrt(np.mean(np.square(estimated[5:-5] - expected[5:-5])))
        self.assertLess(rmse, 0.15)

    def test_resampling_reports_missing_interval(self) -> None:
        times = np.arange(30) / 30
        times = np.delete(times, 10)
        values = np.sin(times)
        _, _, diagnostics = resample_uniform(times, values, target_fps=30)
        self.assertEqual(diagnostics.estimated_missing_frames, 1)

    def test_filter_suppresses_out_of_band_signal(self) -> None:
        times = np.arange(300) / 30
        signal = np.sin(2 * np.pi * 2 * times) + np.sin(2 * np.pi * 12 * times)
        filtered = bandpass_filter(signal, 30, (0.3, 8), causal=False).values
        reference = np.sin(2 * np.pi * 2 * times)
        correlation = np.corrcoef(filtered[30:-30], reference[30:-30])[0, 1]
        self.assertGreater(correlation, 0.98)

    def test_feature_dominant_frequency(self) -> None:
        times = np.arange(120) / 30
        common = np.stack([np.sin(2 * np.pi * 2 * times), np.zeros_like(times)], axis=1)
        residual = np.zeros((times.size, 12, 2))
        visible = np.ones((times.size, 12), dtype=bool)
        features = extract_motion_features(times, common, residual, visible)
        self.assertAlmostEqual(features["common_dominant_frequency_hz"], 2.0, places=6)
        self.assertEqual(diagnose_timebase(times).estimated_missing_frames, 0)
        self.assertEqual(finite_difference(times, common).shape, common.shape)


if __name__ == "__main__":
    unittest.main()
