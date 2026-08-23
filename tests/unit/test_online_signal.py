import unittest

import numpy as np

from seismic_motion.pga.online import RunningPGAEstimator
from seismic_motion.signal.derivatives import (
    CausalDerivativeEstimator,
    backward_finite_difference,
)
from seismic_motion.signal.online import OnlineSignalProcessor
from benchmarks.signal.evaluate_causal import evaluate


class CausalDerivativeTests(unittest.TestCase):
    def test_backward_second_derivative_is_causal_on_irregular_quadratic(self) -> None:
        timestamps = np.asarray([0.0, 0.1, 0.23, 0.4, 0.7])
        values = timestamps**2
        result = backward_finite_difference(timestamps, values, derivative_order=2)
        np.testing.assert_allclose(result[2:], 2.0, atol=1e-12)
        self.assertTrue(np.isnan(result[:2]).all())

    def test_causal_polynomial_has_bounded_history_and_no_future_dependence(self) -> None:
        timestamps = np.arange(30, dtype=float) / 30.0
        values = np.column_stack([timestamps**2, 3 * timestamps**2])
        first = CausalDerivativeEstimator(window_length=7, polynomial_order=3)
        second = CausalDerivativeEstimator(window_length=7, polynomial_order=3)
        prefix_a = [first.update(t, x).acceleration.copy() for t, x in zip(timestamps, values)]
        changed = values.copy()
        changed[20:] += 1000
        prefix_b = [second.update(t, x).acceleration.copy() for t, x in zip(timestamps, changed)]
        np.testing.assert_allclose(prefix_a[:20], prefix_b[:20], equal_nan=True)
        self.assertEqual(first.retained_samples, 7)
        np.testing.assert_allclose(prefix_a[-1], [2, 6], atol=1e-9)


class OnlineSignalTests(unittest.TestCase):
    def test_processor_state_is_bounded(self) -> None:
        processor = OnlineSignalProcessor(
            sample_rate_hz=30,
            derivative_method="causal_polynomial",
            window_length=9,
            polynomial_order=3,
        )
        state = None
        for index in range(300):
            timestamp = index / 30
            common = np.asarray([np.sin(timestamp), np.cos(timestamp)])
            local = np.tile(common, (4, 1)) * 0.1
            state = processor.update(timestamp, common, local, "GOOD", reseed_id=index // 100)
        self.assertIsNotNone(state)
        self.assertEqual(state.retained_history_samples, 9)
        self.assertGreater(state.filter_state_bytes, 0)
        self.assertEqual(state.samples_seen, 300)

    def test_uncalibrated_running_pga_is_explicitly_research_only(self) -> None:
        proxy_only = RunningPGAEstimator(coefficient_gal_per_px_s2=2.0)
        state = proxy_only.update(0, 3.0, quality="GOOD", scale_valid=False)
        self.assertIsNone(state.pga_running_est_gal)
        self.assertEqual(state.interpretation, "PIXEL_PROXY_ONLY")
        research = RunningPGAEstimator(
            coefficient_gal_per_px_s2=2.0, allow_uncalibrated_research=True
        )
        first = research.update(0, 3.0, quality="GOOD", scale_valid=False)
        second = research.update(1, 2.0, quality="GOOD", scale_valid=False)
        self.assertEqual(first.pga_running_est_gal, 6.0)
        self.assertEqual(second.pga_running_est_gal, 6.0)
        self.assertFalse(second.deployment_prediction_allowed)
        self.assertEqual(second.interpretation, "RESEARCH_ONLY_UNCALIBRATED")

    def test_offline_causal_benchmark_reports_required_errors(self) -> None:
        rows, _ = evaluate(fps=30, duration_s=8)
        self.assertEqual(len(rows), 4)
        for row in rows:
            for field in (
                "amplitude_bias_fraction",
                "rmse",
                "phase_lag_s",
                "peak_timing_error_s",
                "peak_amplitude_error",
            ):
                self.assertTrue(np.isfinite(float(row[field])), field)


if __name__ == "__main__":
    unittest.main()
