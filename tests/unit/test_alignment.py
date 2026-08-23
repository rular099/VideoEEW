import unittest

import numpy as np

from seismic_motion.pga.alignment import (
    acceleration_to_displacement,
    estimate_time_offset,
    full_containment_offset_range,
)


class AlignmentTests(unittest.TestCase):
    def test_recovers_axis_sign_and_offset(self) -> None:
        visual_times = np.arange(300) / 30
        sensor_times = np.arange(1000) / 100
        visual = np.stack(
            [
                np.sin(2 * np.pi * 1.7 * visual_times),
                np.cos(2 * np.pi * 0.6 * visual_times),
            ],
            axis=1,
        )
        true_offset = 0.18
        sensor = np.stack(
            [
                np.zeros_like(sensor_times),
                -np.sin(2 * np.pi * 1.7 * (sensor_times - true_offset)),
            ],
            axis=1,
        )
        result = estimate_time_offset(
            visual_times,
            visual,
            sensor_times,
            sensor,
            max_offset_s=0.5,
            step_s=0.01,
        )
        self.assertAlmostEqual(result.offset_s, true_offset, places=2)
        self.assertEqual(result.visual_channel, 0)
        self.assertEqual(result.sensor_channel, 1)
        self.assertEqual(result.polarity, -1)
        self.assertEqual(result.status, "ALIGNED")

    def test_full_sensor_start_search_and_displacement(self) -> None:
        visual_times = np.arange(120) / 30
        sensor_times = np.arange(2000) / 100
        self.assertEqual(
            full_containment_offset_range(visual_times, sensor_times),
            (0.0, sensor_times[-1] - visual_times[-1]),
        )
        acceleration = np.sin(2 * np.pi * sensor_times)[:, None]
        displacement = acceleration_to_displacement(sensor_times, acceleration)
        self.assertEqual(displacement.shape, acceleration.shape)
        self.assertTrue(np.isfinite(displacement).all())


if __name__ == "__main__":
    unittest.main()
