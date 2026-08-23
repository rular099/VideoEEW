import unittest

import numpy as np

from seismic_motion.calibration.scale import CalibrationState, ScaleCalibration


class ScaleTests(unittest.TestCase):
    def test_known_length_conversion(self) -> None:
        scale = ScaleCalibration.from_known_length(
            scale_id="ruler-1", known_length_mm=100, known_length_px=250
        )
        self.assertEqual(scale.state, CalibrationState.VALID)
        np.testing.assert_allclose(scale.convert_displacement([1, -2]), [0.4, -0.8])

    def test_uncalibrated_conversion_is_rejected(self) -> None:
        scale = ScaleCalibration.uncalibrated()
        self.assertEqual(scale.state, CalibrationState.UNCALIBRATED)
        with self.assertRaises(RuntimeError):
            scale.convert_displacement([1, 2])


if __name__ == "__main__":
    unittest.main()

