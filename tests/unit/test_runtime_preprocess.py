import unittest

import numpy as np

from seismic_motion.runtime.pipeline import _resize_rgb


class RuntimePreprocessTests(unittest.TestCase):
    def test_target_size_frame_is_not_resampled_again(self) -> None:
        frame = np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3)
        resized = _resize_rgb(frame, (12, 16))
        np.testing.assert_array_equal(resized, frame)


if __name__ == "__main__":
    unittest.main()
