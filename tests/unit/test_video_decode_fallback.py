from pathlib import Path
import tempfile
import unittest

import numpy as np

from seismic_motion.runtime.pipeline import _iter_video_rgb, _video_fps


class VideoDecodeFallbackTests(unittest.TestCase):
    def test_incremental_decoder_preserves_rgb_and_requested_size(self) -> None:
        try:
            import cv2
        except ImportError:
            self.skipTest("OpenCV is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.avi"
            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (32, 24)
            )
            self.assertTrue(writer.isOpened())
            for index in range(3):
                frame_bgr = np.zeros((24, 32, 3), dtype=np.uint8)
                frame_bgr[:, :, 2] = 40 + 20 * index
                writer.write(frame_bgr)
            writer.release()

            fps, source, metadata = _video_fps(path, None)
            frames = list(_iter_video_rgb(path, output_size=(12, 16)))
            self.assertAlmostEqual(fps, 10.0, places=1)
            self.assertIn("container_constant_frame_rate", source)
            self.assertEqual(tuple(metadata["size"]), (32, 24))
            self.assertEqual(len(frames), 3)
            self.assertEqual(frames[0].shape, (12, 16, 3))
            self.assertGreater(float(np.mean(frames[0][:, :, 0])), 30.0)
            self.assertLess(float(np.mean(frames[0][:, :, 2])), 5.0)


if __name__ == "__main__":
    unittest.main()
