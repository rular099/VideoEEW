import unittest

import numpy as np

from seismic_motion.tracking.online_buffer import (
    AuditedBoundedQueue,
    BufferOverload,
    SlidingFrameBuffer,
)


class SlidingFrameBufferTests(unittest.TestCase):
    def test_emits_fixed_windows_without_growing(self) -> None:
        buffer = SlidingFrameBuffer(window_len=4, step=2)
        emitted = []
        for index in range(10):
            window = buffer.append(
                np.full((3, 5, 3), index, dtype=np.uint8), index * 0.02, index
            )
            if window is not None:
                emitted.append(window.frame_indices.tolist())
            self.assertLessEqual(buffer.size, 4)
        self.assertEqual(emitted, [[0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 7], [6, 7, 8, 9]])

    def test_non_monotonic_timestamp_is_rejected(self) -> None:
        buffer = SlidingFrameBuffer(window_len=4, step=2)
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        buffer.append(frame, 1.0)
        with self.assertRaises(ValueError):
            buffer.append(frame, 1.0)


class AuditedQueueTests(unittest.TestCase):
    def test_full_queue_is_visible_and_counted(self) -> None:
        queue = AuditedBoundedQueue[int](maxsize=2, name="tracker")
        queue.put(1)
        queue.put(2)
        with self.assertRaises(BufferOverload):
            queue.put(3)
        self.assertEqual(queue.metrics()["rejected_items"], 1)
        self.assertEqual(queue.metrics()["max_observed_depth"], 2)

    def test_full_queue_control_retry_is_not_counted_as_data_loss(self) -> None:
        queue = AuditedBoundedQueue[int | None](maxsize=1, name="control-test")
        queue.put(1)
        with self.assertRaises(BufferOverload):
            queue.put(None, count_rejection=False)
        self.assertEqual(queue.metrics()["rejected_items"], 0)
        self.assertEqual(queue.metrics()["max_observed_depth"], 1)


if __name__ == "__main__":
    unittest.main()
