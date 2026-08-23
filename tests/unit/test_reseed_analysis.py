import unittest

import numpy as np

from seismic_motion.tracking.reseed_analysis import analyze_reseed_arrays


class ReseedAnalysisTests(unittest.TestCase):
    def test_detects_injected_reseed_jump(self) -> None:
        timestamps = np.arange(20, dtype=float) / 10
        frame_indices = np.arange(20)
        query = np.column_stack([np.arange(5), np.arange(5)]).astype(float)
        tracks = np.stack([query + [0.1 * index, 0] for index in range(20)])
        tracks[10:] += [2.0, 0]
        common = np.column_stack([0.1 * np.arange(20), np.zeros(20)])
        common[10:, 0] += 2
        rotation = np.zeros(20)
        acceleration = np.zeros((20, 2))
        acceleration[10, 0] = 20
        rows = analyze_reseed_arrays(
            timestamps,
            frame_indices,
            tracks,
            common,
            rotation,
            acceleration,
            np.full(20, "GOOD"),
            [{"frame_index": 10, "reseed_id": 1}],
        )
        self.assertEqual(len(rows), 1)
        self.assertGreater(rows[0]["point_position_jump_mean_px"], 1.9)
        self.assertGreater(rows[0]["common_translation_jump_px"], 1.9)
        self.assertEqual(rows[0]["quality_after"], "GOOD")


if __name__ == "__main__":
    unittest.main()
