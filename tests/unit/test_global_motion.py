import unittest

import numpy as np

from seismic_motion.motion.global_motion import apply_transform, fit_global_transform
from seismic_motion.motion.quality import MotionQuality, QualityThresholds, assess_motion_quality
from seismic_motion.motion.residual_motion import decompose_tracks


class GlobalMotionTests(unittest.TestCase):
    def setUp(self) -> None:
        xx, yy = np.meshgrid(np.linspace(10, 190, 8), np.linspace(10, 90, 5))
        self.points = np.stack([xx.ravel(), yy.ravel()], axis=-1)

    def test_similarity_ransac_rejects_local_motion(self) -> None:
        angle = 0.02
        scale = 1.003
        matrix = np.asarray(
            [
                [scale * np.cos(angle), -scale * np.sin(angle), 0.4],
                [scale * np.sin(angle), scale * np.cos(angle), -0.25],
                [0, 0, 1],
            ]
        )
        observed = apply_transform(matrix, self.points)
        observed[-8:] += [4.0, -3.0]
        estimate = fit_global_transform(
            self.points,
            observed,
            model="similarity",
            ransac_threshold_px=0.2,
            frame_size=(100, 200),
        )
        np.testing.assert_allclose(estimate.matrix, matrix, atol=1e-8)
        self.assertEqual(estimate.num_inliers, 32)
        self.assertAlmostEqual(estimate.inlier_ratio, 0.8)

    def test_affine_fit(self) -> None:
        matrix = np.asarray([[1.01, 0.02, 2], [-0.01, 0.99, -1], [0, 0, 1]])
        observed = apply_transform(matrix, self.points)
        estimate = fit_global_transform(self.points, observed, model="affine", use_ransac=False)
        np.testing.assert_allclose(estimate.matrix, matrix, atol=1e-10)

    def test_quality_rejects_too_few_points(self) -> None:
        estimate = fit_global_transform(
            self.points[:4], self.points[:4] + 1, model="translation", use_ransac=False
        )
        decision = assess_motion_quality(
            estimate, QualityThresholds(min_valid_tracks=10, min_spatial_coverage=0)
        )
        self.assertEqual(decision.quality, MotionQuality.INVALID)
        self.assertIn("insufficient_tracks", decision.reasons)

    def test_residual_decomposition_recovers_local_delta(self) -> None:
        frames = 10
        tracks = np.repeat(self.points[None], frames, axis=0)
        visibility = np.ones(tracks.shape[:2], dtype=bool)
        expected = np.zeros_like(tracks)
        for index in range(frames):
            common = np.asarray([0.2 * index, -0.1 * index])
            tracks[index] += common
            expected[index, -8:] = [2 * np.sin(index), np.cos(index)]
            tracks[index] += expected[index]
        result = decompose_tracks(
            self.points,
            tracks,
            visibility,
            model="translation",
            ransac_threshold_px=0.3,
            frame_size=(100, 200),
            quality_thresholds=QualityThresholds(min_spatial_coverage=0.05),
        )
        np.testing.assert_allclose(result.residual_xy_px[:, -8:], expected[:, -8:], atol=1e-10)
        self.assertTrue(np.all(result.quality == "GOOD"))


if __name__ == "__main__":
    unittest.main()

