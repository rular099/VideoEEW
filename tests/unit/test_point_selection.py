import unittest

import numpy as np

from seismic_motion.tracking.point_selection import (
    select_distributed_corners,
    spatial_coverage,
    validate_manual_points,
)


class PointSelectionTests(unittest.TestCase):
    def test_manual_points_are_bounds_checked(self) -> None:
        points = validate_manual_points([[0, 0], [9, 7]], width=10, height=8)
        self.assertEqual(points.shape, (2, 2))
        with self.assertRaises(ValueError):
            validate_manual_points([[10, 0]], width=10, height=8)

    def test_corner_selection_spans_texture(self) -> None:
        image = np.zeros((120, 160, 3), dtype=np.uint8)
        for y in range(10, 111, 20):
            for x in range(10, 151, 20):
                image[y - 3 : y + 4, x - 3 : x + 4] = 255
        points = select_distributed_corners(image, 16, min_distance=6)
        self.assertEqual(points.shape, (16, 2))
        self.assertGreater(spatial_coverage(points, 160, 120), 0.25)


if __name__ == "__main__":
    unittest.main()

