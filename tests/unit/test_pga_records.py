from pathlib import Path
import tempfile
import unittest

import numpy as np

from seismic_motion.pga.records import (
    discover_dataset_pairs,
    load_strong_motion_files,
    load_strong_motion_txt,
)


class StrongMotionRecordTests(unittest.TestCase):
    def test_split_records_are_concatenated_with_monotone_time(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "x-1.txt"
            second = root / "x-2.txt"
            content = "time\tew\tns\tud\n0\t1\t0\t0\n0.01\t2\t0\t0\n"
            first.write_text(content, encoding="utf-8")
            second.write_text(content, encoding="utf-8")
            record = load_strong_motion_files([str(first), str(second)])
            self.assertEqual(record.timestamps_s.size, 4)
            self.assertTrue(np.all(np.diff(record.timestamps_s) > 0))
            self.assertEqual(record.ew_gal.tolist(), [1, 2, 1, 2])

    def test_load_and_pga_definitions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.txt"
            path.write_text(
                "Time(s)\tEW-gal\tNS-gal\tUD-gal\n"
                "0.01\t3\t4\t0\n"
                "0.02\t6\t8\t1\n",
                encoding="utf-8",
            )
            record = load_strong_motion_txt(path)
            self.assertEqual(record.pga_gal("horizontal_vector"), 10.0)
            self.assertEqual(record.pga_gal("max_horizontal_component"), 8.0)
            self.assertAlmostEqual(record.pga_gal("three_component_vector"), np.sqrt(101))

    def test_discovery_groups_split_sensor_segments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "视频").mkdir()
            (root / "强震仪").mkdir()
            (root / "视频" / "54.avi").touch()
            (root / "视频" / "1.avi").touch()
            (root / "强震仪" / "54-1.txt").touch()
            (root / "强震仪" / "54-2.txt").touch()
            pairs = {pair.record_id: pair for pair in discover_dataset_pairs(root)}
            self.assertEqual(pairs["54"].pairing_status, "paired_split_sensor")
            self.assertEqual(len(pairs["54"].strong_motion_paths), 2)
            self.assertEqual(pairs["1"].pairing_status, "missing_strong_motion")


if __name__ == "__main__":
    unittest.main()
