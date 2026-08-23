from pathlib import Path
import tempfile
import unittest

from seismic_motion.runtime.metrics import sample_memory, write_memory_csv


class RuntimeMetricTests(unittest.TestCase):
    def test_memory_sample_is_finite_and_serializable(self) -> None:
        sample = sample_memory(1.25, 3)
        self.assertGreater(sample.rss_mb, 0)
        self.assertGreater(sample.peak_rss_mb, 0)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "memory.csv"
            write_memory_csv(output, [sample])
            text = output.read_text(encoding="utf-8")
            self.assertIn("peak_rss_mb", text)
            self.assertIn(",3,", text)


if __name__ == "__main__":
    unittest.main()
