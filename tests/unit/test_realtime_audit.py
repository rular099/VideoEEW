from pathlib import Path
import tempfile
import unittest

from seismic_motion.config import load_config
from seismic_motion.runtime.realtime import RealtimeRunner


class RealtimeAuditTests(unittest.TestCase):
    def test_empty_finite_run_still_writes_audit_contract(self) -> None:
        repository = Path(__file__).resolve().parents[2]
        config = load_config(repository / "configs/pc_baseline.yaml")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint.pth"
            checkpoint.write_bytes(b"test-checkpoint")
            output = root / "run"
            runner = RealtimeRunner(
                0,
                config,
                cotracker_root=root / "missing-cotracker",
                checkpoint=checkpoint,
                device="cpu",
                output_directory=output,
            )
            runner._capture_worker = lambda duration_s: None
            runner._tracker_worker = lambda: None
            runner._writer_worker = lambda: None
            summary = runner.run(duration_s=0.0)
            self.assertFalse(summary["stopped_for_overload"])
            for name in (
                "manifest.json",
                "config.yaml",
                "memory.csv",
                "timing.csv",
                "queue.csv",
                "events.jsonl",
                "metrics.json",
            ):
                self.assertTrue((output / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
