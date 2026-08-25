from pathlib import Path
import tempfile
import time
import unittest

import numpy as np

from seismic_motion.config import load_config
from seismic_motion.runtime.realtime import ProcessedBlock, RealtimeRunner
from seismic_motion.tracking.types import TrackBatch


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

    def test_writer_emits_causal_signal_and_running_proxy(self) -> None:
        repository = Path(__file__).resolve().parents[2]
        config = load_config(repository / "configs/causal_realtime.yaml")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint.pth"
            checkpoint.write_bytes(b"test-checkpoint")
            runner = RealtimeRunner(
                0,
                config,
                cotracker_root=root / "missing-cotracker",
                checkpoint=checkpoint,
                device="cpu",
                output_directory=root / "run",
            )
            x, y = np.meshgrid(np.linspace(20, 490, 4), np.linspace(20, 360, 3))
            query = np.column_stack([x.ravel(), y.ravel()]).astype(np.float32)
            timestamps = np.arange(8, dtype=float) / 30
            tracks = np.stack(
                [query + np.asarray([0.2 * index, 0.1 * index]) for index in range(8)]
            )
            batch = TrackBatch(
                timestamps=timestamps,
                frame_indices=np.arange(8),
                point_ids=np.arange(query.shape[0]),
                xy_px=tracks,
                visible=np.ones((8, query.shape[0]), dtype=bool),
                confidence=np.ones((8, query.shape[0]), dtype=np.float32),
                query_xy_px=query,
            )
            now = time.monotonic()
            runner.output_queue.put(
                ProcessedBlock(
                    batch=batch,
                    capture_timestamp=now - 0.3,
                    enqueue_timestamp=now - 0.2,
                    window_ready_timestamp=now - 0.2,
                    tracker_start=now - 0.1,
                    tracker_end=now,
                    source_path="synthetic",
                    playlist_index=0,
                    loop_index=0,
                    source_boundary=True,
                )
            )
            runner.output_queue.put(None)
            runner._writer_worker()
            output = root / "run"
            self.assertEqual(len((output / "online_signal.csv").read_text().splitlines()), 9)
            running = (output / "running_pga.csv").read_text(encoding="utf-8")
            self.assertIn("PIXEL_PROXY_ONLY", running)
            self.assertEqual(len(runner.timing_records), 1)
            self.assertEqual(runner.frames_written, 8)
            contract = runner._causality_contract()
            self.assertEqual(contract["signal_pga_causality"], "PASS")
            self.assertEqual(
                contract["tracker_source_timestamp_causality"],
                "FAIL_FUTURE_CONTEXT",
            )
            self.assertEqual(contract["end_to_end_source_timestamp_causality"], "FAIL")


if __name__ == "__main__":
    unittest.main()
