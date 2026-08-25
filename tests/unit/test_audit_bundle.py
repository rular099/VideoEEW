from pathlib import Path
import tempfile
import unittest

from scripts.build_audit_bundle import build_bundle


class AuditBundleTests(unittest.TestCase):
    def test_builds_summary_and_tables_from_partial_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "runs" / "sample"
            run.mkdir(parents=True)
            (run / "manifest.json").write_text(
                '{"git_commit":"abc","git_dirty":false,"device":"cpu"}\n',
                encoding="utf-8",
            )
            (run / "metrics.json").write_text('{"rmse":0.1}\n', encoding="utf-8")
            (run / "timing.csv").write_text(
                "timestamp,tracker_ms,total_pipeline_ms\n0,10,12\n1,20,23\n",
                encoding="utf-8",
            )
            (run / "queue.csv").write_text(
                "timestamp,capture_queue_depth,tracker_queue_depth,output_queue_depth,dropped_frames,dropped_blocks,overload_state\n"
                "0,0,0,0,0,0,NORMAL\n1,0,1,0,0,0,NORMAL\n",
                encoding="utf-8",
            )
            (run / "motion_quality.csv").write_text(
                "timestamp,quality,reasons\n0,GOOD,\n1,INVALID,low_tracks\n",
                encoding="utf-8",
            )
            (run / "memory.csv").write_text(
                "timestamp,block_index,rss_mb,peak_rss_mb,system_available_mem_mb\n"
                "0,0,100,110,2000\n",
                encoding="utf-8",
            )
            audit = build_bundle(run, root / "audit", make_zip=True)
            self.assertTrue((audit / "AUDIT_SUMMARY.md").is_file())
            self.assertTrue((audit / "timing_summary.csv").is_file())
            self.assertTrue((audit / "memory_summary.csv").is_file())
            self.assertTrue((root / "audit" / "sample.zip").is_file())
            summary = (audit / "AUDIT_SUMMARY.md").read_text(encoding="utf-8")
            self.assertIn("abc", summary)
            self.assertIn("INVALID", summary)
            self.assertIn("110.0 MB", summary)
            self.assertIn("PC 30 FPS realtime: NOT_TESTED", summary)
            self.assertEqual(
                (audit / "RK3588_STATUS.md").read_text(encoding="utf-8"),
                "NOT_MEASURED\n",
            )


if __name__ == "__main__":
    unittest.main()
