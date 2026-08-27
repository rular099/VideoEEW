from pathlib import Path
import tempfile
import unittest

from scripts.assemble_stage_audit import _copy_group
from scripts.build_audit_bundle import build_bundle


class AuditBundleTests(unittest.TestCase):
    def test_composite_csv_copy_normalizes_line_endings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            output.mkdir()
            (source / "metrics.csv").write_bytes(b"name,value\r\na,1\r\n")
            _copy_group(source, ("metrics.csv",), output)
            self.assertEqual(
                (output / "metrics.csv").read_bytes(), b"name,value\na,1\n"
            )

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
            (run / "reseed_summary.json").write_text(
                "NOT_MEASURED\n", encoding="utf-8"
            )
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
            (run / "pga_eval_all.csv").write_text(
                "record_id,included,true_pga_gal,predicted_pga_gal\n"
                "1,True,10,9\n2,False,20,nan\n",
                encoding="utf-8",
            )
            (run / "plots").mkdir()
            (run / "plots" / "plot_manifest.json").write_text(
                '{"runtime_memory.png":"GENERATED"}\n', encoding="utf-8"
            )
            audit = build_bundle(run, root / "audit", make_zip=True)
            self.assertTrue((audit / "AUDIT_SUMMARY.md").is_file())
            self.assertTrue((audit / "timing_summary.csv").is_file())
            self.assertTrue((audit / "memory_summary.csv").is_file())
            self.assertTrue((audit / "plots" / "plot_manifest.json").is_file())
            self.assertTrue((root / "audit" / "sample.zip").is_file())
            summary = (audit / "AUDIT_SUMMARY.md").read_text(encoding="utf-8")
            self.assertIn("abc", summary)
            self.assertIn("INVALID", summary)
            self.assertIn("110.0 MB", summary)
            self.assertIn("PGA rows included: `1`", summary)
            self.assertIn("PC 30 FPS realtime: NOT_TESTED", summary)
            self.assertIn("PASS_NO_DROP_RECORDED", summary)
            self.assertEqual(
                (audit / "RK3588_STATUS.md").read_text(encoding="utf-8"),
                "NOT_MEASURED\n",
            )

    def test_single_benign_reseed_is_not_reported_as_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            run.mkdir()
            (run / "manifest.json").write_text(
                '{"git_commit":"abc","git_dirty":false,"device":"cpu"}\n',
                encoding="utf-8",
            )
            (run / "metrics.json").write_text("{}\n", encoding="utf-8")
            (run / "reseed_summary.json").write_text(
                '{"reseed_events_analyzed":1,"acceleration_spike_ratio_p95":1.2}\n',
                encoding="utf-8",
            )
            audit = build_bundle(run, root / "audit")
            summary = (audit / "AUDIT_SUMMARY.md").read_text(encoding="utf-8")
            self.assertIn("NOT_EVALUABLE_SINGLE_EVENT_BELOW_THRESHOLD", summary)
            self.assertIn("analyzed events `1`", summary)


if __name__ == "__main__":
    unittest.main()
