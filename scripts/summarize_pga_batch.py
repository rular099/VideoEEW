#!/usr/bin/env python3
"""Reduce a multi-record PGA feature run to small reviewable tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil

import numpy as np


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(
    run_root: Path,
    output: Path,
    *,
    batch_peak_rss_kb: int | None = None,
    batch_observed_rss_kb: int | None = None,
    gpu_memory_mib: int | None = None,
    performance_caveat: str = "not a controlled realtime acceptance run unless documented otherwise",
) -> dict[str, object]:
    run_root = run_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    feature_source = run_root / "pga_features.csv"
    if feature_source.is_file():
        shutil.copy2(feature_source, output / "pga_features.csv")
    alignment_rows: list[dict[str, object]] = []
    runtime_rows: list[dict[str, object]] = []
    tracker_times: list[float] = []
    quality_totals: dict[str, int] = {}
    for record_dir in sorted(run_root.glob("record-*")):
        record_id = record_dir.name.removeprefix("record-")
        alignment = _json(record_dir / "alignment_v2.json")
        if alignment:
            preferred_name = str(alignment["preferred"])
            preferred = alignment[preferred_name]
            acceleration = alignment["acceleration"]
            displacement = alignment["displacement"]
            alignment_rows.append(
                {
                    "record_id": record_id,
                    "preferred_domain": preferred_name,
                    "preferred_offset_s": preferred["offset_s"],
                    "preferred_correlation": preferred["correlation"],
                    "preferred_status": preferred["status"],
                    "acceleration_correlation": acceleration["correlation"],
                    "displacement_correlation": displacement["correlation"],
                    "search_min_s": alignment["searched_offset_range_s"][0],
                    "search_max_s": alignment["searched_offset_range_s"][1],
                    "interpretation": alignment.get("interpretation", ""),
                }
            )
        metrics = _json(record_dir / "metrics.json")
        input_manifest = _json(record_dir / "input_manifest.json")
        timing_rows = _csv_rows(record_dir / "timing.csv")
        values = []
        for row in timing_rows:
            try:
                value = float(row["tracker_ms"])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(value):
                values.append(value)
                tracker_times.append(value)
        quality = metrics.get("motion_quality_counts", {})
        for name, count in quality.items():
            quality_totals[str(name)] = quality_totals.get(str(name), 0) + int(count)
        runtime_rows.append(
            {
                "record_id": record_id,
                "frame_count": input_manifest.get("frame_count", ""),
                "fps": input_manifest.get("fps", ""),
                "tracker_blocks": len(values),
                "tracker_mean_ms": float(np.mean(values)) if values else float("nan"),
                "tracker_p95_ms": float(np.percentile(values, 95)) if values else float("nan"),
                "tracker_p99_ms": float(np.percentile(values, 99)) if values else float("nan"),
                "good_frames": int(quality.get("GOOD", 0)),
                "degraded_frames": int(quality.get("DEGRADED", 0)),
                "invalid_frames": int(quality.get("INVALID", 0)),
            }
        )
    _write_rows(output / "alignment_summary.csv", alignment_rows)
    _write_rows(output / "runtime_summary.csv", runtime_rows)
    tracker = np.asarray(tracker_times, dtype=float)
    summary = {
        "records": len(runtime_rows),
        "alignment_records": len(alignment_rows),
        "quality_frame_totals": quality_totals,
        "tracker_blocks": int(tracker.size),
        "tracker_ms": (
            {
                "mean": float(np.mean(tracker)),
                "p50": float(np.percentile(tracker, 50)),
                "p95": float(np.percentile(tracker, 95)),
                "p99": float(np.percentile(tracker, 99)),
                "max": float(np.max(tracker)),
            }
            if tracker.size
            else {}
        ),
        "resource_snapshot": {
            "batch_peak_rss_kb": batch_peak_rss_kb,
            "batch_observed_rss_kb": batch_observed_rss_kb,
            "gpu_memory_mib": gpu_memory_mib,
            "scope": "one live process snapshot during record 85; not a per-block trace",
        },
        "performance_caveat": performance_caveat,
        "deployment_pga_allowed": False,
    }
    (output / "batch_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-peak-rss-kb", type=int)
    parser.add_argument("--batch-observed-rss-kb", type=int)
    parser.add_argument("--gpu-memory-mib", type=int)
    parser.add_argument("--performance-caveat", default="not a controlled realtime acceptance run unless documented otherwise")
    args = parser.parse_args()
    print(
        json.dumps(
            summarize(
                Path(args.run_root),
                Path(args.output),
                batch_peak_rss_kb=args.batch_peak_rss_kb,
                batch_observed_rss_kb=args.batch_observed_rss_kb,
                gpu_memory_mib=args.gpu_memory_mib,
                performance_caveat=args.performance_caveat,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
