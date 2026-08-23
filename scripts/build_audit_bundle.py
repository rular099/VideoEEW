#!/usr/bin/env python3
"""Build a compact, self-explaining review bundle for one VideoEEW run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np


CORE_FILES = (
    "manifest.json",
    "config.yaml",
    "git_diff.patch",
    "environment.txt",
    "device_info.txt",
    "metrics.json",
    "input_manifest.json",
    "git_status.txt",
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _numeric(rows: list[dict[str, str]], column: str) -> np.ndarray:
    values = []
    for row in rows:
        try:
            value = float(row.get(column, ""))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=float)


def _summary_table(rows: list[dict[str, str]], columns: tuple[str, ...]) -> list[dict[str, object]]:
    summary = []
    for column in columns:
        values = _numeric(rows, column)
        if values.size:
            summary.append(
                {
                    "metric": column,
                    "count": int(values.size),
                    "mean": float(np.mean(values)),
                    "p50": float(np.percentile(values, 50)),
                    "p95": float(np.percentile(values, 95)),
                    "p99": float(np.percentile(values, 99)),
                    "max": float(np.max(values)),
                }
            )
    return summary


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("metric,count,mean,p50,p95,p99,max\n", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_bundle(
    run_directory: Path,
    audit_root: Path,
    *,
    baseline_directory: Path | None = None,
    make_zip: bool = False,
) -> Path:
    run = run_directory.resolve()
    if not run.is_dir():
        raise FileNotFoundError(f"run directory not found: {run}")
    run_id = run.name
    audit = audit_root.resolve() / run_id
    audit.mkdir(parents=True, exist_ok=True)
    for name in CORE_FILES:
        source = run / name
        if source.is_file():
            shutil.copy2(source, audit / name)
    manifest = _read_json(run / "manifest.json")
    metrics = _read_json(run / "metrics.json")
    timing_rows = _read_csv(run / "timing.csv")
    memory_rows = _read_csv(run / "memory.csv")
    queue_rows = _read_csv(run / "queue.csv")
    motion_rows = _read_csv(run / "motion_quality.csv")
    pga_rows = _read_csv(run / "pga_predictions.csv")
    timing_summary = _summary_table(
        timing_rows,
        (
            "capture_ms",
            "preprocess_ms",
            "encoder_ms",
            "tracker_ms",
            "motion_fit_ms",
            "signal_ms",
            "pga_ms",
            "total_pipeline_ms",
            "end_to_end_latency_ms",
        ),
    )
    _write_csv(audit / "timing_summary.csv", timing_summary)
    memory_summary = _summary_table(
        memory_rows,
        ("rss_mb", "peak_rss_mb", "system_available_mem_mb"),
    )
    _write_csv(audit / "memory_summary.csv", memory_summary)
    queue_summary = _summary_table(
        queue_rows,
        (
            "capture_queue_depth",
            "tracker_queue_depth",
            "output_queue_depth",
            "dropped_frames",
            "dropped_blocks",
        ),
    )
    _write_csv(audit / "queue_summary.csv", queue_summary)
    quality_counts: dict[str, int] = {}
    for row in motion_rows:
        quality = row.get("quality", "UNKNOWN")
        quality_counts[quality] = quality_counts.get(quality, 0) + 1
    with (audit / "motion_quality_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["quality", "count"])
        writer.writerows(sorted(quality_counts.items()))
    if pga_rows:
        with (audit / "pga_predictions_sample.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(pga_rows[0]))
            writer.writeheader()
            writer.writerows(pga_rows[:100])
    plots_source = run / "plots"
    if plots_source.is_dir():
        plots_output = audit / "plots"
        plots_output.mkdir(exist_ok=True)
        for plot in plots_source.iterdir():
            if plot.is_file() and plot.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg", ".pdf"}:
                shutil.copy2(plot, plots_output / plot.name)
    tracker_timing = next(
        (row for row in timing_summary if row["metric"] == "tracker_ms"), {}
    )
    total_timing = next(
        (row for row in timing_summary if row["metric"] == "total_pipeline_ms"), {}
    )
    queue_depth_columns = (
        "capture_queue_depth",
        "tracker_queue_depth",
        "output_queue_depth",
    )
    queue_growth: dict[str, float] = {}
    for column in queue_depth_columns:
        values = _numeric(queue_rows, column)
        if values.size >= 2:
            queue_growth[column] = float(
                np.polyfit(np.arange(values.size, dtype=float), values, 1)[0]
            )
    dropped_frames = int(np.max(_numeric(queue_rows, "dropped_frames"), initial=0))
    dropped_blocks = int(np.max(_numeric(queue_rows, "dropped_blocks"), initial=0))
    overload_states = sorted(
        {row.get("overload_state", "") for row in queue_rows if row.get("overload_state")}
    )
    event_rows = []
    events_path = run / "events.jsonl"
    if events_path.is_file():
        for line in events_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    event_rows.append(json.loads(line))
                except json.JSONDecodeError:
                    event_rows.append({"event": "malformed_event_line"})
    event_counts: dict[str, int] = {}
    for event in event_rows:
        name = str(event.get("event", "UNKNOWN"))
        event_counts[name] = event_counts.get(name, 0) + 1
    peak_rss_values = _numeric(memory_rows, "peak_rss_mb")
    peak_rss = float(np.max(peak_rss_values)) if peak_rss_values.size else float("nan")
    baseline_metrics = _read_json(baseline_directory / "metrics.json") if baseline_directory else {}
    summary_lines = [
        f"# Audit summary: {run_id}",
        "",
        "## Scope and provenance",
        "",
        f"- Git commit: `{manifest.get('git_commit', 'unknown')}`; dirty: `{manifest.get('git_dirty', 'unknown')}`.",
        f"- Input: `{manifest.get('input_id', 'see input_manifest.json')}`.",
        f"- Device: `{manifest.get('device', 'unknown')}`.",
        f"- Checkpoint SHA-256: `{manifest.get('checkpoint_sha256', 'unknown')}`.",
        f"- Tracker parameters: `{json.dumps(manifest.get('tracker_parameters', {}), sort_keys=True)}`.",
        f"- Motion parameters: `{json.dumps(manifest.get('motion_parameters', {}), sort_keys=True)}`.",
        f"- Signal parameters: `{json.dumps(manifest.get('signal_parameters', {}), sort_keys=True)}`.",
        f"- PGA model: `{manifest.get('pga_model_version', None)}`.",
        "",
        "## Accuracy and quality",
        "",
        f"- Metrics: `{json.dumps(metrics, sort_keys=True)}`.",
        f"- Motion quality counts: `{json.dumps(quality_counts, sort_keys=True)}`.",
        f"- PGA rows included: `{len(pga_rows)}`.",
        f"- Scale state: `{metrics.get('scale_state', 'unknown')}`.",
        "",
        "## Runtime behavior",
        "",
        f"- Tracker latency summary: `{json.dumps(tracker_timing, sort_keys=True)}`.",
        f"- Total latency summary: `{json.dumps(total_timing, sort_keys=True)}`.",
        f"- Queue depth slopes per row: `{json.dumps(queue_growth, sort_keys=True)}`.",
        f"- Dropped frames / blocks: `{dropped_frames}` / `{dropped_blocks}`.",
        f"- Overload states: `{overload_states}`.",
        f"- Peak RSS: `{peak_rss if np.isfinite(peak_rss) else 'not recorded'} MB`.",
        f"- Event counts: `{json.dumps(event_counts, sort_keys=True)}`.",
        "",
        "## Baseline comparison and known limitations",
        "",
        f"- Baseline metrics: `{json.dumps(baseline_metrics, sort_keys=True) if baseline_metrics else 'not supplied'}`.",
        "- Large videos, checkpoints, raw tracks and binary signals are intentionally excluded from this compact bundle.",
        "- Any missing field is visible as missing; the builder does not invent measurements.",
        "- Failed/degraded samples are represented by motion-quality counts and events; raw large artifacts remain in the source run.",
        "",
    ]
    (audit / "AUDIT_SUMMARY.md").write_text("\n".join(summary_lines), encoding="utf-8")
    if make_zip:
        shutil.make_archive(str(audit), "zip", root_dir=audit.parent, base_dir=audit.name)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--audit-root", default="audit")
    parser.add_argument("--baseline")
    parser.add_argument("--zip", action="store_true")
    args = parser.parse_args()
    output = build_bundle(
        Path(args.run),
        Path(args.audit_root),
        baseline_directory=Path(args.baseline) if args.baseline else None,
        make_zip=args.zip,
    )
    print(output)


if __name__ == "__main__":
    main()
