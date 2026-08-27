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
    "batch_summary.json",
    "alignment_summary.csv",
    "runtime_summary.csv",
    "pga_metrics_by_group.csv",
    "git_commit.txt",
    "git_status_porcelain.txt",
    "git_diff_cached.patch",
    "effective_config.yaml",
    "model_config.yaml",
    "signal_config.yaml",
    "runtime_config.yaml",
    "deployment_config.yaml",
    "runtime_timing.csv",
    "runtime_queue.csv",
    "runtime_memory.csv",
    "runtime_events.csv",
    "pga_eval_all.csv",
    "pga_eval_video_quality.csv",
    "pga_eval_posthoc_aligned.csv",
    "pga_metrics.json",
    "pga_bootstrap_ci.json",
    "pga_model_research.json",
    "alignment_candidates.csv",
    "null_max_corr_distribution.csv",
    "alignment_significance.csv",
    "tracking_stress_summary.csv",
    "common_motion_metrics.csv",
    "local_motion_metrics.csv",
    "rotation_metrics.csv",
    "stress_manifest.json",
    "stress_failures.csv",
    "reseed_boundary.csv",
    "reseed_summary.json",
    "RK3588_STATUS.md",
    "IMPLEMENTATION_STATUS.md",
)

REQUIRED_V2_FILES = (
    "git_commit.txt",
    "git_status_porcelain.txt",
    "git_diff.patch",
    "git_diff_cached.patch",
    "effective_config.yaml",
    "model_config.yaml",
    "signal_config.yaml",
    "runtime_config.yaml",
    "deployment_config.yaml",
    "runtime_timing.csv",
    "runtime_queue.csv",
    "runtime_memory.csv",
    "runtime_events.csv",
    "pga_eval_all.csv",
    "pga_eval_video_quality.csv",
    "pga_eval_posthoc_aligned.csv",
    "pga_metrics.json",
    "pga_bootstrap_ci.json",
    "alignment_candidates.csv",
    "null_max_corr_distribution.csv",
    "alignment_significance.csv",
    "tracking_stress_summary.csv",
    "common_motion_metrics.csv",
    "local_motion_metrics.csv",
    "rotation_metrics.csv",
    "reseed_boundary.csv",
    "RK3588_STATUS.md",
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # Composite audit placeholders intentionally contain NOT_MEASURED.
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_first_csv(run: Path, *names: str) -> list[dict[str, str]]:
    for name in names:
        path = run / name
        if path.is_file():
            return _read_csv(path)
    return []


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
    for name in REQUIRED_V2_FILES:
        destination = audit / name
        if not destination.exists():
            destination.write_text("NOT_MEASURED\n", encoding="utf-8")
    manifest = _read_json(run / "manifest.json")
    metrics = _read_json(run / "metrics.json")
    if not metrics:
        metrics = _read_json(run / "pga_metrics.json")
        if metrics:
            shutil.copy2(run / "pga_metrics.json", audit / "metrics.json")
    batch_summary = _read_json(run / "batch_summary.json")
    diff_text = (run / "git_diff.patch").read_text(encoding="utf-8") if (run / "git_diff.patch").is_file() else ""
    changed_paths = sorted(
        {
            line.split(" b/", 1)[1]
            for line in diff_text.splitlines()
            if line.startswith("diff --git a/") and " b/" in line
        }
    )
    added_lines = sum(
        1
        for line in diff_text.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    removed_lines = sum(
        1
        for line in diff_text.splitlines()
        if line.startswith("-") and not line.startswith("---")
    )
    timing_rows = _read_first_csv(run, "runtime_timing.csv", "timing.csv")
    memory_rows = _read_first_csv(run, "runtime_memory.csv", "memory.csv")
    queue_rows = _read_first_csv(run, "runtime_queue.csv", "queue.csv")
    motion_rows = _read_csv(run / "motion_quality.csv")
    pga_rows = _read_csv(run / "pga_predictions.csv")
    timing_summary = _summary_table(
        timing_rows,
        (
            "capture_to_tracker_end",
            "capture_to_pga",
            "queue_wait",
            "tracker_compute",
            "postprocess_compute",
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
    if not quality_counts and batch_summary.get("quality_frame_totals"):
        quality_counts = {
            str(name): int(count)
            for name, count in batch_summary["quality_frame_totals"].items()
        }
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
            if plot.is_file() and plot.suffix.lower() in {
                ".png",
                ".jpg",
                ".jpeg",
                ".svg",
                ".pdf",
                ".json",
            }:
                shutil.copy2(plot, plots_output / plot.name)
    tracker_timing = next(
        (row for row in timing_summary if row["metric"] == "tracker_ms"), {}
    )
    if not tracker_timing and batch_summary.get("tracker_ms"):
        tracker_timing = {
            "metric": "tracker_ms",
            "count": batch_summary.get("tracker_blocks"),
            **batch_summary["tracker_ms"],
        }
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
    dropped_frame_values = _numeric(queue_rows, "dropped_frames")
    dropped_block_values = _numeric(queue_rows, "dropped_blocks")
    dropped_frames = (
        int(np.max(dropped_frame_values)) if dropped_frame_values.size else None
    )
    dropped_blocks = (
        int(np.max(dropped_block_values)) if dropped_block_values.size else None
    )
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
    elif (run / "runtime_events.csv").is_file():
        event_rows = _read_csv(run / "runtime_events.csv")
    event_counts: dict[str, int] = {}
    for event in event_rows:
        name = str(event.get("event", "UNKNOWN"))
        event_counts[name] = event_counts.get(name, 0) + 1
    peak_rss_values = _numeric(memory_rows, "peak_rss_mb")
    peak_rss = float(np.max(peak_rss_values)) if peak_rss_values.size else float("nan")
    if not np.isfinite(peak_rss):
        peak_kb = batch_summary.get("resource_snapshot", {}).get("batch_peak_rss_kb")
        if peak_kb is not None:
            peak_rss = float(peak_kb) / 1024.0
    baseline_metrics = _read_json(baseline_directory / "metrics.json") if baseline_directory else {}
    target_fps = metrics.get("target_fps")
    runtime_status = str(metrics.get("realtime_acceptance", "NOT_TESTED"))
    pc_30_status = (
        runtime_status if target_fps is not None and abs(float(target_fps) - 30.0) < 0.1 else str(metrics.get("pc_30_fps_realtime", "NOT_TESTED"))
    )
    pc_50_status = (
        runtime_status if target_fps is not None and abs(float(target_fps) - 50.0) < 0.1 else str(metrics.get("pc_50_fps_realtime", "NOT_TESTED"))
    )
    pga_details = _read_json(run / "pga_metrics.json")
    reseed_details = _read_json(run / "reseed_summary.json")
    stress_details = _read_json(run / "stress_manifest.json")
    causal_pga_status = str(
        metrics.get(
            "causal_pga_status", pga_details.get("causal_pga_status", "NOT_TESTED")
        )
    )
    if causal_pga_status == "PASS_EVENT_END_ONLY":
        causal_pga_status = "PASS"
    scientific_validity = str(metrics.get("scientific_validity", "RESEARCH_ONLY"))
    geometric_scale = str(metrics.get("geometric_scale", metrics.get("scale_state", "UNCALIBRATED")))
    signal_causality = str(metrics.get("signal_pga_causality", causal_pga_status))
    tracker_causality = str(
        metrics.get("tracker_source_timestamp_causality", "NOT_TESTED")
    )
    end_to_end_causality = str(
        metrics.get("end_to_end_source_timestamp_causality", "NOT_TESTED")
    )
    dropped_frames_for_review = metrics.get("dropped_frames", dropped_frames)
    dropped_blocks_for_review = metrics.get("dropped_blocks", dropped_blocks)
    captured_for_review = metrics.get("captured_frames")
    written_for_review = metrics.get("frames_written")
    if dropped_frames_for_review is None or dropped_blocks_for_review is None:
        silent_drop_status = "NOT_TESTED"
    elif (
        captured_for_review is not None
        and written_for_review is not None
        and int(captured_for_review) != int(written_for_review)
        and int(dropped_frames_for_review) == 0
        and int(dropped_blocks_for_review) == 0
    ):
        silent_drop_status = "FAIL_POSSIBLE_UNACCOUNTED_FRAME_LOSS"
    elif int(dropped_frames_for_review) == 0 and int(dropped_blocks_for_review) == 0:
        silent_drop_status = "PASS_NO_DROP_RECORDED"
    else:
        silent_drop_status = "NO_SILENT_DROP_EXPLICIT_REJECTION_ACCEPTANCE_FAIL"
    reseed_p95 = reseed_details.get("acceleration_spike_ratio_p95")
    reseed_events = reseed_details.get("reseed_events_analyzed")
    if reseed_p95 is None:
        reseed_status = "NOT_EVALUABLE"
    elif float(reseed_p95) > 3.0:
        # One observed harmful boundary is sufficient to require review.
        reseed_status = "FAIL_REVIEW_REQUIRED"
    elif reseed_events is None:
        reseed_status = "NOT_EVALUABLE_EVENT_COUNT_UNKNOWN"
    elif int(reseed_events) < 2:
        # Do not turn one benign boundary into a general safety claim.
        reseed_status = "NOT_EVALUABLE_SINGLE_EVENT_BELOW_THRESHOLD"
    else:
        reseed_status = "PASS_P95_RATIO_LE_3"
    subsets = pga_details.get("subsets", {})
    all_metrics = subsets.get("all", {}) if isinstance(subsets, dict) else {}
    quality_metrics = subsets.get("video_quality", {}) if isinstance(subsets, dict) else {}
    all_primary = all_metrics.get("algorithms", {}).get(
        pga_details.get("primary_algorithm", "single_coefficient"), {}
    )
    quality_primary = quality_metrics.get("algorithms", {}).get(
        pga_details.get("primary_algorithm", "single_coefficient"), {}
    )
    truth_blind_selection = bool(
        all_metrics
        and quality_metrics
        and not all_metrics.get("selection_uses_strong_motion", True)
        and not quality_metrics.get("selection_uses_strong_motion", True)
    )
    summary_lines = [
        f"# Audit summary: {run_id}",
        "",
        "## Deployment status",
        "",
        f"- PC 30 FPS realtime: {pc_30_status}",
        f"- 50 FPS realtime: {pc_50_status}",
        "- RK3588 realtime: BLOCKED",
        f"- Causal PGA: {causal_pga_status}",
        f"- Signal/PGA zero-lookahead causality: {signal_causality}",
        f"- Tracker source-timestamp causality: {tracker_causality}",
        f"- End-to-end source-timestamp causality: {end_to_end_causality}",
        f"- PGA scientific validity: {scientific_validity}",
        f"- Geometric scale: {geometric_scale}",
        "- Strict-causality interpretation: online availability alone is not treated as "
        "source-timestamp causality; see the manifest tracker future-context range.",
        "",
        "## Required review questions (A-L)",
        "",
        f"- A. Strict realtime causality: `{end_to_end_causality}`; signal/PGA `{signal_causality}`, tracker `{tracker_causality}`.",
        f"- B. PC 30 FPS without backlog: `{pc_30_status}`.",
        f"- C. PC 50 FPS realtime: `{pc_50_status}`.",
        f"- D. Silent frame drop: `{silent_drop_status}`; frames/blocks `{dropped_frames_for_review}` / `{dropped_blocks_for_review}`.",
        f"- E. Reseed fake peak: `{reseed_status}`; analyzed events `{reseed_events}`, acceleration-spike p95 ratio `{reseed_p95}`.",
        f"- F. Strong-motion data used for ALL/VIDEO-QUALITY selection: `{'NO' if truth_blind_selection else 'NOT_VERIFIED'}`.",
        f"- G. ALL primary PGA metrics: `{json.dumps(all_primary, sort_keys=True) if all_primary else 'NOT_EVALUABLE'}`.",
        f"- H. VIDEO-QUALITY primary PGA metrics: `{json.dumps(quality_primary, sort_keys=True) if quality_primary else 'NOT_EVALUABLE'}`.",
        "- I. Offline versus causal difference: see `offline_vs_causal_pga.png` and the causal signal benchmark; `NOT_EVALUABLE` if absent.",
        f"- J. CoTracker common/local/rotation stress evidence: `{json.dumps(stress_details, sort_keys=True) if stress_details else 'NOT_MEASURED'}`.",
        "- K. RK3588 measured: `NO_BLOCKED_NO_DEVICE`.",
        f"- L. Evidence commit/config: `{manifest.get('git_commit', 'unknown')}` / `effective_config.yaml`.",
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
        f"- Change summary: `{manifest.get('change_summary', 'not recorded')}`.",
        f"- Working-tree patch: `{len(changed_paths)} files, +{added_lines}/-{removed_lines} lines`.",
        f"- Changed paths: `{changed_paths[:30]}`.",
        "",
        "## Accuracy and quality",
        "",
        f"- Metrics: `{json.dumps(metrics, sort_keys=True)}`.",
        f"- Motion quality counts: `{json.dumps(quality_counts, sort_keys=True)}`.",
        f"- PGA rows included: `{len(pga_rows)}`.",
        f"- Scale state: `{metrics.get('scale_state', manifest.get('scale_parameters', {}).get('method', 'unknown'))}`.",
        "",
        "## Runtime behavior",
        "",
        f"- Tracker latency summary: `{json.dumps(tracker_timing, sort_keys=True)}`.",
        f"- Total latency summary: `{json.dumps(total_timing, sort_keys=True)}`.",
        f"- Queue depth slopes per row: `{json.dumps(queue_growth, sort_keys=True)}`.",
        f"- Dropped frames / blocks: `{dropped_frames if dropped_frames is not None else 'not recorded'}` / `{dropped_blocks if dropped_blocks is not None else 'not recorded'}`.",
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
