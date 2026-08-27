#!/usr/bin/env python3
"""Extract one leakage-safe feature row per selected video/strong-motion record."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from seismic_motion.config import load_config
from seismic_motion.pga.alignment import (
    acceleration_to_displacement,
    estimate_time_offset,
    full_containment_offset_range,
)
from seismic_motion.pga.records import discover_dataset_pairs, load_strong_motion_files
from seismic_motion.runtime.pipeline import run_offline_video, write_run_artifacts


FAILURE_FIELDS = (
    "record_id",
    "status",
    "exception_type",
    "reason",
    "attempt_utc",
)


def _read_failure_rows(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_failure_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FAILURE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _migrate_failure_history(current: Path, history: Path) -> list[dict[str, object]]:
    rows = _read_failure_rows(history)
    for previous in _read_failure_rows(current):
        previous.setdefault("attempt_utc", "UNKNOWN_PREVIOUS_ATTEMPT")
        if previous not in rows:
            rows.append(previous)
    if rows:
        _write_failure_rows(history, rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--record-ids", nargs="+")
    selection.add_argument("--all-paired", action="store_true")
    parser.add_argument("--config", default="configs/real_video_eval.yaml")
    parser.add_argument("--cotracker-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    pairs = {pair.record_id: pair for pair in discover_dataset_pairs(args.data_root)}
    record_ids = (
        [
            record_id
            for record_id, pair in pairs.items()
            if pair.pairing_status in {"paired", "paired_split_sensor"}
        ]
        if args.all_paired
        else list(args.record_ids or [])
    )
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    feature_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []
    current_failure_path = output_root / "failed_records.csv"
    failure_history_path = output_root / "failure_history.csv"
    failure_history = _migrate_failure_history(
        current_failure_path, failure_history_path
    )
    for record_id in record_ids:
        try:
            if record_id not in pairs:
                raise KeyError(f"record not found: {record_id}")
            pair = pairs[record_id]
            if not pair.video_path or not pair.strong_motion_paths:
                raise RuntimeError(f"record {record_id} is not paired")
            run_dir = output_root / f"record-{record_id}"
            cached_features = run_dir / "features.csv"
            cached_alignment = run_dir / "alignment_v2.json"
            if cached_features.is_file() and cached_alignment.is_file():
                with cached_features.open("r", encoding="utf-8") as handle:
                    features = next(csv.DictReader(handle))
                cached_causal = str(features.get("causal", "0")) == str(
                    int(bool(config["signal"]["causal"]))
                )
                if not cached_causal:
                    raise RuntimeError(
                        "cached features have a different causal mode; use a fresh output root"
                    )
                alignment_payload = json.loads(cached_alignment.read_text(encoding="utf-8"))
                record = load_strong_motion_files(pair.strong_motion_paths)
                input_manifest = json.loads(
                    (run_dir / "input_manifest.json").read_text(encoding="utf-8")
                )
            else:
                result = run_offline_video(
                    pair.video_path,
                    config,
                    cotracker_root=args.cotracker_root,
                    checkpoint=args.checkpoint,
                    device=args.device,
                )
                write_run_artifacts(run_dir, result, config)
                input_manifest = result.video_metadata
                record = load_strong_motion_files(pair.strong_motion_paths)
                offset_range = full_containment_offset_range(
                    result.tracks.timestamps, record.timestamps_s
                )
                acceleration_alignment = estimate_time_offset(
                    result.tracks.timestamps,
                    result.acceleration_proxy_px_s2,
                    record.timestamps_s,
                    np.stack([record.ew_gal, record.ns_gal], axis=1),
                    offset_range_s=offset_range,
                    min_correlation=0.4,
                )
                displacement_alignment = estimate_time_offset(
                    result.tracks.timestamps,
                    result.filtered_common_xy_px,
                    record.timestamps_s,
                    acceleration_to_displacement(
                        record.timestamps_s,
                        np.stack([record.ew_gal, record.ns_gal], axis=1),
                    ),
                    offset_range_s=offset_range,
                    min_correlation=0.4,
                )
                preferred = (
                    "acceleration"
                    if acceleration_alignment.correlation >= displacement_alignment.correlation
                    else "displacement"
                )
                alignment_payload = {
                    "offset_definition": "compare visual(t) with sensor(t + offset_s)",
                    "searched_offset_range_s": list(offset_range),
                    "acceleration": asdict(acceleration_alignment),
                    "displacement": asdict(displacement_alignment),
                    "preferred": preferred,
                    "interpretation": "RESEARCH_DIAGNOSTIC_ONLY_NOT_DEPLOYABLE",
                }
                cached_alignment.write_text(
                    json.dumps(alignment_payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                features = {key: value for key, value in result.features.items()}
            preferred_domain = str(alignment_payload["preferred"])
            preferred_alignment = alignment_payload[preferred_domain]
            source_shape = input_manifest.get("source_shape") or ["UNKNOWN", "UNKNOWN"]
            feature_rows.append(
                {
                    "record_id": record_id,
                    "event_id": "UNKNOWN",
                    "camera_id": "UNKNOWN",
                    "camera_model": "UNKNOWN",
                    "site_id": "UNKNOWN",
                    "building_id": "UNKNOWN",
                    "mount_type": "UNKNOWN",
                    "video_start_time": "UNKNOWN",
                    "sensor_start_time": "UNKNOWN",
                    "scene_type": "UNKNOWN",
                    "fps": input_manifest.get("fps", "UNKNOWN"),
                    "resolution": f"{source_shape[1]}x{source_shape[0]}",
                    "duration_s": features.get("window_end", "UNKNOWN"),
                    "scale_state": "UNCALIBRATED",
                    "mm_per_px": "UNKNOWN",
                    "pga_horizontal_vector_gal": record.pga_gal("horizontal_vector"),
                    "alignment_domain": preferred_domain,
                    "alignment_offset_s": preferred_alignment["offset_s"],
                    "alignment_correlation": preferred_alignment["correlation"],
                    "alignment_status": preferred_alignment["status"],
                    "alignment_interpretation": "RESEARCH_DIAGNOSTIC_ONLY_NOT_DEPLOYABLE",
                    "alignment_acceleration_correlation": alignment_payload["acceleration"]["correlation"],
                    "alignment_displacement_correlation": alignment_payload["displacement"]["correlation"],
                    **features,
                }
            )
        except Exception as exc:
            failure_rows.append(
                {
                    "record_id": record_id,
                    "status": "FAILED",
                    "exception_type": type(exc).__name__,
                    "reason": str(exc),
                    "attempt_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
            failure_history.append(failure_rows[-1])
            if args.fail_fast:
                raise
        feature_path = output_root / "pga_features.csv"
        if feature_rows:
            with feature_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(feature_rows[0]))
                writer.writeheader()
                writer.writerows(feature_rows)
        _write_failure_rows(current_failure_path, failure_rows)
        if failure_history:
            _write_failure_rows(failure_history_path, failure_history)
        print(
            f"processed record {record_id}: success={len(feature_rows)} failed={len(failure_rows)} "
            f"total={len(record_ids)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
