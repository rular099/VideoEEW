#!/usr/bin/env python3
"""Extract one leakage-safe feature row per selected video/strong-motion record."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from seismic_motion.config import load_config
from seismic_motion.pga.alignment import (
    acceleration_to_displacement,
    estimate_time_offset,
    full_containment_offset_range,
)
from seismic_motion.pga.records import discover_dataset_pairs, load_strong_motion_txt
from seismic_motion.runtime.pipeline import run_offline_video, write_run_artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--record-ids", nargs="+", required=True)
    parser.add_argument("--config", default="configs/real_video_eval.yaml")
    parser.add_argument("--cotracker-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    pairs = {pair.record_id: pair for pair in discover_dataset_pairs(args.data_root)}
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    feature_rows: list[dict[str, object]] = []
    for record_id in args.record_ids:
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
            alignment_payload = json.loads(cached_alignment.read_text(encoding="utf-8"))
            record = load_strong_motion_txt(pair.strong_motion_paths[0])
        else:
            result = run_offline_video(
                pair.video_path,
                config,
                cotracker_root=args.cotracker_root,
                checkpoint=args.checkpoint,
                device=args.device,
            )
            write_run_artifacts(run_dir, result, config)
            record = load_strong_motion_txt(pair.strong_motion_paths[0])
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
                "interpretation": "exploratory candidate; full-record search is not synchronization proof",
            }
            cached_alignment.write_text(
                json.dumps(alignment_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            features = {key: value for key, value in result.features.items()}
        preferred_domain = str(alignment_payload["preferred"])
        preferred_alignment = alignment_payload[preferred_domain]
        feature_rows.append(
            {
                "record_id": record_id,
                "event_id": record_id,
                "camera_id": "unknown_same_setup",
                "site_id": "shake_table_setup_unknown",
                "scale_state": "UNCALIBRATED",
                "pga_horizontal_vector_gal": record.pga_gal("horizontal_vector"),
                "alignment_domain": preferred_domain,
                "alignment_offset_s": preferred_alignment["offset_s"],
                "alignment_correlation": preferred_alignment["correlation"],
                "alignment_status": preferred_alignment["status"],
                "alignment_acceleration_correlation": alignment_payload["acceleration"]["correlation"],
                "alignment_displacement_correlation": alignment_payload["displacement"]["correlation"],
                **features,
            }
        )
        feature_path = output_root / "pga_features.csv"
        with feature_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(feature_rows[0]))
            writer.writeheader()
            writer.writerows(feature_rows)
        print(f"completed record {record_id}: {len(feature_rows)}/{len(args.record_ids)}", flush=True)


if __name__ == "__main__":
    main()
