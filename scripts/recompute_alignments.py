#!/usr/bin/env python3
"""Recompute acceleration and displacement alignment without rerunning tracking."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from seismic_motion.pga.alignment import (
    acceleration_to_displacement,
    estimate_time_offset,
    full_containment_offset_range,
)
from seismic_motion.pga.records import discover_dataset_pairs, load_strong_motion_txt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--step-s", type=float, default=0.02)
    args = parser.parse_args()
    pairs = {pair.record_id: pair for pair in discover_dataset_pairs(args.data_root)}
    run_root = Path(args.run_root)
    for run_dir in sorted(run_root.glob("record-*")):
        record_id = run_dir.name.removeprefix("record-")
        pair = pairs.get(record_id)
        signals_path = run_dir / "filtered_signals.npz"
        if pair is None or not pair.strong_motion_paths or not signals_path.is_file():
            continue
        visual = np.load(signals_path)
        record = load_strong_motion_txt(pair.strong_motion_paths[0])
        sensor_horizontal = np.stack([record.ew_gal, record.ns_gal], axis=1)
        offset_range = full_containment_offset_range(
            visual["timestamps"], record.timestamps_s
        )
        acceleration_alignment = estimate_time_offset(
            visual["timestamps"],
            visual["acceleration_proxy_px_s2"],
            record.timestamps_s,
            sensor_horizontal,
            offset_range_s=offset_range,
            step_s=args.step_s,
            min_correlation=0.4,
        )
        sensor_displacement = acceleration_to_displacement(
            record.timestamps_s, sensor_horizontal
        )
        displacement_alignment = estimate_time_offset(
            visual["timestamps"],
            visual["filtered_common_xy_px"],
            record.timestamps_s,
            sensor_displacement,
            offset_range_s=offset_range,
            step_s=args.step_s,
            min_correlation=0.4,
        )
        payload = {
            "offset_definition": "compare visual(t) with sensor(t + offset_s)",
            "searched_offset_range_s": list(offset_range),
            "acceleration": asdict(acceleration_alignment),
            "displacement": asdict(displacement_alignment),
            "preferred": (
                "acceleration"
                if acceleration_alignment.correlation >= displacement_alignment.correlation
                else "displacement"
            ),
            "interpretation": "exploratory candidate; full-record search is not synchronization proof",
        }
        (run_dir / "alignment_v2.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            f"{record_id}: acc={acceleration_alignment.correlation:.3f} "
            f"disp={displacement_alignment.correlation:.3f}",
            flush=True,
        )
    feature_path = run_root / "pga_features.csv"
    if feature_path.is_file():
        with feature_path.open("r", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            path = run_root / f"record-{row['record_id']}" / "alignment_v2.json"
            if not path.is_file():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            domain = str(payload["preferred"])
            preferred = payload[domain]
            row.update(
                {
                    "alignment_domain": domain,
                    "alignment_offset_s": preferred["offset_s"],
                    "alignment_correlation": preferred["correlation"],
                    "alignment_status": preferred["status"],
                    "alignment_acceleration_correlation": payload["acceleration"]["correlation"],
                    "alignment_displacement_correlation": payload["displacement"]["correlation"],
                }
            )
        if rows:
            fields = list(rows[0])
            with feature_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)


if __name__ == "__main__":
    main()
