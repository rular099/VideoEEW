#!/usr/bin/env python3
"""Run >=1000 maximum-search null surrogates on cached paired-record signals."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from seismic_motion.pga.alignment import acceleration_to_displacement, full_containment_offset_range
from seismic_motion.pga.alignment_null import alignment_null_test, benjamini_hochberg
from seismic_motion.pga.records import discover_dataset_pairs, load_strong_motion_txt


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("NOT_MEASURED\n", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--record-ids", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument(
        "--null-method", choices=("circular_shift", "phase_randomized"), default="circular_shift"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.iterations < 1000:
        raise ValueError("audit execution requires at least 1000 null iterations")
    pairs = {pair.record_id: pair for pair in discover_dataset_pairs(args.data_root)}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    distribution: list[dict[str, object]] = []
    candidates: list[dict[str, object]] = []
    for record_index, record_id in enumerate(args.record_ids):
        pair = pairs[record_id]
        signals = np.load(Path(args.run_root) / f"record-{record_id}" / "filtered_signals.npz")
        record = load_strong_motion_txt(pair.strong_motion_paths[0])
        sensor_acceleration = np.stack([record.ew_gal, record.ns_gal], axis=1)
        visual_timestamps = signals["timestamps"]
        offset_range = full_containment_offset_range(visual_timestamps, record.timestamps_s)
        summary, rows, candidate = alignment_null_test(
            record_id,
            visual_timestamps,
            {
                "acceleration": signals["acceleration_proxy_px_s2"],
                "displacement": signals["filtered_common_xy_px"],
            },
            record.timestamps_s,
            {
                "acceleration": sensor_acceleration,
                "displacement": acceleration_to_displacement(
                    record.timestamps_s, sensor_acceleration
                ),
            },
            offset_range_s=offset_range,
            iterations=args.iterations,
            seed=args.seed + record_index,
            null_method=args.null_method,
        )
        summaries.append(asdict(summary))
        distribution.extend(rows)
        candidates.append(candidate)
        print(f"completed null {record_id}: p={summary.empirical_p_value:.6g}", flush=True)
    q_values = benjamini_hochberg(
        np.asarray([float(row["empirical_p_value"]) for row in summaries])
    )
    for row, q_value in zip(summaries, q_values):
        row["fdr_q_value"] = float(q_value)
    _write(output / "alignment_candidates.csv", candidates)
    _write(output / "null_max_corr_distribution.csv", distribution)
    _write(output / "alignment_significance.csv", summaries)
    (output / "alignment_null_config.json").write_text(
        json.dumps(vars(args), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
