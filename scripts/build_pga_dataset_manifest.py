#!/usr/bin/env python3
"""Pair videos with strong-motion records without copying private data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from seismic_motion.pga.records import discover_dataset_pairs, load_strong_motion_txt


def _video_metadata(path: str) -> dict[str, float | int | str]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required to inspect video metadata") from exc
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        return {
            "video_status": "unreadable",
            "fps": 0.0,
            "width": 0,
            "height": 0,
            "frame_count": 0,
            "duration_s": 0.0,
        }
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    return {
        "video_status": "readable" if fps > 0 and frames > 0 else "metadata_incomplete",
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": frames,
        "duration_s": frames / fps if fps > 0 else 0.0,
    }


def build_rows(data_root: str | Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for pair in discover_dataset_pairs(data_root):
        video = _video_metadata(pair.video_path) if pair.video_path else {
            "video_status": "missing",
            "fps": 0.0,
            "width": 0,
            "height": 0,
            "frame_count": 0,
            "duration_s": 0.0,
        }
        records = [load_strong_motion_txt(path) for path in pair.strong_motion_paths]
        pga_horizontal = [record.pga_gal("horizontal_vector") for record in records]
        pga_component = [record.pga_gal("max_horizontal_component") for record in records]
        rows.append(
            {
                "record_id": pair.record_id,
                "pairing_status": pair.pairing_status,
                "video_path": pair.video_path,
                **video,
                "timestamp_source": "container_constant_frame_rate_unverified_pts",
                "strong_motion_paths": "|".join(pair.strong_motion_paths),
                "strong_motion_segments": len(records),
                "strong_motion_samples": sum(record.timestamps_s.size for record in records),
                "strong_motion_duration_s": sum(
                    record.timestamps_s[-1] - record.timestamps_s[0] for record in records
                ),
                "pga_horizontal_vector_gal": max(pga_horizontal, default=float("nan")),
                "pga_max_horizontal_component_gal": max(pga_component, default=float("nan")),
                "event_group": pair.record_id,
                "camera_id": "unknown",
                "site_id": "shake_table_setup_unknown",
                "scale_state": "UNCALIBRATED",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary")
    args = parser.parse_args()
    rows = build_rows(args.data_root)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row["pairing_status"])
        counts[status] = counts.get(status, 0) + 1
    valid_pga = [
        float(row["pga_horizontal_vector_gal"])
        for row in rows
        if row["pairing_status"].startswith("paired")
    ]
    summary = {
        "records": len(rows),
        "pairing_status_counts": counts,
        "paired_pga_horizontal_vector_gal": {
            "minimum": min(valid_pga),
            "maximum": max(valid_pga),
            "median": sorted(valid_pga)[len(valid_pga) // 2],
        },
        "deployment_pga_allowed": False,
        "reason": "geometric scale and exact video/sensor alignment are not yet established",
    }
    if args.summary:
        Path(args.summary).write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

