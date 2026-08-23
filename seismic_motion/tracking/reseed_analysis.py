"""Quantify continuity errors and signal spikes at explicit tracker reseeds."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


FIELDS = (
    "reseed_id",
    "frame_id",
    "timestamp",
    "point_position_jump_mean_px",
    "point_position_jump_p95_px",
    "common_translation_jump_px",
    "common_rotation_jump_deg",
    "velocity_jump",
    "acceleration_spike",
    "acceleration_spike_ratio",
    "quality_before",
    "quality_after",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def analyze_reseed_arrays(
    timestamps: np.ndarray,
    frame_indices: np.ndarray,
    tracks_xy: np.ndarray,
    common_xy: np.ndarray,
    rotation_rad: np.ndarray,
    acceleration_xy: np.ndarray,
    quality: np.ndarray,
    reseed_events: list[dict[str, Any]],
) -> list[dict[str, float | int | str]]:
    times = np.asarray(timestamps, dtype=np.float64)
    frames = np.asarray(frame_indices, dtype=np.int64)
    tracks = np.asarray(tracks_xy, dtype=np.float64)
    common = np.asarray(common_xy, dtype=np.float64)
    rotation = np.asarray(rotation_rad, dtype=np.float64)
    acceleration = np.asarray(acceleration_xy, dtype=np.float64)
    qualities = np.asarray(quality, dtype=str)
    rows: list[dict[str, float | int | str]] = []
    for event in reseed_events:
        frame_id = int(event["frame_index"])
        matches = np.flatnonzero(frames == frame_id)
        if matches.size == 0:
            continue
        index = int(matches[0])
        if index < 2 or index + 1 >= times.size:
            continue
        predicted_points = 2 * tracks[index - 1] - tracks[index - 2]
        point_jump = np.linalg.norm(tracks[index] - predicted_points, axis=1)
        predicted_common = 2 * common[index - 1] - common[index - 2]
        common_jump = float(np.linalg.norm(common[index] - predicted_common))
        predicted_rotation = 2 * rotation[index - 1] - rotation[index - 2]
        rotation_jump = float(np.degrees(abs(rotation[index] - predicted_rotation)))
        velocity_before = (common[index - 1] - common[index - 2]) / (
            times[index - 1] - times[index - 2]
        )
        velocity_after = (common[index] - common[index - 1]) / (
            times[index] - times[index - 1]
        )
        velocity_jump = float(np.linalg.norm(velocity_after - velocity_before))
        acceleration_magnitude = np.linalg.norm(acceleration, axis=1)
        spike = float(acceleration_magnitude[index])
        radius = max(3, int(round(0.5 / np.median(np.diff(times)))))
        nearby = acceleration_magnitude[max(0, index - radius) : min(times.size, index + radius + 1)]
        baseline = float(np.nanmedian(nearby))
        rows.append(
            {
                "reseed_id": int(event.get("reseed_id", len(rows) + 1)),
                "frame_id": frame_id,
                "timestamp": float(times[index]),
                "point_position_jump_mean_px": float(np.nanmean(point_jump)),
                "point_position_jump_p95_px": float(np.nanpercentile(point_jump, 95)),
                "common_translation_jump_px": common_jump,
                "common_rotation_jump_deg": rotation_jump,
                "velocity_jump": velocity_jump,
                "acceleration_spike": spike,
                "acceleration_spike_ratio": spike / max(baseline, 1e-12),
                "quality_before": str(qualities[index - 1]),
                "quality_after": str(qualities[index]),
            }
        )
    return rows


def analyze_reseed_run(run_directory: str | Path, output_directory: str | Path) -> list[dict[str, object]]:
    run = Path(run_directory)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    tracks = np.load(run / "tracks.npz")
    common_rows = _read_csv(run / "common_motion.csv")
    signals = np.load(run / "filtered_signals.npz")
    events = [
        event
        for event in _events(run / "events.jsonl")
        if event.get("event") == "tracker_reseed"
    ]
    common_xy = np.asarray(
        [[float(row["tx_px"]), float(row["ty_px"])] for row in common_rows]
    )
    rotation = np.asarray([float(row["rotation_2d_rad"]) for row in common_rows])
    quality = np.asarray([row["motion_quality"] for row in common_rows])
    rows = analyze_reseed_arrays(
        tracks["timestamps"],
        tracks["frame_indices"],
        tracks["xy_px"],
        common_xy,
        rotation,
        signals["acceleration_proxy_px_s2"],
        quality,
        events,
    )
    with (output / "reseed_boundary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    plot_status = "NOT_GENERATED_NO_RESEEDS"
    if rows:
        import matplotlib.pyplot as plt

        times = np.asarray(tracks["timestamps"], dtype=np.float64)
        acceleration = np.asarray(signals["acceleration_proxy_px_s2"], dtype=np.float64)
        residual = np.load(run / "residual_motion.npz")["residual_xy_px"]
        local_rms = np.sqrt(np.nanmean(np.sum(np.square(residual), axis=2), axis=1))
        velocity = np.gradient(common_xy, times, axis=0)
        for row in rows:
            center = float(row["timestamp"])
            selection = np.abs(times - center) <= 1.0
            figure, axes = plt.subplots(5, 1, figsize=(9, 10), sharex=True)
            axes[0].plot(times[selection], common_xy[selection])
            axes[0].set_ylabel("disp px")
            axes[1].plot(times[selection], velocity[selection])
            axes[1].set_ylabel("vel px/s")
            axes[2].plot(times[selection], acceleration[selection])
            axes[2].set_ylabel("acc px/s2")
            axes[3].plot(times[selection], local_rms[selection])
            axes[3].set_ylabel("local RMS")
            quality_code = np.asarray(
                [{"INVALID": 0, "DEGRADED": 1, "GOOD": 2}.get(value, -1) for value in quality]
            )
            axes[4].step(times[selection], quality_code[selection], where="post")
            axes[4].set_ylabel("quality")
            axes[4].set_xlabel("timestamp s")
            for axis in axes:
                axis.axvline(center, color="red", linestyle="--", linewidth=1)
                axis.grid(alpha=0.2)
            figure.tight_layout()
            figure.savefig(output / f"reseed_{int(row['reseed_id'])}_window.png", dpi=160)
            plt.close(figure)
        plot_status = "GENERATED"
    ratios = np.asarray([float(row["acceleration_spike_ratio"]) for row in rows])
    summary = {
        "reseed_events_in_log": len(events),
        "reseed_events_analyzed": len(rows),
        "plot_status": plot_status,
        "acceleration_spike_ratio_p50": (
            float(np.percentile(ratios, 50)) if ratios.size else None
        ),
        "acceleration_spike_ratio_p95": (
            float(np.percentile(ratios, 95)) if ratios.size else None
        ),
        "mask_recommendation": (
            "REVIEW_REQUIRED_IF_RATIO_P95_EXCEEDS_3"
            if ratios.size
            else "NOT_EVALUABLE_NO_RESEEDS"
        ),
    }
    (output / "reseed_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return rows
