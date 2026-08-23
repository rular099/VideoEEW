#!/usr/bin/env python3
"""Run CoTracker on a rendered synthetic sequence and compare exact tracks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from seismic_motion.motion.quality import QualityThresholds
from seismic_motion.motion.residual_motion import decompose_tracks
from seismic_motion.tracking.cotracker_adapter import CoTrackerAdapter, CoTrackerAdapterConfig
from seismic_motion.tracking.types import concatenate_track_batches


def _sinusoid_amplitude_phase(
    timestamps: np.ndarray, values: np.ndarray, frequency_hz: float
) -> tuple[float, float]:
    angular = 2 * np.pi * frequency_hz * timestamps
    design = np.stack([np.sin(angular), np.cos(angular), np.ones_like(angular)], axis=1)
    parameters, *_ = np.linalg.lstsq(design, values, rcond=None)
    return float(np.hypot(parameters[0], parameters[1])), float(
        np.arctan2(parameters[1], parameters[0])
    )


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    data = np.load(args.input)
    frames = data["frames_rgb"]
    timestamps = data["timestamps"]
    truth_tracks = data["tracks_xy_px"]
    truth_visibility = data["visibility"]
    reference = data["reference_xy_px"]
    local_truth = data["local_residual_px"]
    common_truth = data["common_matrices"]
    frequency_hz = float(data["translation_frequency_hz"])
    config = CoTrackerAdapterConfig(
        cotracker_root=args.cotracker_root,
        checkpoint=args.checkpoint,
        device=args.device,
        num_points=reference.shape[0],
        max_blocks_before_reseed=args.max_blocks_before_reseed,
    )
    events: list[dict[str, object]] = []
    adapter = CoTrackerAdapter(config, manual_points=reference, event_sink=events.append)
    batches = []
    block_times_ms = []
    for start in range(0, frames.shape[0] - config.window_len + 1, config.step):
        stop = start + config.window_len
        begin = time.perf_counter()
        batches.append(
            adapter.process_window(
                frames[start:stop], timestamps[start:stop], np.arange(start, stop)
            )
        )
        block_times_ms.append((time.perf_counter() - begin) * 1000)
    pending = adapter.flush_pending()
    if pending is not None:
        batches.append(pending)
    predicted = concatenate_track_batches(batches)
    indices = predicted.frame_indices
    truth = truth_tracks[indices]
    truth_vis = truth_visibility[indices]
    error = np.linalg.norm(predicted.xy_px - truth, axis=2)
    valid_error = np.isfinite(error) & truth_vis
    visibility_disagreement = predicted.visible != truth_vis
    predicted_motion = decompose_tracks(
        reference,
        predicted.xy_px,
        predicted.visible,
        model="similarity",
        ransac_threshold_px=args.ransac_threshold_px,
        frame_size=frames.shape[1:3],
        quality_thresholds=QualityThresholds(
            min_spatial_coverage=0.05, max_fit_rmse_px=max(2.0, args.ransac_threshold_px)
        ),
    )
    predicted_tx = predicted_motion.common_parameters[:, 0]
    truth_tx = common_truth[indices, 0, 2]
    predicted_amplitude, predicted_phase = _sinusoid_amplitude_phase(
        timestamps[indices], predicted_tx, frequency_hz
    )
    truth_amplitude, truth_phase = _sinusoid_amplitude_phase(
        timestamps[indices], truth_tx, frequency_hz
    )
    residual_delta = predicted_motion.residual_xy_px - local_truth[indices]
    finite_residual = np.isfinite(residual_delta)
    warm_times = np.asarray(block_times_ms[1:] or block_times_ms)
    return {
        "input": str(Path(args.input).resolve()),
        "frames_evaluated": int(indices.size),
        "points": int(reference.shape[0]),
        "tracking_rmse_px": float(np.sqrt(np.mean(np.square(error[valid_error])))),
        "tracking_mean_error_px": float(np.mean(error[valid_error])),
        "tracking_p95_error_px": float(np.percentile(error[valid_error], 95)),
        "tracking_max_error_px": float(np.max(error[valid_error])),
        "visibility_disagreement_fraction": float(np.mean(visibility_disagreement)),
        "common_amplitude_error_px": abs(predicted_amplitude - truth_amplitude),
        "common_phase_error_rad": abs(predicted_phase - truth_phase),
        "local_residual_rmse_px": float(
            np.sqrt(np.mean(np.square(residual_delta[finite_residual])))
        ),
        "motion_good_fraction": float(np.mean(predicted_motion.quality == "GOOD")),
        "block_time_ms": {
            "first_with_model_load": float(block_times_ms[0]),
            "mean_warm": float(np.mean(warm_times)),
            "p95_warm": float(np.percentile(warm_times, 95)),
            "max_warm": float(np.max(warm_times)),
        },
        "reseed_events": events,
        "upstream_history_bound_frames": adapter.upstream_history_bound_frames,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--cotracker-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-blocks-before-reseed", type=int, default=64)
    parser.add_argument("--ransac-threshold-px", type=float, default=1.5)
    parser.add_argument("--output")
    args = parser.parse_args()
    metrics = evaluate(args)
    encoded = json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()

