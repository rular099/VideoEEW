#!/usr/bin/env python3
"""Run a coverage-designed real-CoTracker weak-motion stress benchmark."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import yaml

from benchmarks.synthetic.generator import SyntheticSequence, generate_sequence
from seismic_motion.motion.global_motion import apply_transform, decompose_matrix
from seismic_motion.motion.quality import QualityThresholds
from seismic_motion.motion.residual_motion import decompose_tracks
from seismic_motion.signal.online import OnlineSignalProcessor
from seismic_motion.tracking.cotracker_adapter import CoTrackerAdapter, CoTrackerAdapterConfig
from seismic_motion.tracking.types import concatenate_track_batches


def _sinusoid(timestamps: np.ndarray, values: np.ndarray, frequency: float) -> tuple[float, float]:
    angular = 2 * np.pi * frequency * timestamps
    design = np.column_stack([np.sin(angular), np.cos(angular), np.ones_like(angular)])
    parameters = np.linalg.lstsq(design, values, rcond=None)[0]
    return float(np.hypot(parameters[0], parameters[1])), float(
        np.arctan2(parameters[1], parameters[0])
    )


def _angle_delta(first: float, second: float) -> float:
    return float(abs(np.arctan2(np.sin(first - second), np.cos(first - second))))


def _track_sequence(
    adapter: CoTrackerAdapter, sequence: SyntheticSequence
) -> tuple[object, np.ndarray]:
    frames = sequence.frames_rgb
    timestamps = sequence.timestamps
    config = adapter.config
    batches = []
    block_times = []
    last_start: int | None = None
    for start in range(0, frames.shape[0] - config.window_len + 1, config.step):
        stop = start + config.window_len
        begin = time.perf_counter()
        batches.append(
            adapter.process_window(
                frames[start:stop], timestamps[start:stop], np.arange(start, stop)
            )
        )
        block_times.append((time.perf_counter() - begin) * 1000)
        last_start = start
    if last_start is None:
        raise RuntimeError("stress sequence is shorter than one tracker window")
    last_end = last_start + config.window_len
    if frames.shape[0] > last_end:
        next_start = last_start + config.step
        tail_frames = frames[next_start:]
        tail_times = timestamps[next_start:]
        valid_frames = tail_frames.shape[0]
        pad = config.window_len - valid_frames
        interval = 1 / sequence.fps
        padded_frames = np.concatenate(
            [tail_frames, np.repeat(tail_frames[-1:], pad, axis=0)], axis=0
        )
        padded_times = np.concatenate(
            [tail_times, tail_times[-1] + interval * np.arange(1, pad + 1)]
        )
        begin = time.perf_counter()
        batches.append(
            adapter.process_window(
                padded_frames,
                padded_times,
                np.arange(next_start, next_start + config.window_len),
                final=True,
                valid_frames=valid_frames,
            )
        )
        block_times.append((time.perf_counter() - begin) * 1000)
    else:
        pending = adapter.flush_pending()
        if pending is not None:
            batches.append(pending)
    return concatenate_track_batches(batches), np.asarray(block_times, dtype=np.float64)


def _causal_acceleration(
    timestamps: np.ndarray, common_xy: np.ndarray, fps: float, points: int
) -> np.ndarray:
    processor = OnlineSignalProcessor(
        sample_rate_hz=fps,
        derivative_method="causal_polynomial",
        window_length=9,
        polynomial_order=3,
    )
    rows = []
    for timestamp, common in zip(timestamps, common_xy):
        state = processor.update(timestamp, common, np.zeros((points, 2)), "GOOD")
        rows.append([state.common_acceleration_x, state.common_acceleration_y])
    return np.asarray(rows, dtype=np.float64)


def evaluate_case(
    case_id: str,
    sequence: SyntheticSequence,
    adapter: CoTrackerAdapter,
    block_times: np.ndarray,
    predicted: object,
) -> dict[str, object]:
    indices = predicted.frame_indices
    truth_tracks = sequence.tracks_xy_px[indices]
    truth_visibility = sequence.visibility[indices]
    error = np.linalg.norm(predicted.xy_px - truth_tracks, axis=2)
    valid_error = np.isfinite(error) & truth_visibility
    decomposition = decompose_tracks(
        sequence.reference_xy_px,
        predicted.xy_px,
        predicted.visible,
        model="similarity",
        ransac_threshold_px=1.5,
        frame_size=sequence.frames_rgb.shape[1:3],
        quality_thresholds=QualityThresholds(
            min_valid_tracks=10,
            min_inlier_ratio=0.6,
            min_spatial_coverage=0.05,
            max_fit_rmse_px=2.0,
        ),
    )
    truth_matrices = np.asarray(
        [np.vstack([matrix, [0, 0, 1]]) for matrix in sequence.common_matrices[indices]]
    )
    common_errors = []
    truth_rotation = []
    for predicted_matrix, truth_matrix in zip(
        decomposition.common_matrices, truth_matrices
    ):
        predicted_points = apply_transform(predicted_matrix, sequence.reference_xy_px)
        truth_points = apply_transform(truth_matrix, sequence.reference_xy_px)
        common_errors.append(np.linalg.norm(predicted_points - truth_points, axis=1))
        truth_rotation.append(decompose_matrix(truth_matrix)[2])
    common_error = np.asarray(common_errors)
    predicted_rotation = decomposition.common_parameters[:, 2]
    rotation_delta = np.arctan2(
        np.sin(predicted_rotation - truth_rotation),
        np.cos(predicted_rotation - truth_rotation),
    )
    local_truth = sequence.local_residual_px[indices].astype(np.float64)
    local_truth[~truth_visibility] = np.nan
    local_delta = decomposition.residual_xy_px - local_truth
    predicted_local_peak = float(
        np.nanmax(np.linalg.norm(decomposition.residual_xy_px, axis=2))
    )
    truth_local_peak = float(np.nanmax(np.linalg.norm(local_truth, axis=2)))

    includes_translation = sequence.case not in {"rotation", "rotation_local"}
    predicted_amplitude = predicted_phase = truth_amplitude = truth_phase = float("nan")
    if includes_translation:
        predicted_amplitude, predicted_phase = _sinusoid(
            sequence.timestamps[indices],
            decomposition.common_parameters[:, 0],
            sequence.translation_frequency_hz,
        )
        truth_amplitude, truth_phase = _sinusoid(
            sequence.timestamps[indices],
            truth_matrices[:, 0, 2],
            sequence.translation_frequency_hz,
        )
    predicted_acceleration = _causal_acceleration(
        sequence.timestamps[indices],
        decomposition.common_parameters[:, :2],
        sequence.fps,
        sequence.reference_xy_px.shape[0],
    )
    truth_acceleration = _causal_acceleration(
        sequence.timestamps[indices],
        truth_matrices[:, :2, 2],
        sequence.fps,
        sequence.reference_xy_px.shape[0],
    )
    finite_acceleration = np.isfinite(predicted_acceleration).all(axis=1) & np.isfinite(
        truth_acceleration
    ).all(axis=1)
    predicted_acc_mag = np.linalg.norm(predicted_acceleration[finite_acceleration], axis=1)
    truth_acc_mag = np.linalg.norm(truth_acceleration[finite_acceleration], axis=1)
    finite_times = sequence.timestamps[indices][finite_acceleration]
    predicted_peak_index = int(np.argmax(predicted_acc_mag))
    truth_peak_index = int(np.argmax(truth_acc_mag))
    warm = block_times[1:] if block_times.size > 1 else block_times
    return {
        "case_id": case_id,
        "scene": sequence.case,
        "fps": sequence.fps,
        "duration_s": float(sequence.timestamps[-1] + 1 / sequence.fps),
        "translation_amplitude_px": sequence.translation_amplitude_px,
        "frequency_hz": sequence.translation_frequency_hz,
        "rotation_amplitude_deg": sequence.rotation_amplitude_deg,
        "frames": int(indices.size),
        "points": int(sequence.reference_xy_px.shape[0]),
        "point_rmse_px": float(np.sqrt(np.mean(np.square(error[valid_error])))),
        "point_mean_error_px": float(np.mean(error[valid_error])),
        "point_p95_error_px": float(np.percentile(error[valid_error], 95)),
        "visibility_error": float(np.mean(predicted.visible != truth_visibility)),
        "track_survival_ratio": float(np.mean(predicted.visible[truth_visibility])),
        "translation_amplitude_error_px": (
            abs(predicted_amplitude - truth_amplitude) if includes_translation else float("nan")
        ),
        "translation_phase_error_rad": (
            _angle_delta(predicted_phase, truth_phase) if includes_translation else float("nan")
        ),
        "rotation_rmse_deg": float(
            np.degrees(np.sqrt(np.nanmean(np.square(rotation_delta))))
        ),
        "common_point_rmse_px": float(np.sqrt(np.nanmean(np.square(common_error)))),
        "local_amplitude_error_px": abs(predicted_local_peak - truth_local_peak),
        "local_residual_rmse_px": float(np.sqrt(np.nanmean(np.square(local_delta)))),
        "acceleration_rmse_px_s2": float(
            np.sqrt(np.mean(np.square(predicted_acceleration[finite_acceleration] - truth_acceleration[finite_acceleration])))
        ),
        "peak_acceleration_error_px_s2": abs(
            float(predicted_acc_mag[predicted_peak_index])
            - float(truth_acc_mag[truth_peak_index])
        ),
        "peak_timing_error_s": abs(
            float(finite_times[predicted_peak_index] - finite_times[truth_peak_index])
        ),
        "motion_good_fraction": float(np.mean(decomposition.quality == "GOOD")),
        "tracker_mean_ms": float(np.mean(warm)),
        "tracker_p95_ms": float(np.percentile(warm, 95)),
    }


def coverage_cases(spec: dict[str, object]) -> list[dict[str, object]]:
    translation = spec["translation"]
    rotation = spec["rotation"]
    cases: list[dict[str, object]] = []
    for fps in translation["fps_values"]:
        for amplitude in translation["amplitudes_px"]:
            for frequency in translation["frequencies_hz"]:
                cases.append(
                    {
                        "scene": "translation",
                        "fps": float(fps),
                        "translation_amplitude_px": float(amplitude),
                        "translation_frequency_hz": float(frequency),
                        "rotation_amplitude_deg": 0.0,
                    }
                )
    for fps in translation["fps_values"]:
        for amplitude in rotation["amplitudes_deg"]:
            cases.append(
                {
                    "scene": "rotation",
                    "fps": float(fps),
                    "translation_amplitude_px": 0.0,
                    "translation_frequency_hz": 1.0 / 0.65,
                    "rotation_amplitude_deg": float(amplitude),
                }
            )
    for fps in translation["fps_values"]:
        for scene in spec["cases"]:
            if scene in {"translation", "rotation"}:
                continue
            cases.append(
                {
                    "scene": str(scene),
                    "fps": float(fps),
                    "translation_amplitude_px": 0.25,
                    "translation_frequency_hz": 3.0,
                    "rotation_amplitude_deg": 0.1,
                }
            )
    return cases


def _write_csv(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    if not rows:
        path.write_text("NOT_MEASURED\n", encoding="utf-8")
        return
    selected = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=selected, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default="benchmarks/synthetic/spec.yaml")
    parser.add_argument("--cotracker-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    spec = yaml.safe_load(Path(args.spec).read_text(encoding="utf-8"))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    point_grid = tuple(int(value) for value in spec["point_grid"])
    adapter = CoTrackerAdapter(
        CoTrackerAdapterConfig(
            cotracker_root=args.cotracker_root,
            checkpoint=args.checkpoint,
            device=args.device,
            num_points=point_grid[0] * point_grid[1],
            max_blocks_before_reseed=64,
        ),
        manual_points=np.zeros((point_grid[0] * point_grid[1], 2), dtype=np.float32),
    )
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    cases = coverage_cases(spec)
    for index, case in enumerate(cases):
        case_id = f"stress-{index:04d}"
        duration = max(float(spec["duration_s"]), 2.0 / float(case["translation_frequency_hz"]))
        try:
            sequence = generate_sequence(
                str(case["scene"]),
                fps=float(case["fps"]),
                duration_s=duration,
                image_size=tuple(int(value) for value in spec["image_size"]),
                point_grid=point_grid,
                translation_amplitude_px=float(case["translation_amplitude_px"]),
                translation_frequency_hz=float(case["translation_frequency_hz"]),
                rotation_amplitude_deg=float(case["rotation_amplitude_deg"]),
                local_fraction=float(spec["local_motion"]["point_fraction"]),
                local_amplitude_px=float(spec["local_motion"]["amplitude_px"]),
                local_frequency_hz=float(spec["local_motion"]["frequency_hz"]),
                local_phase_rad=float(spec["local_motion"]["phase_rad"]),
                seed=int(spec["seed"]) + index,
            )
            adapter.manual_points = sequence.reference_xy_px
            if index:
                adapter.reset_stream(0, reason="stress_case_boundary")
            predicted, block_times = _track_sequence(adapter, sequence)
            rows.append(evaluate_case(case_id, sequence, adapter, block_times, predicted))
        except Exception as exc:
            failures.append(
                {
                    "case_id": case_id,
                    **case,
                    "exception_type": type(exc).__name__,
                    "reason": str(exc),
                }
            )
        _write_csv(output / "tracking_stress_summary.csv", rows)
        _write_csv(output / "stress_failures.csv", failures)
        print(f"stress progress {index + 1}/{len(cases)} pass={len(rows)} fail={len(failures)}", flush=True)
    _write_csv(
        output / "common_motion_metrics.csv",
        rows,
        [
            "case_id", "scene", "fps", "translation_amplitude_px", "frequency_hz",
            "translation_amplitude_error_px", "translation_phase_error_rad", "common_point_rmse_px",
        ],
    )
    _write_csv(
        output / "local_motion_metrics.csv",
        rows,
        ["case_id", "scene", "local_amplitude_error_px", "local_residual_rmse_px"],
    )
    _write_csv(
        output / "rotation_metrics.csv",
        rows,
        ["case_id", "scene", "fps", "rotation_amplitude_deg", "rotation_rmse_deg"],
    )
    manifest = {
        "schema_version": 2,
        "matrix_policy": spec["matrix_policy"],
        "planned_cases": len(cases),
        "completed_cases": len(rows),
        "failed_cases": len(failures),
        "oracle_tracks_used": False,
        "tracker": "CoTracker3 online adapter",
    }
    (output / "stress_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
