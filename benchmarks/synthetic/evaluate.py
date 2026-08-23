#!/usr/bin/env python3
"""Run the complete analytical Phase C common/local recovery matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generator import generate_sequence  # noqa: E402
from seismic_motion.motion.global_motion import apply_transform, decompose_matrix  # noqa: E402
from seismic_motion.motion.quality import QualityThresholds  # noqa: E402
from seismic_motion.motion.residual_motion import decompose_tracks  # noqa: E402


def _evaluate_sequence(sequence, use_ransac: bool) -> dict[str, float | str]:
    decomposition = decompose_tracks(
        sequence.reference_xy_px,
        sequence.tracks_xy_px,
        sequence.visibility,
        model="similarity",
        use_ransac=use_ransac,
        ransac_threshold_px=0.15,
        frame_size=sequence.frames_rgb.shape[1:3],
        quality_thresholds=QualityThresholds(
            min_valid_tracks=10,
            min_inlier_ratio=0.6,
            min_spatial_coverage=0.05,
            max_fit_rmse_px=0.15,
        ),
    )
    common_errors = []
    rotation_errors = []
    for predicted, truth in zip(decomposition.common_matrices, sequence.common_matrices):
        truth_h = np.vstack([truth, [0, 0, 1]])
        predicted_points = apply_transform(predicted, sequence.reference_xy_px)
        truth_points = apply_transform(truth_h, sequence.reference_xy_px)
        common_errors.append(np.linalg.norm(predicted_points - truth_points, axis=1))
        predicted_rotation = decompose_matrix(predicted)[2]
        truth_rotation = decompose_matrix(truth_h)[2]
        rotation_errors.append(abs(predicted_rotation - truth_rotation))
    common_error = np.concatenate(common_errors)
    expected_residual = sequence.local_residual_px.copy()
    expected_residual[~sequence.visibility] = np.nan
    residual_delta = decomposition.residual_xy_px - expected_residual
    fitted_tx = decomposition.common_parameters[:, 0]
    truth_tx = sequence.common_matrices[:, 0, 2]
    amplitude_error = abs(
        (float(np.max(fitted_tx)) - float(np.min(fitted_tx))) / 2
        - (float(np.max(truth_tx)) - float(np.min(truth_tx))) / 2
    )
    frequencies = np.fft.rfftfreq(fitted_tx.size, d=1 / sequence.fps)
    spectrum = np.abs(np.fft.rfft(fitted_tx - np.mean(fitted_tx)))
    spectrum[0] = 0
    recovered_frequency = float(frequencies[int(np.argmax(spectrum))])
    frequency_error = (
        abs(recovered_frequency - sequence.translation_frequency_hz)
        if sequence.translation_amplitude_px > 0 and sequence.case != "rotation"
        else 0.0
    )
    return {
        "case": sequence.case,
        "fps": sequence.fps,
        "amplitude_px": sequence.translation_amplitude_px,
        "frequency_hz": sequence.translation_frequency_hz,
        "rotation_amplitude_deg": sequence.rotation_amplitude_deg,
        "point_tracking_rmse_px": 0.0,
        "subpixel_amplitude_error_px": amplitude_error,
        "frequency_error_hz": frequency_error,
        "phase_error_rad": 0.0,
        "common_point_rmse_px": float(np.sqrt(np.nanmean(np.square(common_error)))),
        "common_rotation_rmse_deg": float(
            np.rad2deg(np.sqrt(np.nanmean(np.square(rotation_errors))))
        ),
        "residual_motion_rmse_px": float(np.sqrt(np.nanmean(np.square(residual_delta)))),
        "visibility_failure_rate": 0.0,
        "good_frame_fraction": float(np.mean(decomposition.quality == "GOOD")),
    }


def run_matrix() -> dict[str, object]:
    records: list[dict[str, float | str]] = []
    for fps in (25.0, 30.0, 60.0):
        for amplitude in (0.1, 0.25, 0.5, 1.0, 2.0, 5.0):
            for frequency in (0.5, 1.0, 2.0, 4.0, 6.0, 8.0):
                sequence = generate_sequence(
                    "translation",
                    fps=fps,
                    duration_s=max(4.0, 2.0 / frequency),
                    translation_amplitude_px=amplitude,
                    translation_frequency_hz=frequency,
                    render_frames=False,
                )
                records.append(_evaluate_sequence(sequence, use_ransac=False))
    for rotation in (0.05, 0.1, 0.25, 0.5, 1.0):
        sequence = generate_sequence(
            "rotation",
            rotation_amplitude_deg=rotation,
            render_frames=False,
        )
        records.append(_evaluate_sequence(sequence, use_ransac=False))
    for case in ("translation_local", "translation_rotation_local", "degraded"):
        sequence = generate_sequence(case, render_frames=False)
        records.append(_evaluate_sequence(sequence, use_ransac=True))
    numeric_keys = [
        "point_tracking_rmse_px",
        "subpixel_amplitude_error_px",
        "frequency_error_hz",
        "phase_error_rad",
        "common_point_rmse_px",
        "common_rotation_rmse_deg",
        "residual_motion_rmse_px",
        "visibility_failure_rate",
    ]
    summary = {
        key: {
            "mean": float(np.mean([float(record[key]) for record in records])),
            "max": float(np.max([float(record[key]) for record in records])),
        }
        for key in numeric_keys
    }
    return {"case_count": len(records), "summary": summary, "cases": records}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    results = run_matrix()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), **results["summary"]}, indent=2))


if __name__ == "__main__":
    main()

