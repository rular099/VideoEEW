#!/usr/bin/env python3
"""Compare deployable causal signal states with the zero-phase analysis upper bound."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import signal

from seismic_motion.signal.derivatives import local_polynomial_derivative
from seismic_motion.signal.filtering import bandpass_filter
from seismic_motion.signal.online import OnlineSignalProcessor


def _comparison(reference: np.ndarray, candidate: np.ndarray, timestamps: np.ndarray) -> dict[str, float]:
    finite = np.isfinite(reference) & np.isfinite(candidate)
    ref = np.asarray(reference[finite], dtype=np.float64)
    cand = np.asarray(candidate[finite], dtype=np.float64)
    times = np.asarray(timestamps[finite], dtype=np.float64)
    if ref.size < 10:
        raise ValueError("not enough finite samples for causal comparison")
    ref_centered = ref - np.mean(ref)
    cand_centered = cand - np.mean(cand)
    correlation = signal.correlate(cand_centered, ref_centered, mode="full")
    lags = signal.correlation_lags(cand_centered.size, ref_centered.size, mode="full")
    lag_samples = int(lags[int(np.argmax(correlation))])
    interval = float(np.median(np.diff(times)))
    ref_peak_index = int(np.argmax(np.abs(ref)))
    cand_peak_index = int(np.argmax(np.abs(cand)))
    ref_peak = float(np.abs(ref[ref_peak_index]))
    cand_peak = float(np.abs(cand[cand_peak_index]))
    return {
        "amplitude_bias_fraction": (cand_peak - ref_peak) / max(ref_peak, 1e-12),
        "rmse": float(np.sqrt(np.mean(np.square(cand - ref)))),
        "phase_lag_samples": float(lag_samples),
        "phase_lag_s": lag_samples * interval,
        "peak_timing_error_s": float(times[cand_peak_index] - times[ref_peak_index]),
        "peak_amplitude_error": cand_peak - ref_peak,
        "reference_peak": ref_peak,
        "causal_peak": cand_peak,
    }


def evaluate(
    *,
    fps: float = 30.0,
    duration_s: float = 20.0,
    bandpass_hz: tuple[float, float] = (0.3, 8.0),
) -> tuple[list[dict[str, float | str]], list[dict[str, float]]]:
    timestamps = np.arange(int(round(fps * duration_s)), dtype=np.float64) / fps
    envelope = np.exp(-0.5 * np.square((timestamps - duration_s * 0.55) / 2.0))
    displacement = envelope * (
        0.8 * np.sin(2 * np.pi * 1.2 * timestamps)
        + 0.25 * np.sin(2 * np.pi * 3.5 * timestamps + 0.4)
    )
    common = np.column_stack([displacement, 0.4 * displacement])
    offline_filtered = bandpass_filter(
        common, fps, bandpass_hz, causal=False, apply_detrend=True
    ).values
    offline_acceleration = local_polynomial_derivative(
        timestamps,
        offline_filtered,
        derivative_order=2,
        polynomial_order=3,
        window_length=9,
        causal=False,
    )
    rows: list[dict[str, float | str]] = []
    timeseries: list[dict[str, float]] = []
    for method in ("backward_finite_difference", "causal_polynomial"):
        processor = OnlineSignalProcessor(
            sample_rate_hz=fps,
            bandpass_hz=bandpass_hz,
            derivative_method=method,
            window_length=9,
            polynomial_order=3,
        )
        states = [
            processor.update(timestamp, value, np.zeros((4, 2)), "GOOD")
            for timestamp, value in zip(timestamps, common)
        ]
        causal_filtered = np.asarray([state.filtered_common_x for state in states])
        causal_acceleration = np.asarray([state.common_acceleration_x for state in states])
        warmup = timestamps >= 2.0
        for domain, reference, candidate in (
            ("filtered_displacement_x", offline_filtered[:, 0], causal_filtered),
            ("acceleration_x", offline_acceleration[:, 0], causal_acceleration),
        ):
            metrics = _comparison(reference[warmup], candidate[warmup], timestamps[warmup])
            rows.append(
                {
                    "derivative_method": method,
                    "domain": domain,
                    "offline_mode": "offline_zero_phase",
                    "causal_mode": "causal_sos",
                    "window_length": 9,
                    "polynomial_order": 3,
                    "effective_lookahead_samples": 0,
                    "startup_behavior": states[0].startup_state,
                    **metrics,
                }
            )
        for index, timestamp in enumerate(timestamps):
            timeseries.append(
                {
                    "timestamp": float(timestamp),
                    "offline_filtered_x": float(offline_filtered[index, 0]),
                    "causal_filtered_x": float(causal_filtered[index]),
                    "offline_acceleration_x": float(offline_acceleration[index, 0]),
                    "causal_acceleration_x": float(causal_acceleration[index]),
                    "method_code": 0.0 if method == "backward_finite_difference" else 1.0,
                }
            )
    return rows, timeseries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--duration-s", type=float, default=20.0)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows, timeseries = evaluate(fps=args.fps, duration_s=args.duration_s)
    with (output / "offline_vs_causal_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "offline_vs_causal_timeseries.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(timeseries[0]))
        writer.writeheader()
        writer.writerows(timeseries)
    (output / "offline_vs_causal_metrics.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
