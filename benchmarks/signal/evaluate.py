#!/usr/bin/env python3
"""Quantify noise, peak bias and lag for derivative/filter candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from seismic_motion.signal.derivatives import (  # noqa: E402
    finite_difference,
    local_polynomial_derivative,
)
from seismic_motion.signal.filtering import bandpass_filter  # noqa: E402


def _lag_samples(reference: np.ndarray, estimate: np.ndarray) -> int:
    ref = reference - np.mean(reference)
    est = estimate - np.mean(estimate)
    correlation = np.correlate(est, ref, mode="full")
    return int(np.argmax(correlation) - (reference.size - 1))


def run() -> dict[str, object]:
    rng = np.random.default_rng(17)
    methods: dict[str, list[dict[str, float]]] = {
        "raw_second_difference": [],
        "offline_bandpass_finite_difference": [],
        "offline_bandpass_local_polynomial": [],
        "causal_bandpass_local_polynomial": [],
    }
    for fps in (25.0, 30.0, 60.0):
        for amplitude in (0.1, 0.25, 0.5):
            for frequency in (0.5, 2.0, 6.0):
                count = int(fps * 8)
                intervals = 1 / fps + rng.normal(0, 0.00035, count)
                timestamps = np.cumsum(intervals)
                clean = amplitude * np.sin(2 * np.pi * frequency * timestamps)
                noisy = clean + rng.normal(0, 0.02, count)
                truth = -(2 * np.pi * frequency) ** 2 * clean
                uniform_times = np.arange(count) / fps
                uniform_noisy = np.interp(uniform_times, timestamps - timestamps[0], noisy)
                uniform_truth = -(2 * np.pi * frequency) ** 2 * (
                    amplitude * np.sin(2 * np.pi * frequency * (uniform_times + timestamps[0]))
                )
                candidates = {
                    "raw_second_difference": finite_difference(
                        uniform_times, uniform_noisy, derivative_order=2
                    ),
                    "offline_bandpass_finite_difference": finite_difference(
                        uniform_times,
                        bandpass_filter(
                            uniform_noisy, fps, (0.3, min(8.0, fps / 2 - 0.5)), causal=False
                        ).values,
                        derivative_order=2,
                    ),
                }
                filtered_offline = bandpass_filter(
                    uniform_noisy, fps, (0.3, min(8.0, fps / 2 - 0.5)), causal=False
                ).values
                filtered_causal = bandpass_filter(
                    uniform_noisy, fps, (0.3, min(8.0, fps / 2 - 0.5)), causal=True
                ).values
                candidates["offline_bandpass_local_polynomial"] = local_polynomial_derivative(
                    uniform_times,
                    filtered_offline,
                    derivative_order=2,
                    window_length=7,
                )
                candidates["causal_bandpass_local_polynomial"] = local_polynomial_derivative(
                    uniform_times,
                    filtered_causal,
                    derivative_order=2,
                    window_length=7,
                    causal=True,
                )
                trim = max(int(fps), 8)
                truth_eval = uniform_truth[trim:-trim]
                for name, estimate in candidates.items():
                    estimate_eval = estimate[trim:-trim]
                    methods[name].append(
                        {
                            "rmse_px_s2": float(
                                np.sqrt(np.mean(np.square(estimate_eval - truth_eval)))
                            ),
                            "peak_relative_bias": float(
                                (np.max(np.abs(estimate_eval)) - np.max(np.abs(truth_eval)))
                                / max(np.max(np.abs(truth_eval)), 1e-12)
                            ),
                            "lag_samples": float(_lag_samples(truth_eval, estimate_eval)),
                        }
                    )
    summary = {}
    for name, records in methods.items():
        summary[name] = {
            key: {
                "median": float(np.median([record[key] for record in records])),
                "p95_abs": float(np.percentile(np.abs([record[key] for record in records]), 95)),
            }
            for key in ("rmse_px_s2", "peak_relative_bias", "lag_samples")
        }
    return {
        "case_count_per_method": len(next(iter(methods.values()))),
        "noise_sigma_px": 0.02,
        "timestamp_jitter_sigma_s": 0.00035,
        "summary": summary,
        "cases": methods,
        "note": "Offline zero-phase results are evaluation-only and are never labelled realtime.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    results = run()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "summary": results["summary"]}, indent=2))


if __name__ == "__main__":
    main()

