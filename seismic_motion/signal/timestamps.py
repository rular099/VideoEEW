"""Timestamp validation and auditable uniform resampling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TimebaseDiagnostics:
    nominal_interval_s: float
    effective_fps: float
    max_jitter_s: float
    rms_jitter_s: float
    estimated_missing_frames: int
    resampled: bool


def validate_timestamps(timestamps: np.ndarray) -> np.ndarray:
    values = np.asarray(timestamps, dtype=np.float64)
    if values.ndim != 1 or values.shape[0] < 2:
        raise ValueError("at least two one-dimensional timestamps are required")
    if not np.isfinite(values).all() or np.any(np.diff(values) <= 0):
        raise ValueError("timestamps must be finite and strictly increasing")
    return values


def diagnose_timebase(timestamps: np.ndarray) -> TimebaseDiagnostics:
    values = validate_timestamps(timestamps)
    intervals = np.diff(values)
    nominal = float(np.median(intervals))
    jitter = intervals - nominal
    missing = int(np.sum(np.maximum(np.rint(intervals / nominal).astype(int) - 1, 0)))
    return TimebaseDiagnostics(
        nominal_interval_s=nominal,
        effective_fps=1.0 / nominal,
        max_jitter_s=float(np.max(np.abs(jitter))),
        rms_jitter_s=float(np.sqrt(np.mean(np.square(jitter)))),
        estimated_missing_frames=missing,
        resampled=False,
    )


def resample_uniform(
    timestamps: np.ndarray,
    values: np.ndarray,
    target_fps: float | None = None,
) -> tuple[np.ndarray, np.ndarray, TimebaseDiagnostics]:
    times = validate_timestamps(timestamps)
    samples = np.asarray(values, dtype=np.float64)
    if samples.shape[0] != times.shape[0]:
        raise ValueError("values first dimension must match timestamps")
    diagnostics = diagnose_timebase(times)
    fps = diagnostics.effective_fps if target_fps is None else float(target_fps)
    if fps <= 0:
        raise ValueError("target_fps must be positive")
    count = int(np.floor((times[-1] - times[0]) * fps)) + 1
    uniform_times = times[0] + np.arange(count, dtype=np.float64) / fps
    flat = samples.reshape(samples.shape[0], -1)
    interpolated = np.empty((count, flat.shape[1]), dtype=np.float64)
    for column in range(flat.shape[1]):
        finite = np.isfinite(flat[:, column])
        if finite.sum() < 2:
            interpolated[:, column] = np.nan
        else:
            interpolated[:, column] = np.interp(
                uniform_times, times[finite], flat[finite, column]
            )
    result_diagnostics = TimebaseDiagnostics(
        nominal_interval_s=1.0 / fps,
        effective_fps=fps,
        max_jitter_s=diagnostics.max_jitter_s,
        rms_jitter_s=diagnostics.rms_jitter_s,
        estimated_missing_frames=diagnostics.estimated_missing_frames,
        resampled=True,
    )
    return uniform_times, interpolated.reshape((count, *samples.shape[1:])), result_diagnostics

