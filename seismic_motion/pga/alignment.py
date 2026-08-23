"""Estimate small unknown video/sensor offsets after visual acceleration exists."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import integrate, signal

from seismic_motion.signal.timestamps import validate_timestamps


@dataclass(frozen=True)
class AlignmentResult:
    offset_s: float
    correlation: float
    visual_channel: int
    sensor_channel: int
    polarity: int
    overlap_samples: int
    status: str


def _standardize(values: np.ndarray) -> np.ndarray:
    centered = values - np.mean(values)
    scale = np.std(centered)
    return centered / scale if scale > np.finfo(float).eps else np.zeros_like(centered)


def estimate_time_offset(
    visual_timestamps: np.ndarray,
    visual_acceleration: np.ndarray,
    sensor_timestamps: np.ndarray,
    sensor_acceleration: np.ndarray,
    *,
    max_offset_s: float = 2.0,
    offset_range_s: tuple[float, float] | None = None,
    step_s: float | None = None,
    min_correlation: float = 0.25,
) -> AlignmentResult:
    """Search sensor(t + offset) against visual(t), including axis/sign ambiguity."""

    visual_times = validate_timestamps(visual_timestamps)
    sensor_times = validate_timestamps(sensor_timestamps)
    visual = np.asarray(visual_acceleration, dtype=np.float64)
    sensor = np.asarray(sensor_acceleration, dtype=np.float64)
    if visual.ndim == 1:
        visual = visual[:, None]
    if sensor.ndim == 1:
        sensor = sensor[:, None]
    if visual.shape[0] != visual_times.size or sensor.shape[0] != sensor_times.size:
        raise ValueError("signal lengths must match timestamps")
    if step_s is None:
        step_s = min(np.median(np.diff(visual_times)), np.median(np.diff(sensor_times)))
    if step_s <= 0 or max_offset_s < 0:
        raise ValueError("step_s must be positive and max_offset_s non-negative")
    if offset_range_s is None:
        offset_min, offset_max = -max_offset_s, max_offset_s
    else:
        offset_min, offset_max = (float(value) for value in offset_range_s)
        if offset_max < offset_min:
            raise ValueError("offset_range_s must be increasing")
    offsets = np.arange(offset_min, offset_max + step_s / 2, step_s)
    best = AlignmentResult(0.0, 0.0, 0, 0, 1, 0, "LOW_CORRELATION")
    for offset in offsets:
        sensor_query_times = visual_times + offset
        overlap = (sensor_query_times >= sensor_times[0]) & (sensor_query_times <= sensor_times[-1])
        if int(overlap.sum()) < max(10, int(0.25 * visual_times.size)):
            continue
        for visual_channel in range(visual.shape[1]):
            visual_values = visual[overlap, visual_channel]
            finite_visual = np.isfinite(visual_values)
            for sensor_channel in range(sensor.shape[1]):
                interpolated = np.interp(
                    sensor_query_times[overlap], sensor_times, sensor[:, sensor_channel]
                )
                valid = finite_visual & np.isfinite(interpolated)
                if int(valid.sum()) < 10:
                    continue
                correlation = float(
                    np.mean(
                        _standardize(visual_values[valid])
                        * _standardize(interpolated[valid])
                    )
                )
                if abs(correlation) > abs(best.correlation):
                    best = AlignmentResult(
                        offset_s=float(offset),
                        correlation=abs(correlation),
                        visual_channel=visual_channel,
                        sensor_channel=sensor_channel,
                        polarity=1 if correlation >= 0 else -1,
                        overlap_samples=int(valid.sum()),
                        status="ALIGNED" if abs(correlation) >= min_correlation else "LOW_CORRELATION",
                    )
    return best


def full_containment_offset_range(
    visual_timestamps: np.ndarray, sensor_timestamps: np.ndarray
) -> tuple[float, float]:
    """Offsets for which the complete visual duration lies in the sensor record."""

    visual = validate_timestamps(visual_timestamps)
    sensor = validate_timestamps(sensor_timestamps)
    minimum = float(sensor[0] - visual[0])
    maximum = float(sensor[-1] - visual[-1])
    if maximum < minimum:
        raise ValueError("sensor record is shorter than the visual record")
    return minimum, maximum


def acceleration_to_displacement(
    timestamps: np.ndarray,
    acceleration: np.ndarray,
    *,
    bandpass_hz: tuple[float, float] = (0.3, 8.0),
) -> np.ndarray:
    """Bandpass, double integrate and bandpass again for alignment only."""

    times = validate_timestamps(timestamps)
    values = np.asarray(acceleration, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.shape[0] != times.size:
        raise ValueError("acceleration first dimension must match timestamps")
    sample_rate = 1 / np.median(np.diff(times))
    low, high = bandpass_hz
    high = min(float(high), 0.95 * sample_rate / 2)
    sos = signal.butter(4, [float(low), high], btype="bandpass", fs=sample_rate, output="sos")
    output = np.empty_like(values)
    for channel in range(values.shape[1]):
        filtered_acceleration = signal.sosfiltfilt(
            sos, signal.detrend(values[:, channel])
        )
        velocity = integrate.cumulative_trapezoid(
            filtered_acceleration, times, initial=0.0
        )
        velocity = signal.detrend(velocity)
        displacement = integrate.cumulative_trapezoid(velocity, times, initial=0.0)
        output[:, channel] = signal.sosfiltfilt(sos, signal.detrend(displacement))
    return output
