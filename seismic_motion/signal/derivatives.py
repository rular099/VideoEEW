"""Timestamp-aware offline and strictly causal derivative estimators."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .timestamps import validate_timestamps


def finite_difference(
    timestamps: np.ndarray, values: np.ndarray, derivative_order: int = 1
) -> np.ndarray:
    times = validate_timestamps(timestamps)
    result = np.asarray(values, dtype=np.float64)
    if result.shape[0] != times.shape[0]:
        raise ValueError("values first dimension must match timestamps")
    if derivative_order not in {1, 2}:
        raise ValueError("derivative_order must be 1 or 2")
    for _ in range(derivative_order):
        result = np.gradient(result, times, axis=0, edge_order=2)
    return result


def backward_finite_difference(
    timestamps: np.ndarray,
    values: np.ndarray,
    derivative_order: int = 1,
) -> np.ndarray:
    """Differentiate using only the current and earlier samples.

    Unlike :func:`finite_difference`, this function never calls ``np.gradient``
    and therefore never uses a sample to the right of the output timestamp.
    The first derivative needs two samples and the second derivative needs
    three; unavailable startup outputs are represented by NaN.
    """

    times = validate_timestamps(timestamps)
    samples = np.asarray(values, dtype=np.float64)
    if samples.shape[0] != times.shape[0]:
        raise ValueError("values first dimension must match timestamps")
    if derivative_order not in {1, 2}:
        raise ValueError("derivative_order must be 1 or 2")
    flat = samples.reshape(samples.shape[0], -1)
    output = np.full_like(flat, np.nan, dtype=np.float64)
    delta_t = np.diff(times)
    slopes = np.diff(flat, axis=0) / delta_t[:, None]
    if derivative_order == 1:
        output[1:] = slopes
    elif slopes.shape[0] >= 2:
        # Irregular-grid three-point backward second derivative.  It is the
        # slope change divided by the distance between interval midpoints.
        output[2:] = 2.0 * np.diff(slopes, axis=0) / (
            delta_t[1:, None] + delta_t[:-1, None]
        )
    return output.reshape(samples.shape)


@dataclass(frozen=True)
class CausalDerivativeState:
    velocity: np.ndarray
    acceleration: np.ndarray
    samples_seen: int
    retained_samples: int
    startup_state: str


class CausalDerivativeEstimator:
    """Bounded timestamp-aware derivative state for one online signal."""

    def __init__(
        self,
        *,
        method: str = "causal_polynomial",
        window_length: int = 9,
        polynomial_order: int = 3,
    ) -> None:
        if method not in {"backward_finite_difference", "causal_polynomial"}:
            raise ValueError(f"unsupported causal derivative method: {method}")
        if window_length <= polynomial_order or window_length < 3:
            raise ValueError("window_length must exceed polynomial_order and be at least 3")
        if polynomial_order < 2:
            raise ValueError("polynomial_order must be at least 2")
        self.method = method
        self.window_length = int(window_length)
        self.polynomial_order = int(polynomial_order)
        self._timestamps: deque[float] = deque(maxlen=self.window_length)
        self._values: deque[np.ndarray] = deque(maxlen=self.window_length)
        self._value_shape: tuple[int, ...] | None = None
        self._samples_seen = 0

    @property
    def retained_samples(self) -> int:
        return len(self._timestamps)

    @property
    def samples_seen(self) -> int:
        return self._samples_seen

    def update(self, timestamp: float, values: np.ndarray) -> CausalDerivativeState:
        sample = np.asarray(values, dtype=np.float64)
        if not np.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        if not np.isfinite(sample).all():
            raise ValueError("online derivative values must be finite")
        if self._value_shape is None:
            self._value_shape = sample.shape
        elif sample.shape != self._value_shape:
            raise ValueError("online derivative value shape changed")
        if self._timestamps and timestamp <= self._timestamps[-1]:
            raise ValueError("timestamps must be strictly increasing")
        self._timestamps.append(float(timestamp))
        self._values.append(sample.copy())
        self._samples_seen += 1

        times = np.asarray(self._timestamps, dtype=np.float64)
        stacked = np.stack(tuple(self._values), axis=0)
        velocity = np.full(sample.shape, np.nan, dtype=np.float64)
        acceleration = np.full(sample.shape, np.nan, dtype=np.float64)
        required = 3 if self.method == "backward_finite_difference" else self.polynomial_order + 1
        if self.method == "backward_finite_difference":
            if times.size >= 2:
                velocity = backward_finite_difference(times, stacked, 1)[-1]
            if times.size >= 3:
                acceleration = backward_finite_difference(times, stacked, 2)[-1]
        elif times.size >= required:
            local_times = times - times[-1]
            flat = stacked.reshape(stacked.shape[0], -1)
            velocity_flat = np.empty(flat.shape[1], dtype=np.float64)
            acceleration_flat = np.empty(flat.shape[1], dtype=np.float64)
            for column in range(flat.shape[1]):
                coefficients = np.polynomial.polynomial.polyfit(
                    local_times, flat[:, column], self.polynomial_order
                )
                velocity_flat[column] = coefficients[1]
                acceleration_flat[column] = 2.0 * coefficients[2]
            velocity = velocity_flat.reshape(sample.shape)
            acceleration = acceleration_flat.reshape(sample.shape)
        startup = "READY" if times.size >= required else f"WARMUP_{times.size}_OF_{required}"
        return CausalDerivativeState(
            velocity=velocity,
            acceleration=acceleration,
            samples_seen=self._samples_seen,
            retained_samples=times.size,
            startup_state=startup,
        )


def local_polynomial_derivative(
    timestamps: np.ndarray,
    values: np.ndarray,
    *,
    derivative_order: int = 1,
    polynomial_order: int = 3,
    window_length: int = 9,
    causal: bool = False,
) -> np.ndarray:
    """Fit a local polynomial on actual timestamps at every sample."""

    times = validate_timestamps(timestamps)
    samples = np.asarray(values, dtype=np.float64)
    if samples.shape[0] != times.shape[0]:
        raise ValueError("values first dimension must match timestamps")
    if derivative_order < 0 or derivative_order > polynomial_order:
        raise ValueError("derivative_order must be in [0, polynomial_order]")
    if window_length <= polynomial_order or window_length < 3:
        raise ValueError("window_length must exceed polynomial_order")
    flat = samples.reshape(samples.shape[0], -1)
    output = np.full_like(flat, np.nan, dtype=np.float64)
    half = window_length // 2
    factorial = float(np.prod(np.arange(1, derivative_order + 1))) if derivative_order else 1.0
    for index in range(times.shape[0]):
        if causal:
            end = index + 1
            start = max(0, end - window_length)
        else:
            start = max(0, index - half)
            end = min(times.shape[0], start + window_length)
            start = max(0, end - window_length)
        local_times = times[start:end] - times[index]
        local_values = flat[start:end]
        finite_rows = np.isfinite(local_values)
        for column in range(flat.shape[1]):
            valid = finite_rows[:, column]
            if int(valid.sum()) <= polynomial_order:
                continue
            coefficients = np.polynomial.polynomial.polyfit(
                local_times[valid], local_values[valid, column], polynomial_order
            )
            output[index, column] = factorial * coefficients[derivative_order]
    return output.reshape(samples.shape)
