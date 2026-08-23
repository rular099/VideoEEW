"""Timestamp-aware finite-difference and local-polynomial derivatives."""

from __future__ import annotations

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
