"""Windowed common, local and quality feature extraction."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .derivatives import finite_difference, local_polynomial_derivative
from .timestamps import diagnose_timebase


FEATURE_VERSION = "videoeew-motion-v1"


def _safe_stat(function, values: np.ndarray, default: float = np.nan) -> float:
    finite = values[np.isfinite(values)]
    return float(function(finite)) if finite.size else float(default)


def _dominant_frequency(values: np.ndarray, sample_rate_hz: float) -> float:
    finite = np.asarray(values, dtype=np.float64)
    if finite.size < 4 or not np.isfinite(finite).all():
        return np.nan
    centered = finite - np.mean(finite)
    frequencies = np.fft.rfftfreq(centered.size, d=1 / sample_rate_hz)
    amplitude = np.abs(np.fft.rfft(centered))
    if amplitude.size <= 1:
        return np.nan
    amplitude[0] = 0
    return float(frequencies[int(np.argmax(amplitude))])


def _band_energy(values: np.ndarray, sample_rate_hz: float, low: float, high: float) -> float:
    finite = np.asarray(values, dtype=np.float64)
    if finite.size < 4 or not np.isfinite(finite).all():
        return np.nan
    frequencies = np.fft.rfftfreq(finite.size, d=1 / sample_rate_hz)
    spectrum = np.abs(np.fft.rfft(finite - np.mean(finite))) ** 2
    selection = (frequencies >= low) & (frequencies < high)
    return float(np.sum(spectrum[selection]) / max(finite.size**2, 1))


def extract_motion_features(
    timestamps: np.ndarray,
    common_xy: np.ndarray,
    residual_xy: np.ndarray,
    visibility: np.ndarray,
    *,
    inlier_ratio: np.ndarray | None = None,
    fit_rmse_px: np.ndarray | None = None,
    quality_flags: Iterable[str] | None = None,
    derivative_method: str = "local_polynomial",
    causal: bool = False,
) -> dict[str, float | str | int]:
    times = np.asarray(timestamps, dtype=np.float64)
    common = np.asarray(common_xy, dtype=np.float64)
    residual = np.asarray(residual_xy, dtype=np.float64)
    visible = np.asarray(visibility, dtype=bool)
    if common.shape != (times.shape[0], 2):
        raise ValueError("common_xy must have shape [T,2]")
    if residual.ndim != 3 or residual.shape[0] != times.shape[0] or residual.shape[2] != 2:
        raise ValueError("residual_xy must have shape [T,N,2]")
    if visible.shape != residual.shape[:2]:
        raise ValueError("visibility must have shape [T,N]")
    diagnostics = diagnose_timebase(times)
    if derivative_method == "local_polynomial":
        velocity = local_polynomial_derivative(times, common, derivative_order=1, causal=causal)
        acceleration = local_polynomial_derivative(times, common, derivative_order=2, causal=causal)
    elif derivative_method == "finite_difference":
        velocity = finite_difference(times, common, derivative_order=1)
        acceleration = finite_difference(times, common, derivative_order=2)
    else:
        raise ValueError(f"unknown derivative method: {derivative_method}")
    common_magnitude = np.linalg.norm(common, axis=1)
    velocity_magnitude = np.linalg.norm(velocity, axis=1)
    acceleration_magnitude = np.linalg.norm(acceleration, axis=1)
    residual_magnitude = np.linalg.norm(residual, axis=2)
    residual_magnitude[~visible] = np.nan
    point_rms = np.sqrt(np.nanmean(np.square(residual_magnitude), axis=0))
    per_frame_active = np.nanmean(residual_magnitude > 3 * max(np.nanmedian(residual_magnitude), 1e-6), axis=1)
    flags = sorted(set(quality_flags or ()))
    features: dict[str, float | str | int] = {
        "feature_version": FEATURE_VERSION,
        "window_start": float(times[0]),
        "window_end": float(times[-1]),
        "sample_count": int(times.size),
        "effective_fps": diagnostics.effective_fps,
        "timestamp_rms_jitter_s": diagnostics.rms_jitter_s,
        "estimated_missing_frames": diagnostics.estimated_missing_frames,
        "common_peak_displacement_px": _safe_stat(np.max, common_magnitude),
        "common_rms_displacement_px": float(np.sqrt(np.nanmean(np.square(common_magnitude)))),
        "common_peak_velocity_px_s": _safe_stat(np.max, velocity_magnitude),
        "common_peak_acceleration_px_s2": _safe_stat(np.max, acceleration_magnitude),
        "common_rms_acceleration_px_s2": float(np.sqrt(np.nanmean(np.square(acceleration_magnitude)))),
        "common_dominant_frequency_hz": _dominant_frequency(common[:, 0], diagnostics.effective_fps),
        "common_energy_0p3_1_hz": _band_energy(common[:, 0], diagnostics.effective_fps, 0.3, 1.0),
        "common_energy_1_3_hz": _band_energy(common[:, 0], diagnostics.effective_fps, 1.0, 3.0),
        "common_energy_3_8_hz": _band_energy(common[:, 0], diagnostics.effective_fps, 3.0, 8.0),
        "strong_motion_duration_s": float(np.sum(acceleration_magnitude >= 0.1 * np.nanmax(acceleration_magnitude)) / diagnostics.effective_fps),
        "residual_rms_px": float(np.sqrt(np.nanmean(np.square(residual_magnitude)))),
        "residual_peak_px": _safe_stat(np.max, residual_magnitude),
        "residual_median_point_rms_px": _safe_stat(np.median, point_rms),
        "residual_p95_point_rms_px": _safe_stat(lambda value: np.percentile(value, 95), point_rms),
        "residual_point_dispersion_px": _safe_stat(np.std, point_rms),
        "active_object_fraction": _safe_stat(np.mean, per_frame_active),
        "mean_visibility": float(np.mean(visible)),
        "mean_inlier_ratio": _safe_stat(np.mean, np.asarray(inlier_ratio, dtype=float)) if inlier_ratio is not None else np.nan,
        "mean_fit_rmse_px": _safe_stat(np.mean, np.asarray(fit_rmse_px, dtype=float)) if fit_rmse_px is not None else np.nan,
        "mean_active_track_count": float(np.mean(np.sum(visible, axis=1))),
        "quality_flags": "|".join(flags),
        "derivative_method": derivative_method,
        "causal": int(causal),
    }
    return features

