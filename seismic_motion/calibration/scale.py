"""Pixel-to-mm conversion that refuses to invent missing geometry."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

import numpy as np


class CalibrationState(str, Enum):
    VALID = "VALID"
    UNCALIBRATED = "UNCALIBRATED"
    OUTSIDE_VALID_REGION = "OUTSIDE_VALID_REGION"


@dataclass(frozen=True)
class ScaleCalibration:
    scale_id: str
    method: str
    mm_per_px: float | None
    reference_points: tuple[tuple[float, float], ...] = ()
    reference_length_mm: float | None = None
    reference_length_px: float | None = None
    valid_roi: tuple[float, float, float, float] | None = None
    calibration_timestamp: str | None = None
    calibration_error: float | None = None

    @classmethod
    def uncalibrated(cls, scale_id: str = "none") -> "ScaleCalibration":
        return cls(scale_id=scale_id, method="uncalibrated", mm_per_px=None)

    @classmethod
    def from_known_length(
        cls,
        *,
        scale_id: str,
        known_length_mm: float,
        known_length_px: float,
        reference_points: tuple[tuple[float, float], ...] = (),
        valid_roi: tuple[float, float, float, float] | None = None,
        calibration_timestamp: str | None = None,
        calibration_error: float | None = None,
    ) -> "ScaleCalibration":
        if known_length_mm <= 0 or known_length_px <= 0:
            raise ValueError("known lengths must be positive")
        return cls(
            scale_id=scale_id,
            method="known_length",
            mm_per_px=float(known_length_mm / known_length_px),
            reference_points=reference_points,
            reference_length_mm=float(known_length_mm),
            reference_length_px=float(known_length_px),
            valid_roi=valid_roi,
            calibration_timestamp=calibration_timestamp,
            calibration_error=calibration_error,
        )

    @property
    def state(self) -> CalibrationState:
        return CalibrationState.VALID if self.mm_per_px is not None else CalibrationState.UNCALIBRATED

    def validate_points(self, points_xy: np.ndarray) -> np.ndarray:
        points = np.asarray(points_xy, dtype=np.float64)
        valid = np.ones(points.shape[:-1], dtype=bool)
        if self.valid_roi is None:
            return valid
        x, y, width, height = self.valid_roi
        return (
            (points[..., 0] >= x)
            & (points[..., 0] <= x + width)
            & (points[..., 1] >= y)
            & (points[..., 1] <= y + height)
        )

    def convert_displacement(self, displacement_px: np.ndarray) -> np.ndarray:
        if self.mm_per_px is None:
            raise RuntimeError("scale is uncalibrated; millimetre conversion is unavailable")
        return np.asarray(displacement_px, dtype=np.float64) * self.mm_per_px

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["state"] = self.state.value
        return value

