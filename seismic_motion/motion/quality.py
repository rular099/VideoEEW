"""Quality gates that reject plausible-looking but unsupported transforms."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .global_motion import TransformEstimate


class MotionQuality(str, Enum):
    GOOD = "GOOD"
    DEGRADED = "DEGRADED"
    INVALID = "INVALID"


@dataclass(frozen=True)
class QualityThresholds:
    min_valid_tracks: int = 10
    min_inlier_ratio: float = 0.6
    min_spatial_coverage: float = 0.08
    max_fit_rmse_px: float = 2.0
    max_condition_number: float = 1e8
    max_translation_jump_px: float = 20.0
    max_rotation_jump_rad: float = np.deg2rad(3.0)


@dataclass(frozen=True)
class QualityDecision:
    quality: MotionQuality
    reasons: tuple[str, ...]


def assess_motion_quality(
    estimate: TransformEstimate,
    thresholds: QualityThresholds = QualityThresholds(),
    previous: TransformEstimate | None = None,
) -> QualityDecision:
    invalid: list[str] = []
    degraded: list[str] = []
    if estimate.num_valid_tracks < thresholds.min_valid_tracks:
        invalid.append("insufficient_tracks")
    if estimate.num_inliers < max(2, thresholds.min_valid_tracks // 2):
        invalid.append("insufficient_inliers")
    if estimate.inlier_ratio < thresholds.min_inlier_ratio:
        invalid.append("low_inlier_ratio")
    if not np.isfinite(estimate.fit_rmse_px) or estimate.fit_rmse_px > thresholds.max_fit_rmse_px:
        invalid.append("high_fit_rmse")
    if not np.isfinite(estimate.condition_number) or estimate.condition_number > thresholds.max_condition_number:
        invalid.append("ill_conditioned_transform")
    if estimate.spatial_coverage < thresholds.min_spatial_coverage:
        degraded.append("poor_spatial_coverage")
    if previous is not None:
        translation_jump = np.hypot(
            estimate.tx_px - previous.tx_px, estimate.ty_px - previous.ty_px
        )
        rotation_jump = abs(estimate.rotation_2d_rad - previous.rotation_2d_rad)
        if translation_jump > thresholds.max_translation_jump_px:
            degraded.append("translation_jump")
        if rotation_jump > thresholds.max_rotation_jump_rad:
            degraded.append("rotation_jump")
    if invalid:
        return QualityDecision(MotionQuality.INVALID, tuple(invalid + degraded))
    if degraded:
        return QualityDecision(MotionQuality.DEGRADED, tuple(degraded))
    return QualityDecision(MotionQuality.GOOD, ())

