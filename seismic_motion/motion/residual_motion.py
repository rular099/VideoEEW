"""Sequence-level common/local decomposition with frame-wise audit state."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .global_motion import GlobalModel, TransformEstimate, apply_transform, fit_global_transform
from .quality import MotionQuality, QualityDecision, QualityThresholds, assess_motion_quality


@dataclass(frozen=True)
class MotionDecomposition:
    common_matrices: np.ndarray
    common_parameters: np.ndarray
    residual_xy_px: np.ndarray
    inlier_mask: np.ndarray
    quality: np.ndarray
    quality_reasons: tuple[tuple[str, ...], ...]
    estimates: tuple[TransformEstimate | None, ...]


def decompose_tracks(
    reference_xy: np.ndarray,
    tracks_xy: np.ndarray,
    visibility: np.ndarray,
    *,
    model: GlobalModel = "similarity",
    use_ransac: bool = True,
    ransac_threshold_px: float = 1.5,
    frame_size: tuple[int, int] | None = None,
    quality_thresholds: QualityThresholds = QualityThresholds(),
    seed: int = 0,
) -> MotionDecomposition:
    reference = np.asarray(reference_xy, dtype=np.float64)
    tracks = np.asarray(tracks_xy, dtype=np.float64)
    visible = np.asarray(visibility, dtype=bool)
    if tracks.ndim != 3 or tracks.shape[1:] != reference.shape:
        raise ValueError("tracks_xy must have shape [T,N,2] matching reference_xy")
    if visible.shape != tracks.shape[:2]:
        raise ValueError("visibility must have shape [T,N]")
    frames, points = visible.shape
    matrices = np.full((frames, 3, 3), np.nan, dtype=np.float64)
    parameters = np.full((frames, 8), np.nan, dtype=np.float64)
    residuals = np.full((frames, points, 2), np.nan, dtype=np.float64)
    inlier_mask = np.zeros((frames, points), dtype=bool)
    quality = np.full(frames, MotionQuality.INVALID.value, dtype="U10")
    reasons: list[tuple[str, ...]] = []
    estimates: list[TransformEstimate | None] = []
    previous: TransformEstimate | None = None
    for frame_index in range(frames):
        try:
            estimate = fit_global_transform(
                reference,
                tracks[frame_index],
                visible[frame_index],
                model=model,
                use_ransac=use_ransac,
                ransac_threshold_px=ransac_threshold_px,
                seed=seed + frame_index,
                frame_size=frame_size,
            )
            decision: QualityDecision = assess_motion_quality(
                estimate, quality_thresholds, previous=previous
            )
        except (ValueError, np.linalg.LinAlgError):
            estimates.append(None)
            reasons.append(("transform_fit_failed",))
            continue
        matrices[frame_index] = estimate.matrix
        parameters[frame_index] = [
            estimate.tx_px,
            estimate.ty_px,
            estimate.rotation_2d_rad,
            estimate.scale,
            estimate.shear,
            estimate.inlier_ratio,
            estimate.fit_rmse_px,
            estimate.spatial_coverage,
        ]
        predicted = apply_transform(estimate.matrix, reference)
        frame_residuals = tracks[frame_index] - predicted
        frame_residuals[~visible[frame_index]] = np.nan
        residuals[frame_index] = frame_residuals
        inlier_mask[frame_index] = estimate.inliers
        quality[frame_index] = decision.quality.value
        reasons.append(decision.reasons)
        estimates.append(estimate)
        if decision.quality != MotionQuality.INVALID:
            previous = estimate
    return MotionDecomposition(
        common_matrices=matrices,
        common_parameters=parameters,
        residual_xy_px=residuals,
        inlier_mask=inlier_mask,
        quality=quality,
        quality_reasons=tuple(reasons),
        estimates=tuple(estimates),
    )

