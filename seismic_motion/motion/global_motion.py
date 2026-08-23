"""Deterministic robust 2-D common image-motion estimation."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Literal

import numpy as np


GlobalModel = Literal["translation", "similarity", "affine", "homography"]
_MINIMUM_POINTS: dict[str, int] = {
    "translation": 1,
    "similarity": 2,
    "affine": 3,
    "homography": 4,
}


@dataclass(frozen=True)
class TransformEstimate:
    model: str
    matrix: np.ndarray
    inliers: np.ndarray
    residuals_px: np.ndarray
    tx_px: float
    ty_px: float
    rotation_2d_rad: float
    scale: float
    shear: float
    inlier_ratio: float
    fit_rmse_px: float
    num_valid_tracks: int
    num_inliers: int
    spatial_coverage: float
    condition_number: float


def apply_transform(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    homogeneous = np.concatenate([values, np.ones((values.shape[0], 1))], axis=1)
    projected = homogeneous @ np.asarray(matrix, dtype=np.float64).T
    denominator = projected[:, 2:3]
    valid = np.abs(denominator) > np.finfo(np.float64).eps
    result = np.full((values.shape[0], 2), np.nan, dtype=np.float64)
    result[valid[:, 0]] = projected[valid[:, 0], :2] / denominator[valid[:, 0]]
    return result


def _fit_translation(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    delta = np.mean(target - source, axis=0)
    matrix = np.eye(3, dtype=np.float64)
    matrix[:2, 2] = delta
    return matrix, 1.0


def _fit_similarity(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    count = source.shape[0]
    design = np.zeros((2 * count, 4), dtype=np.float64)
    values = target.reshape(-1)
    x, y = source[:, 0], source[:, 1]
    design[0::2] = np.stack([x, -y, np.ones(count), np.zeros(count)], axis=1)
    design[1::2] = np.stack([y, x, np.zeros(count), np.ones(count)], axis=1)
    parameters, _, rank, _ = np.linalg.lstsq(design, values, rcond=None)
    if rank < 4:
        raise np.linalg.LinAlgError("degenerate similarity points")
    a, b, tx, ty = parameters
    matrix = np.asarray([[a, -b, tx], [b, a, ty], [0, 0, 1]], dtype=np.float64)
    return matrix, float(np.linalg.cond(design))


def _fit_affine(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    count = source.shape[0]
    design = np.zeros((2 * count, 6), dtype=np.float64)
    values = target.reshape(-1)
    x, y = source[:, 0], source[:, 1]
    design[0::2, 0:3] = np.stack([x, y, np.ones(count)], axis=1)
    design[1::2, 3:6] = np.stack([x, y, np.ones(count)], axis=1)
    parameters, _, rank, _ = np.linalg.lstsq(design, values, rcond=None)
    if rank < 6:
        raise np.linalg.LinAlgError("degenerate affine points")
    matrix = np.eye(3, dtype=np.float64)
    matrix[:2] = parameters.reshape(2, 3)
    return matrix, float(np.linalg.cond(design))


def _fit_homography(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    count = source.shape[0]
    design = np.zeros((2 * count, 9), dtype=np.float64)
    for index, ((x, y), (u, v)) in enumerate(zip(source, target)):
        design[2 * index] = [-x, -y, -1, 0, 0, 0, u * x, u * y, u]
        design[2 * index + 1] = [0, 0, 0, -x, -y, -1, v * x, v * y, v]
    _, singular_values, vectors = np.linalg.svd(design)
    matrix = vectors[-1].reshape(3, 3)
    if abs(matrix[2, 2]) < np.finfo(np.float64).eps:
        raise np.linalg.LinAlgError("degenerate homography scale")
    matrix /= matrix[2, 2]
    condition = float(singular_values[0] / max(singular_values[-1], np.finfo(float).eps))
    return matrix, condition


def _fit(model: GlobalModel, source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    if model == "translation":
        return _fit_translation(source, target)
    if model == "similarity":
        return _fit_similarity(source, target)
    if model == "affine":
        return _fit_affine(source, target)
    if model == "homography":
        return _fit_homography(source, target)
    raise ValueError(f"unsupported global motion model: {model}")


def _candidate_subsets(count: int, sample_size: int, iterations: int, seed: int):
    total_combinations = 1
    for value in range(sample_size):
        total_combinations = total_combinations * (count - value) // (value + 1)
    if total_combinations <= iterations:
        yield from combinations(range(count), sample_size)
        return
    rng = np.random.default_rng(seed)
    seen: set[tuple[int, ...]] = set()
    while len(seen) < iterations:
        subset = tuple(sorted(rng.choice(count, sample_size, replace=False).tolist()))
        if subset not in seen:
            seen.add(subset)
            yield subset


def _coverage(points: np.ndarray, frame_size: tuple[int, int] | None) -> float:
    if points.shape[0] < 2:
        return 0.0
    extent = np.ptp(points, axis=0)
    if frame_size is None:
        full_extent = np.ptp(points, axis=0)
        denominator = max(float(full_extent[0] * full_extent[1]), 1.0)
    else:
        height, width = frame_size
        denominator = max(float(height * width), 1.0)
    return float(np.clip(float(extent[0] * extent[1]) / denominator, 0.0, 1.0))


def decompose_matrix(matrix: np.ndarray) -> tuple[float, float, float, float, float]:
    values = np.asarray(matrix, dtype=np.float64)
    linear = values[:2, :2]
    tx, ty = values[:2, 2]
    scale_x = float(np.linalg.norm(linear[:, 0]))
    scale_y = float(np.linalg.norm(linear[:, 1]))
    scale = float(np.sqrt(max(scale_x * scale_y, 0.0)))
    rotation = float(np.arctan2(linear[1, 0], linear[0, 0]))
    normalized = linear / max(scale, np.finfo(float).eps)
    shear = float(np.dot(normalized[:, 0], normalized[:, 1]))
    return float(tx), float(ty), rotation, scale, shear


def fit_global_transform(
    reference_xy: np.ndarray,
    observed_xy: np.ndarray,
    visible: np.ndarray | None = None,
    *,
    model: GlobalModel = "similarity",
    use_ransac: bool = True,
    ransac_threshold_px: float = 1.5,
    ransac_iterations: int = 128,
    seed: int = 0,
    frame_size: tuple[int, int] | None = None,
) -> TransformEstimate:
    """Fit one frame transform while retaining an inlier mask for all points."""

    reference = np.asarray(reference_xy, dtype=np.float64)
    observed = np.asarray(observed_xy, dtype=np.float64)
    if reference.shape != observed.shape or reference.ndim != 2 or reference.shape[1] != 2:
        raise ValueError("reference_xy and observed_xy must both have shape [N,2]")
    finite = np.isfinite(reference).all(axis=1) & np.isfinite(observed).all(axis=1)
    if visible is not None:
        visibility = np.asarray(visible, dtype=bool)
        if visibility.shape != (reference.shape[0],):
            raise ValueError("visible must have shape [N]")
        finite &= visibility
    valid_indices = np.flatnonzero(finite)
    minimum = _MINIMUM_POINTS[model]
    if valid_indices.size < minimum:
        raise ValueError(f"{model} requires at least {minimum} valid tracks")
    source = reference[valid_indices]
    target = observed[valid_indices]
    best_inliers = np.ones(source.shape[0], dtype=bool)
    if use_ransac and source.shape[0] > minimum:
        best_score = (-1, np.inf)
        for subset in _candidate_subsets(source.shape[0], minimum, ransac_iterations, seed):
            try:
                candidate, _ = _fit(model, source[list(subset)], target[list(subset)])
            except np.linalg.LinAlgError:
                continue
            residuals = np.linalg.norm(apply_transform(candidate, source) - target, axis=1)
            inliers = residuals <= ransac_threshold_px
            inlier_count = int(inliers.sum())
            median_error = float(np.median(residuals[inliers])) if inlier_count else np.inf
            score = (inlier_count, -median_error)
            if score > best_score:
                best_score = score
                best_inliers = inliers
    if int(best_inliers.sum()) < minimum:
        best_inliers = np.ones(source.shape[0], dtype=bool)
    matrix, fit_condition = _fit(model, source[best_inliers], target[best_inliers])
    all_valid_residuals = np.linalg.norm(apply_transform(matrix, source) - target, axis=1)
    refined_inliers = all_valid_residuals <= ransac_threshold_px if use_ransac else best_inliers
    if int(refined_inliers.sum()) >= minimum and not np.array_equal(refined_inliers, best_inliers):
        matrix, fit_condition = _fit(model, source[refined_inliers], target[refined_inliers])
        all_valid_residuals = np.linalg.norm(apply_transform(matrix, source) - target, axis=1)
    else:
        refined_inliers = best_inliers
    inliers_full = np.zeros(reference.shape[0], dtype=bool)
    inliers_full[valid_indices[refined_inliers]] = True
    residuals_full = np.full(reference.shape[0], np.nan, dtype=np.float64)
    residuals_full[valid_indices] = all_valid_residuals
    rmse = float(np.sqrt(np.mean(np.square(all_valid_residuals[refined_inliers]))))
    tx, ty, rotation, scale, shear = decompose_matrix(matrix)
    linear_condition = float(np.linalg.cond(matrix[:2, :2]))
    return TransformEstimate(
        model=model,
        matrix=matrix,
        inliers=inliers_full,
        residuals_px=residuals_full,
        tx_px=tx,
        ty_px=ty,
        rotation_2d_rad=rotation,
        scale=scale,
        shear=shear,
        inlier_ratio=float(refined_inliers.mean()),
        fit_rmse_px=rmse,
        num_valid_tracks=int(valid_indices.size),
        num_inliers=int(refined_inliers.sum()),
        spatial_coverage=_coverage(source[refined_inliers], frame_size),
        condition_number=max(linear_condition, fit_condition),
    )

