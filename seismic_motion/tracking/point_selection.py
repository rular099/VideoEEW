"""Manual and spatially distributed automatic query-point selection."""

from __future__ import annotations

import numpy as np


def validate_manual_points(
    points: np.ndarray, width: int, height: int, expected_count: int | None = None
) -> np.ndarray:
    result = np.asarray(points, dtype=np.float32)
    if result.ndim != 2 or result.shape[1] != 2:
        raise ValueError("manual points must have shape [N,2]")
    if expected_count is not None and result.shape[0] != expected_count:
        raise ValueError(f"expected {expected_count} points, got {result.shape[0]}")
    if not np.isfinite(result).all():
        raise ValueError("manual points contain non-finite coordinates")
    if np.any(result[:, 0] < 0) or np.any(result[:, 0] >= width):
        raise ValueError("manual x coordinates are outside the image")
    if np.any(result[:, 1] < 0) or np.any(result[:, 1] >= height):
        raise ValueError("manual y coordinates are outside the image")
    return result


def _roi_bounds(
    width: int, height: int, roi: tuple[int, int, int, int] | None
) -> tuple[int, int, int, int]:
    if roi is None:
        return 0, 0, width, height
    x, y, w, h = (int(value) for value in roi)
    if w <= 0 or h <= 0:
        raise ValueError("ROI width and height must be positive")
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(width, x + w), min(height, y + h)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("ROI does not intersect the image")
    return x0, y0, x1, y1


def select_distributed_corners(
    frame_rgb: np.ndarray,
    num_points: int,
    roi: tuple[int, int, int, int] | None = None,
    cells: tuple[int, int] | None = None,
    quality_level: float = 0.01,
    min_distance: float = 8.0,
) -> np.ndarray:
    """Select Shi-Tomasi corners with a per-cell quota for spatial coverage."""

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - exercised on deployment images
        raise RuntimeError("OpenCV is required for automatic corner selection") from exc
    frame = np.asarray(frame_rgb)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame_rgb must have shape [H,W,3]")
    if num_points <= 0:
        raise ValueError("num_points must be positive")
    height, width = frame.shape[:2]
    x0, y0, x1, y1 = _roi_bounds(width, height, roi)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    if cells is None:
        columns = max(1, int(np.ceil(np.sqrt(num_points * (x1 - x0) / max(y1 - y0, 1)))))
        rows = max(1, int(np.ceil(num_points / columns)))
    else:
        rows, columns = cells
    quota = max(1, int(np.ceil(num_points / (rows * columns))))
    candidates: list[tuple[float, float, float]] = []
    for row in range(rows):
        cy0 = y0 + (y1 - y0) * row // rows
        cy1 = y0 + (y1 - y0) * (row + 1) // rows
        for column in range(columns):
            cx0 = x0 + (x1 - x0) * column // columns
            cx1 = x0 + (x1 - x0) * (column + 1) // columns
            cell = gray[cy0:cy1, cx0:cx1]
            corners = cv2.goodFeaturesToTrack(
                cell,
                maxCorners=quota,
                qualityLevel=quality_level,
                minDistance=min_distance,
                blockSize=5,
                useHarrisDetector=False,
            )
            if corners is None:
                continue
            for point in corners.reshape(-1, 2):
                px, py = float(point[0] + cx0), float(point[1] + cy0)
                response = float(gray[int(round(py)), int(round(px))])
                candidates.append((response, px, py))
    if len(candidates) < num_points:
        fallback = cv2.goodFeaturesToTrack(
            gray[y0:y1, x0:x1],
            maxCorners=num_points * 3,
            qualityLevel=max(quality_level / 2, 1e-5),
            minDistance=max(2.0, min_distance / 2),
            blockSize=5,
        )
        if fallback is not None:
            for point in fallback.reshape(-1, 2):
                candidates.append((0.0, float(point[0] + x0), float(point[1] + y0)))
    selected: list[np.ndarray] = []
    for _, px, py in sorted(candidates, reverse=True):
        candidate = np.asarray([px, py], dtype=np.float32)
        if all(np.linalg.norm(candidate - point) >= min_distance / 2 for point in selected):
            selected.append(candidate)
        if len(selected) == num_points:
            break
    if len(selected) < num_points:
        raise RuntimeError(f"found only {len(selected)} distributed corners; need {num_points}")
    return np.stack(selected)


def spatial_coverage(points: np.ndarray, width: int, height: int) -> float:
    """Axis-aligned point extent divided by image area, in [0, 1]."""

    values = np.asarray(points, dtype=np.float64)
    if values.shape[0] < 2 or width <= 0 or height <= 0:
        return 0.0
    extent = np.ptp(values, axis=0)
    return float(np.clip((extent[0] * extent[1]) / (width * height), 0.0, 1.0))

