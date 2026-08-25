"""Deterministic subpixel image-motion and track-ground-truth generator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:  # OpenCV is optional in the shared GPU environment.
    import cv2 as _cv2
except ImportError:  # pragma: no cover - exercised on server 242
    _cv2 = None


@dataclass(frozen=True)
class SyntheticSequence:
    frames_rgb: np.ndarray
    timestamps: np.ndarray
    tracks_xy_px: np.ndarray
    visibility: np.ndarray
    reference_xy_px: np.ndarray
    common_matrices: np.ndarray
    local_residual_px: np.ndarray
    case: str
    fps: float
    translation_amplitude_px: float
    translation_frequency_hz: float
    rotation_amplitude_deg: float

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            frames_rgb=self.frames_rgb,
            timestamps=self.timestamps,
            tracks_xy_px=self.tracks_xy_px,
            visibility=self.visibility,
            reference_xy_px=self.reference_xy_px,
            common_matrices=self.common_matrices,
            local_residual_px=self.local_residual_px,
            case=np.asarray(self.case),
            fps=np.asarray(self.fps),
            translation_amplitude_px=np.asarray(self.translation_amplitude_px),
            translation_frequency_hz=np.asarray(self.translation_frequency_hz),
            rotation_amplitude_deg=np.asarray(self.rotation_amplitude_deg),
        )


def _texture(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(127, 45, size=(height, width)).clip(0, 255).astype(np.uint8)
    if _cv2 is not None:
        noise = _cv2.GaussianBlur(noise, (0, 0), 1.2)
    else:
        from scipy import ndimage

        noise = ndimage.gaussian_filter(noise, 1.2).astype(np.uint8)
    rgb = np.stack(
        [noise, np.roll(noise, 7, axis=0), np.roll(noise, 11, axis=1)], axis=-1
    )
    for _ in range(35):
        x = int(rng.integers(5, max(6, width - 5)))
        y = int(rng.integers(5, max(6, height - 5)))
        radius = int(rng.integers(2, 8))
        color = tuple(int(v) for v in rng.integers(20, 235, size=3))
        if _cv2 is not None:
            _cv2.circle(rgb, (x, y), radius, color, -1, lineType=_cv2.LINE_AA)
        else:
            yy, xx = np.ogrid[:height, :width]
            rgb[(xx - x) ** 2 + (yy - y) ** 2 <= radius**2] = color
    return rgb


def _warp_affine(image: np.ndarray, matrix: np.ndarray, width: int, height: int) -> np.ndarray:
    if _cv2 is not None:
        return _cv2.warpAffine(
            image,
            matrix,
            (width, height),
            flags=_cv2.INTER_LINEAR,
            borderMode=_cv2.BORDER_REFLECT101,
        )
    from scipy import ndimage

    homogeneous = np.eye(3, dtype=np.float64)
    homogeneous[:2] = matrix
    inverse = np.linalg.inv(homogeneous)
    swap = np.asarray([[0.0, 1.0], [1.0, 0.0]])
    affine_yx = swap @ inverse[:2, :2] @ swap
    offset_yx = swap @ inverse[:2, 2]
    channels = [
        ndimage.affine_transform(
            image[:, :, channel],
            affine_yx,
            offset=offset_yx,
            output_shape=(height, width),
            order=1,
            mode="reflect",
            prefilter=False,
        )
        for channel in range(image.shape[2])
    ]
    return np.stack(channels, axis=-1).clip(0, 255).astype(np.uint8)


def _draw_cross(image: np.ndarray, x: int, y: int, color: tuple[int, int, int]) -> None:
    if _cv2 is not None:
        _cv2.drawMarker(
            image,
            (x, y),
            color,
            _cv2.MARKER_CROSS,
            5,
            1,
            _cv2.LINE_AA,
        )
        return
    image[y, max(0, x - 2) : min(image.shape[1], x + 3)] = color
    image[max(0, y - 2) : min(image.shape[0], y + 3), x] = color


def _grid_points(height: int, width: int, rows: int, columns: int) -> np.ndarray:
    xs = np.linspace(width * 0.08, width * 0.92, columns)
    ys = np.linspace(height * 0.08, height * 0.92, rows)
    xx, yy = np.meshgrid(xs, ys)
    return np.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1).astype(np.float32)


def generate_sequence(
    case: str = "translation_rotation_local",
    *,
    fps: float = 30.0,
    duration_s: float = 4.0,
    image_size: tuple[int, int] = (240, 320),
    point_grid: tuple[int, int] = (8, 10),
    translation_amplitude_px: float = 0.5,
    translation_frequency_hz: float = 2.0,
    rotation_amplitude_deg: float = 0.1,
    local_fraction: float = 0.2,
    local_amplitude_px: float = 1.0,
    local_frequency_hz: float = 3.0,
    local_phase_rad: float = 0.7,
    seed: int = 7,
    render_frames: bool = True,
) -> SyntheticSequence:
    """Generate frames and exact point tracks for one required Phase C case."""

    valid_cases = {
        "translation",
        "translation_local",
        "rotation",
        "rotation_local",
        "translation_rotation",
        "translation_rotation_local",
        "occlusion",
        "motion_blur",
        "illumination_change",
        "low_texture",
        "degraded",
    }
    if case not in valid_cases:
        raise ValueError(f"unknown synthetic case: {case}")
    if fps <= 0 or duration_s <= 0:
        raise ValueError("fps and duration_s must be positive")
    rng = np.random.default_rng(seed)
    height, width = image_size
    frames_count = int(round(duration_s * fps))
    timestamps = np.arange(frames_count, dtype=np.float64) / fps
    reference = _grid_points(height, width, *point_grid)
    points_count = reference.shape[0]
    local_count = int(round(points_count * local_fraction))
    local_mask = np.zeros(points_count, dtype=bool)
    if "local" in case or case == "degraded":
        local_mask[-local_count:] = True
    base = _texture(height, width, rng) if render_frames else None
    if base is not None and case == "low_texture":
        from scipy import ndimage

        mean = np.mean(base, axis=(0, 1), keepdims=True)
        base = np.clip(mean + 0.08 * (base - mean), 0, 255)
        base = ndimage.gaussian_filter(base, sigma=(3.0, 3.0, 0.0)).astype(np.uint8)
    center = np.asarray([(width - 1) / 2, (height - 1) / 2], dtype=np.float64)
    frames = (
        np.empty((frames_count, height, width, 3), dtype=np.uint8)
        if render_frames
        else np.empty((0, height, width, 3), dtype=np.uint8)
    )
    tracks = np.empty((frames_count, points_count, 2), dtype=np.float32)
    visibility = np.ones((frames_count, points_count), dtype=bool)
    matrices = np.empty((frames_count, 2, 3), dtype=np.float64)
    residual = np.zeros_like(tracks)
    includes_translation = case in {
        "translation",
        "translation_local",
        "translation_rotation",
        "translation_rotation_local",
        "occlusion",
        "motion_blur",
        "illumination_change",
        "low_texture",
        "degraded",
    }
    includes_rotation = case in {
        "rotation",
        "rotation_local",
        "translation_rotation",
        "translation_rotation_local",
        "degraded",
    }
    for index, time_s in enumerate(timestamps):
        phase = 2 * np.pi * translation_frequency_hz * time_s
        dx = translation_amplitude_px * np.sin(phase) if includes_translation else 0.0
        dy = 0.35 * translation_amplitude_px * np.sin(phase + 0.35) if includes_translation else 0.0
        angle = rotation_amplitude_deg * np.sin(phase * 0.65) if includes_rotation else 0.0
        angle_rad = np.deg2rad(angle)
        alpha, beta = np.cos(angle_rad), np.sin(angle_rad)
        matrix = np.asarray(
            [
                [alpha, beta, (1 - alpha) * center[0] - beta * center[1]],
                [-beta, alpha, beta * center[0] + (1 - alpha) * center[1]],
            ],
            dtype=np.float64,
        )
        matrix[:, 2] += [dx, dy]
        matrices[index] = matrix
        homogeneous = np.concatenate(
            [reference.astype(np.float64), np.ones((points_count, 1))], axis=1
        )
        transformed = homogeneous @ matrix.T
        if np.any(local_mask):
            local_phase = 2 * np.pi * local_frequency_hz * time_s + local_phase_rad
            local_delta = np.asarray(
                [local_amplitude_px * np.sin(local_phase), 0.6 * local_amplitude_px * np.cos(local_phase)],
                dtype=np.float32,
            )
            residual[index, local_mask] = local_delta
            transformed[local_mask] += local_delta
        tracks[index] = transformed
        if case in {"occlusion", "degraded"} and frames_count // 3 <= index < frames_count // 2:
            x0 = width * 2 // 3
            visibility[index, reference[:, 0] >= x0] = False
        if render_frames:
            assert base is not None
            rendered = _warp_affine(base, matrix, width, height)
            for point_index in np.flatnonzero(local_mask):
                px, py = np.round(tracks[index, point_index]).astype(int)
                if 0 <= px < width and 0 <= py < height:
                    _draw_cross(rendered, px, py, (245, 35, 35))
            if case in {"illumination_change", "degraded"}:
                illumination = 1.0 + 0.35 * np.sin(2 * np.pi * 0.7 * time_s)
                rendered = np.clip(rendered.astype(np.float32) * illumination, 0, 255).astype(np.uint8)
            if case in {"motion_blur", "degraded"}:
                if index % max(2, int(round(fps / 4))) == 0:
                    from scipy import ndimage

                    rendered = ndimage.uniform_filter1d(
                        rendered, size=7, axis=1, mode="reflect"
                    ).astype(np.uint8)
            if case in {"occlusion", "degraded"}:
                if frames_count // 3 <= index < frames_count // 2:
                    rendered[:, x0:width] = 0
            frames[index] = rendered
    return SyntheticSequence(
        frames_rgb=frames,
        timestamps=timestamps,
        tracks_xy_px=tracks,
        visibility=visibility,
        reference_xy_px=reference,
        common_matrices=matrices,
        local_residual_px=residual,
        case=case,
        fps=float(fps),
        translation_amplitude_px=float(translation_amplitude_px),
        translation_frequency_hz=float(translation_frequency_hz),
        rotation_amplitude_deg=float(rotation_amplitude_deg),
    )
