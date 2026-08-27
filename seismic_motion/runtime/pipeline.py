"""Complete offline video-to-features pipeline using bounded tracker windows."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np
import yaml

from seismic_motion.calibration.scale import ScaleCalibration
from seismic_motion.config import config_sha256
from seismic_motion.diagnostics.provenance import (
    environment_snapshot,
    git_state,
    sha256_file,
)
from seismic_motion.motion.quality import QualityThresholds
from seismic_motion.motion.residual_motion import MotionDecomposition, decompose_tracks
from seismic_motion.signal.derivatives import local_polynomial_derivative
from seismic_motion.signal.features import extract_motion_features
from seismic_motion.signal.filtering import bandpass_filter
from seismic_motion.tracking.cotracker_adapter import CoTrackerAdapter, CoTrackerAdapterConfig
from seismic_motion.tracking.online_buffer import SlidingFrameBuffer
from seismic_motion.tracking.types import TrackBatch, concatenate_track_batches

from .metrics import (
    MemoryRecord,
    TimingRecord,
    sample_memory,
    summarize_timings,
    write_memory_csv,
    write_timing_csv,
)


@dataclass(frozen=True)
class OfflinePipelineResult:
    tracks: TrackBatch
    motion: MotionDecomposition
    filtered_common_xy_px: np.ndarray
    acceleration_proxy_px_s2: np.ndarray
    features: dict[str, float | str | int]
    timing: tuple[TimingRecord, ...]
    memory: tuple[MemoryRecord, ...]
    events: tuple[dict[str, object], ...]
    video_metadata: dict[str, object]


def _resize_rgb(frame: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
    height, width = output_size
    if np.asarray(frame).shape[:2] == (height, width):
        return np.asarray(frame)
    try:
        import cv2

        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    except ImportError:  # pragma: no cover - server path
        from PIL import Image

        return np.asarray(Image.fromarray(frame).resize((width, height), Image.Resampling.LANCZOS))


def _preprocess_frame(
    frame: np.ndarray,
    roi: list[int] | tuple[int, int, int, int] | None,
    output_size: tuple[int, int],
) -> np.ndarray:
    values = np.asarray(frame)
    if roi is not None:
        x, y, width, height = (int(value) for value in roi)
        values = values[y : y + height, x : x + width]
        if values.size == 0:
            raise ValueError("configured ROI is empty for the source frame")
    return _resize_rgb(values, output_size)


def _video_fps(path: str | Path, fps_override: float | None) -> tuple[float, str, dict[str, Any]]:
    if fps_override is not None:
        if fps_override <= 0:
            raise ValueError("fps_override must be positive")
        return float(fps_override), "user_override", {}
    try:
        import imageio.v3 as iio

        metadata = iio.immeta(path, plugin="FFMPEG")
        timestamp_source = "container_constant_frame_rate"
    except ImportError:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("video decoding requires imageio[ffmpeg] or OpenCV") from exc
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"video cannot be opened: {path}")
        metadata = {
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
            "size": (
                int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            ),
            "nframes": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            "decoder_backend": "opencv_fallback",
        }
        capture.release()
        timestamp_source = "container_constant_frame_rate_opencv_fallback"
    fps = float(metadata.get("fps", 0.0))
    if fps <= 0:
        raise RuntimeError("video FPS is unavailable; provide --fps")
    return fps, timestamp_source, metadata


def _iter_video_rgb(
    path: str | Path, *, output_size: tuple[int, int] | None = None
):
    """Decode incrementally, falling back to OpenCV without changing environments."""

    try:
        import imageio.v3 as iio
    except ImportError:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("video decoding requires imageio[ffmpeg] or OpenCV") from exc
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"video cannot be opened: {path}")
        try:
            while True:
                ok, frame_bgr = capture.read()
                if not ok:
                    break
                if output_size is not None:
                    height, width = output_size
                    frame_bgr = cv2.resize(
                        frame_bgr, (width, height), interpolation=cv2.INTER_AREA
                    )
                yield cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        finally:
            capture.release()
        return
    iterator_options: dict[str, object] = {"plugin": "FFMPEG"}
    if output_size is not None:
        height, width = output_size
        iterator_options["size"] = (width, height)
    yield from iio.imiter(path, **iterator_options)


def run_offline_video(
    video_path: str | Path,
    config: dict[str, Any],
    *,
    cotracker_root: str,
    checkpoint: str,
    device: str = "cuda",
    fps_override: float | None = None,
) -> OfflinePipelineResult:
    """Decode incrementally, track bounded windows, then produce audit arrays."""

    tracker_config = config["tracker"]
    runtime_config = config["runtime"]
    video_config = config["video"]
    signal_config = config["signal"]
    run_start_utc = datetime.now(timezone.utc).isoformat()
    model_size = tuple(int(value) for value in tracker_config["model_resolution"])
    fps, timestamp_source, source_metadata = _video_fps(video_path, fps_override)
    adapter_config = CoTrackerAdapterConfig(
        cotracker_root=cotracker_root,
        checkpoint=checkpoint,
        device=device,
        num_points=int(tracker_config["num_points"]),
        window_len=int(tracker_config["window_len"]),
        step=int(tracker_config["step"]),
        iters=int(tracker_config["iters"]),
        point_mode=str(tracker_config["point_mode"]),
        roi=None,
        max_blocks_before_reseed=int(tracker_config.get("max_blocks_before_reseed", 64)),
    )
    events: list[dict[str, object]] = []
    adapter = CoTrackerAdapter(adapter_config, event_sink=events.append)
    buffer = SlidingFrameBuffer(adapter_config.window_len, adapter_config.step)
    batches: list[TrackBatch] = []
    timing: list[TimingRecord] = []
    memory: list[MemoryRecord] = []
    frame_count = 0
    source_shape: tuple[int, ...] | None = None
    last_processed_start: int | None = None
    recent_capture_ms: list[float] = []
    recent_preprocess_ms: list[float] = []
    direct_decode_resize = bool(
        video_config.get("decode_resize_backend") == "ffmpeg"
        and video_config.get("roi") is None
    )
    if direct_decode_resize:
        source_size = source_metadata.get("size") or source_metadata.get("source_size")
        if isinstance(source_size, (tuple, list)) and len(source_size) == 2:
            source_shape = (int(source_size[1]), int(source_size[0]), 3)
    for frame_index, frame in enumerate(
        _iter_video_rgb(
            video_path, output_size=model_size if direct_decode_resize else None
        )
    ):
        capture_end = time.perf_counter()
        frame_array = np.asarray(frame)
        if source_shape is None:
            source_shape = frame_array.shape
        preprocess_start = time.perf_counter()
        processed = (
            frame_array
            if direct_decode_resize
            else _preprocess_frame(frame_array, video_config.get("roi"), model_size)
        )
        preprocess_ms = (time.perf_counter() - preprocess_start) * 1000
        recent_capture_ms.append(0.0 if frame_index == 0 else 0.0)
        recent_preprocess_ms.append(preprocess_ms)
        window = buffer.append(processed, frame_index / fps, frame_index)
        frame_count += 1
        if window is None:
            continue
        tracker_start = time.perf_counter()
        batch = adapter.process_window(window.frames, window.timestamps, window.frame_indices)
        tracker_ms = (time.perf_counter() - tracker_start) * 1000
        batches.append(batch)
        last_processed_start = int(window.frame_indices[0])
        consumed = adapter_config.window_len if len(timing) == 0 else adapter_config.step
        timing.append(
            TimingRecord(
                timestamp=float(window.timestamps[0]),
                block_index=len(timing),
                capture_ms=float(np.sum(recent_capture_ms[-consumed:])),
                preprocess_ms=float(np.sum(recent_preprocess_ms[-consumed:])),
                tracker_ms=tracker_ms,
                total_pipeline_ms=tracker_ms + float(np.sum(recent_preprocess_ms[-consumed:])),
            )
        )
        memory.append(sample_memory(float(window.timestamps[0]), len(timing) - 1))
        _ = capture_end  # capture timing is unavailable from imageio's iterator boundary.
    if frame_count == 0:
        raise RuntimeError("video contains no decoded frames")
    retained = buffer.retained()
    if last_processed_start is None:
        start = int(retained.frame_indices[0])
        valid_frames = retained.frames.shape[0]
        pad = adapter_config.window_len - valid_frames
        interval = 1 / fps
        frames = np.concatenate(
            [retained.frames, np.repeat(retained.frames[-1:], pad, axis=0)], axis=0
        )
        timestamps = np.concatenate(
            [
                retained.timestamps,
                retained.timestamps[-1] + interval * np.arange(1, pad + 1),
            ]
        )
        indices = np.arange(start, start + adapter_config.window_len)
        tracker_start = time.perf_counter()
        batches.append(
            adapter.process_window(
                frames, timestamps, indices, final=True, valid_frames=valid_frames
            )
        )
        tracker_ms = (time.perf_counter() - tracker_start) * 1000
        timing.append(
            TimingRecord(
                timestamp=float(timestamps[0]),
                block_index=0,
                preprocess_ms=float(np.sum(recent_preprocess_ms)),
                tracker_ms=tracker_ms,
                total_pipeline_ms=tracker_ms + float(np.sum(recent_preprocess_ms)),
            )
        )
        memory.append(sample_memory(float(timestamps[0]), 0))
    else:
        last_window_end = last_processed_start + adapter_config.window_len
        if frame_count > last_window_end:
            next_start = last_processed_start + adapter_config.step
            selection = retained.frame_indices >= next_start
            tail_frames = retained.frames[selection]
            tail_times = retained.timestamps[selection]
            valid_frames = tail_frames.shape[0]
            pad = adapter_config.window_len - valid_frames
            interval = 1 / fps
            frames = np.concatenate(
                [tail_frames, np.repeat(tail_frames[-1:], pad, axis=0)], axis=0
            )
            timestamps = np.concatenate(
                [tail_times, tail_times[-1] + interval * np.arange(1, pad + 1)]
            )
            indices = np.arange(next_start, next_start + adapter_config.window_len)
            tracker_start = time.perf_counter()
            batches.append(
                adapter.process_window(
                    frames, timestamps, indices, final=True, valid_frames=valid_frames
                )
            )
            tracker_ms = (time.perf_counter() - tracker_start) * 1000
            timing.append(
                TimingRecord(
                    timestamp=float(timestamps[0]),
                    block_index=len(timing),
                    preprocess_ms=float(np.sum(recent_preprocess_ms[-valid_frames:])),
                    tracker_ms=tracker_ms,
                    total_pipeline_ms=tracker_ms
                    + float(np.sum(recent_preprocess_ms[-valid_frames:])),
                )
            )
            memory.append(sample_memory(float(timestamps[0]), len(timing) - 1))
        else:
            pending = adapter.flush_pending()
            if pending is not None:
                batches.append(pending)
    tracks = concatenate_track_batches(batches)
    if tracks.num_frames != frame_count:
        raise RuntimeError(
            f"tracker output covers {tracks.num_frames} frames, decoded {frame_count}"
        )
    motion_start = time.perf_counter()
    motion_config = config["motion"]
    thresholds = QualityThresholds(
        min_valid_tracks=int(motion_config["min_valid_tracks"]),
        min_inlier_ratio=float(motion_config["min_inlier_ratio"]),
        min_spatial_coverage=float(motion_config["min_spatial_coverage"]),
        max_fit_rmse_px=float(motion_config["max_fit_rmse_px"]),
    )
    motion = decompose_tracks(
        tracks.query_xy_px,
        tracks.xy_px,
        tracks.visible,
        model=str(motion_config["global_model"]),
        use_ransac=bool(motion_config["use_ransac"]),
        ransac_threshold_px=float(motion_config["ransac_threshold_px"]),
        frame_size=model_size,
        quality_thresholds=thresholds,
    )
    motion_ms = (time.perf_counter() - motion_start) * 1000
    signal_start = time.perf_counter()
    common_xy = motion.common_parameters[:, :2]
    effective_fps = 1 / np.median(np.diff(tracks.timestamps))
    bandpass = tuple(float(value) for value in signal_config["bandpass_hz"])
    filtered = bandpass_filter(
        common_xy,
        effective_fps,
        bandpass,
        causal=bool(signal_config["causal"]),
        apply_detrend=bool(signal_config["detrend"]),
    )
    acceleration = local_polynomial_derivative(
        tracks.timestamps,
        filtered.values,
        derivative_order=2,
        causal=bool(signal_config["causal"]),
    )
    features = extract_motion_features(
        tracks.timestamps,
        filtered.values,
        motion.residual_xy_px,
        tracks.visible,
        inlier_ratio=motion.common_parameters[:, 5],
        fit_rmse_px=motion.common_parameters[:, 6],
        quality_flags=np.unique(motion.quality),
        derivative_method=str(signal_config["derivative_method"]),
        causal=bool(signal_config["causal"]),
    )
    signal_ms = (time.perf_counter() - signal_start) * 1000
    if timing:
        # Motion decomposition and offline signal processing operate on the full
        # sequence. Report their amortized per-block cost so percentile summaries
        # do not contain one large terminal spike and many artificial zeros.
        motion_per_block = motion_ms / len(timing)
        signal_per_block = signal_ms / len(timing)
        timing = [
            TimingRecord(
                **{
                    **record.__dict__,
                    "motion_fit_ms": motion_per_block,
                    "signal_ms": signal_per_block,
                    "total_pipeline_ms": record.total_pipeline_ms
                    + motion_per_block
                    + signal_per_block,
                }
            )
            for record in timing
        ]
    return OfflinePipelineResult(
        tracks=tracks,
        motion=motion,
        filtered_common_xy_px=filtered.values,
        acceleration_proxy_px_s2=acceleration,
        features=features,
        timing=tuple(timing),
        memory=tuple(memory),
        events=tuple(events),
        video_metadata={
            "path": str(Path(video_path).resolve()),
            "fps": fps,
            "timestamp_source": timestamp_source,
            "source_shape": source_shape,
            "processed_shape": model_size,
            "decode_resize_backend": (
                "ffmpeg_decode_time_resize" if direct_decode_resize else "post_decode_resize"
            ),
            "frame_count": frame_count,
            "source_metadata": {
                key: value
                for key, value in source_metadata.items()
                if isinstance(value, (str, int, float, bool, type(None)))
            },
            "queue_bound_blocks": int(runtime_config["max_queue_blocks"]),
            "checkpoint_path": str(Path(checkpoint).resolve()),
            "cotracker_root": str(Path(cotracker_root).resolve()),
            "device": device,
            "run_start_utc": run_start_utc,
        },
    )


def write_run_artifacts(
    run_directory: str | Path,
    result: OfflinePipelineResult,
    config: dict[str, Any],
) -> None:
    repository = Path(__file__).resolve().parents[2]
    try:
        project_git = git_state(repository)
    except (OSError, subprocess.SubprocessError):
        project_git = {"commit": "unknown", "dirty": True, "status": []}
    try:
        project_diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD"],
            cwd=repository,
            check=True,
            text=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        project_diff = ""
    output = Path(run_directory)
    output.mkdir(parents=True, exist_ok=True)
    result.tracks.save_npz(output / "tracks.npz")
    np.savez_compressed(
        output / "residual_motion.npz",
        timestamps=result.tracks.timestamps,
        residual_xy_px=result.motion.residual_xy_px,
        inlier_mask=result.motion.inlier_mask,
    )
    np.savez_compressed(
        output / "filtered_signals.npz",
        timestamps=result.tracks.timestamps,
        filtered_common_xy_px=result.filtered_common_xy_px,
        acceleration_proxy_px_s2=result.acceleration_proxy_px_s2,
    )
    with (output / "common_motion.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "timestamp",
            "tx_px",
            "ty_px",
            "rotation_2d_rad",
            "scale",
            "shear",
            "inlier_ratio",
            "fit_rmse_px",
            "spatial_coverage",
            "motion_quality",
        ]
        writer = csv.writer(handle)
        writer.writerow(fields)
        for timestamp, parameters, quality in zip(
            result.tracks.timestamps, result.motion.common_parameters, result.motion.quality
        ):
            writer.writerow([timestamp, *parameters[:8], quality])
    with (output / "motion_quality.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "quality", "reasons"])
        for timestamp, quality, reasons in zip(
            result.tracks.timestamps, result.motion.quality, result.motion.quality_reasons
        ):
            writer.writerow([timestamp, quality, "|".join(reasons)])
    with (output / "features.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result.features))
        writer.writeheader()
        writer.writerow(result.features)
    write_timing_csv(output / "timing.csv", list(result.timing))
    write_memory_csv(output / "memory.csv", list(result.memory))
    with (output / "queue.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "timestamp",
                "capture_queue_depth",
                "tracker_queue_depth",
                "output_queue_depth",
                "dropped_frames",
                "dropped_blocks",
                "overload_state",
            ]
        )
        writer.writerow(
            [
                result.tracks.timestamps[-1],
                0,
                0,
                0,
                0,
                0,
                "OFFLINE_BOUNDED_WINDOWS",
            ]
        )
    with (output / "tracks_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["point_id", "visible_fraction", "start_x_px", "start_y_px"])
        for point_index, point_id in enumerate(result.tracks.point_ids):
            writer.writerow(
                [
                    int(point_id),
                    float(np.mean(result.tracks.visible[:, point_index])),
                    *result.tracks.query_xy_px[point_index],
                ]
            )
    (output / "events.jsonl").write_text(
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in result.events),
        encoding="utf-8",
    )
    (output / "input_manifest.json").write_text(
        json.dumps(result.video_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    checkpoint_path = Path(str(result.video_metadata["checkpoint_path"]))
    cotracker_path = Path(str(result.video_metadata["cotracker_root"]))
    try:
        cotracker_git = git_state(cotracker_path)
    except (OSError, subprocess.SubprocessError):
        cotracker_git = {"commit": "unknown", "dirty": True, "status": []}
    environment = environment_snapshot()
    (output / "environment.txt").write_text(
        json.dumps(environment, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "device_info.txt").write_text(
        json.dumps(
            {
                "requested_device": result.video_metadata["device"],
                "platform": environment["platform"],
                "machine": environment["machine"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "git_status.txt").write_text(
        "\n".join(project_git.get("status", [])) + "\n", encoding="utf-8"
    )
    (output / "git_diff.patch").write_text(project_diff, encoding="utf-8")
    manifest = {
        "run_id": output.name,
        "git_commit": project_git.get("commit", "unknown"),
        "git_dirty": bool(project_git.get("dirty", True)),
        "cotracker_commit": cotracker_git.get("commit", "unknown"),
        "cotracker_dirty": bool(cotracker_git.get("dirty", True)),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "config_sha256": config_sha256(config),
        "input_id": result.video_metadata["path"],
        "start_time": result.video_metadata["run_start_utc"],
        "device": result.video_metadata["device"],
        "software_versions": environment["software_versions"],
        "tracker_parameters": config["tracker"],
        "motion_parameters": config["motion"],
        "scale_parameters": config["scale"],
        "signal_parameters": config["signal"],
        "pga_model_version": None,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metrics = {
        "timing": summarize_timings(list(result.timing)),
        "motion_quality_counts": {
            quality: int(np.sum(result.motion.quality == quality))
            for quality in np.unique(result.motion.quality)
        },
        "scale_state": ScaleCalibration.uncalibrated().state.value,
        "peak_rss_mb": max((sample.peak_rss_mb for sample in result.memory), default=None),
        "pga_est": None,
        "pga_rejection_reason": "scale_invalid_and_model_not_trained",
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
