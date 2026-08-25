"""Bounded three-worker realtime capture, tracking and audit writer."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from threading import Event, Thread
import time
from typing import Any, Sequence

import numpy as np
import yaml

from seismic_motion.config import config_sha256
from seismic_motion.diagnostics.provenance import environment_snapshot, git_state, sha256_file
from seismic_motion.motion.global_motion import apply_transform, fit_global_transform
from seismic_motion.motion.quality import QualityThresholds, assess_motion_quality
from seismic_motion.pga.online import RunningPGAEstimator
from seismic_motion.signal.online import OnlineSignalProcessor
from seismic_motion.tracking.cotracker_adapter import CoTrackerAdapter, CoTrackerAdapterConfig
from seismic_motion.tracking.online_buffer import (
    AuditedBoundedQueue,
    BufferOverload,
    SlidingFrameBuffer,
)
from seismic_motion.tracking.types import TrackBatch

from .pipeline import _preprocess_frame
from .metrics import (
    MemoryRecord,
    TimingRecord,
    sample_memory,
    summarize_timings,
    write_memory_csv,
    write_timing_csv,
)


@dataclass(frozen=True)
class CapturedFrame:
    frame_rgb: np.ndarray
    timestamp: float
    frame_index: int
    capture_monotonic: float
    enqueue_monotonic: float
    source_timestamp: float
    source_path: str
    playlist_index: int = 0
    loop_index: int = 0
    source_boundary: bool = False


@dataclass(frozen=True)
class ProcessedBlock:
    batch: TrackBatch
    capture_timestamp: float
    enqueue_timestamp: float
    window_ready_timestamp: float
    tracker_start: float
    tracker_end: float
    source_path: str
    playlist_index: int
    loop_index: int
    source_boundary: bool


class RealtimeRunner:
    """Run finite bounded queues; overload stops capture instead of silent loss."""

    def __init__(
        self,
        source: str | int | Sequence[str],
        config: dict[str, Any],
        *,
        cotracker_root: str,
        checkpoint: str,
        device: str,
        output_directory: str | Path,
    ) -> None:
        tracker = config["tracker"]
        queue_bound = int(config["runtime"]["max_queue_blocks"])
        self.source = list(source) if isinstance(source, (list, tuple)) else source
        self.config = config
        self.cotracker_root = str(Path(cotracker_root).resolve())
        self.checkpoint = str(Path(checkpoint).resolve())
        self.device = device
        self.run_start_utc = datetime.now(timezone.utc).isoformat()
        self.output = Path(output_directory)
        self.output.mkdir(parents=True, exist_ok=True)
        self.capture_queue: AuditedBoundedQueue[CapturedFrame | None] = AuditedBoundedQueue(
            maxsize=max(queue_bound * int(tracker["step"]), int(tracker["window_len"])),
            name="capture",
        )
        self.output_queue: AuditedBoundedQueue[ProcessedBlock | None] = AuditedBoundedQueue(
            maxsize=queue_bound, name="output"
        )
        self.stop_event = Event()
        self.events: list[dict[str, object]] = []
        self.timing_records: list[TimingRecord] = []
        self.frames_written = 0
        self.adapter = CoTrackerAdapter(
            CoTrackerAdapterConfig(
                cotracker_root=cotracker_root,
                checkpoint=checkpoint,
                device=device,
                num_points=int(tracker["num_points"]),
                window_len=int(tracker["window_len"]),
                step=int(tracker["step"]),
                iters=int(tracker["iters"]),
                point_mode=str(tracker["point_mode"]),
                max_blocks_before_reseed=int(tracker.get("max_blocks_before_reseed", 64)),
            ),
            event_sink=self._record_event,
        )

    def _record_event(self, event: dict[str, object]) -> None:
        payload = {"monotonic_s": time.monotonic(), **event}
        self.events.append(payload)

    def _capture_worker(self, duration_s: float | None) -> None:
        try:
            import cv2
        except ImportError:  # File replay falls back to imageio below.
            cv2 = None  # type: ignore[assignment]
        sources: list[str | int] = (
            list(self.source) if isinstance(self.source, list) else [self.source]
        )
        target_fps = float(self.config["runtime"]["target_fps"])
        playlist_loop = bool(self.config["runtime"].get("playlist_loop", False))
        wall_start = time.monotonic()
        frame_index = 0
        loop_index = 0
        completed = False
        while not completed and not self.stop_event.is_set():
            for playlist_index, source in enumerate(sources):
                if self.stop_event.is_set():
                    break
                capture = None
                iterator = None
                if cv2 is not None:
                    capture = cv2.VideoCapture(source)
                    if not capture.isOpened():
                        self._record_event(
                            {
                                "event": "capture_error",
                                "reason": "source_open_failed",
                                "source_path": str(source),
                            }
                        )
                        self.stop_event.set()
                        break
                    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
                elif isinstance(source, str):
                    import imageio.v3 as iio

                    metadata = iio.immeta(source, plugin="FFMPEG")
                    source_fps = float(metadata.get("fps", 0.0))
                    iterator = iter(iio.imiter(source, plugin="FFMPEG"))
                else:
                    self._record_event(
                        {
                            "event": "capture_error",
                            "reason": "opencv_required_for_camera_source",
                            "source_path": str(source),
                        }
                    )
                    self.stop_event.set()
                    break
                if source_fps <= 0:
                    source_fps = target_fps
                source_frame_index = 0
                next_emit_source_s = 0.0
                emitted_in_source = 0
                self._record_event(
                    {
                        "event": "playlist_source_start",
                        "source_path": str(source),
                        "playlist_index": playlist_index,
                        "loop_index": loop_index,
                        "source_fps": source_fps,
                        "target_fps": target_fps,
                    }
                )
                while not self.stop_event.is_set():
                    if duration_s is not None and time.monotonic() - wall_start >= duration_s:
                        completed = True
                        break
                    if capture is not None:
                        ok, frame_bgr = capture.read()
                        if not ok:
                            break
                        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    else:
                        assert iterator is not None
                        try:
                            frame_rgb = np.asarray(next(iterator))
                        except StopIteration:
                            break
                    source_timestamp = source_frame_index / source_fps
                    source_frame_index += 1
                    if source_timestamp + 0.5 / source_fps < next_emit_source_s:
                        continue
                    next_emit_source_s += 1.0 / target_fps
                    deadline = wall_start + frame_index / target_fps
                    remaining = deadline - time.monotonic()
                    if remaining > 0:
                        time.sleep(remaining)
                    captured = time.monotonic()
                    boundary = emitted_in_source == 0 and frame_index > 0
                    frame = CapturedFrame(
                        frame_rgb=frame_rgb,
                        timestamp=frame_index / target_fps,
                        frame_index=frame_index,
                        capture_monotonic=captured,
                        enqueue_monotonic=time.monotonic(),
                        source_timestamp=source_timestamp,
                        source_path=str(source),
                        playlist_index=playlist_index,
                        loop_index=loop_index,
                        source_boundary=boundary,
                    )
                    try:
                        self.capture_queue.put(frame)
                    except BufferOverload:
                        self._record_event(
                            {
                                "event": "overload",
                                "stage": "capture_queue",
                                "frame_index": frame_index,
                                "action": "stop_without_silent_drop",
                            }
                        )
                        self.stop_event.set()
                        break
                    frame_index += 1
                    emitted_in_source += 1
                if capture is not None:
                    capture.release()
                self._record_event(
                    {
                        "event": "playlist_source_end",
                        "source_path": str(source),
                        "playlist_index": playlist_index,
                        "loop_index": loop_index,
                        "emitted_frames": emitted_in_source,
                    }
                )
                if completed:
                    break
            if completed or not playlist_loop or not all(isinstance(item, str) for item in sources):
                completed = True
            else:
                loop_index += 1
        while not self.stop_event.is_set():
            try:
                self.capture_queue.put(None, timeout=0.1)
                break
            except BufferOverload:
                continue

    def _queue_processed_block(
        self,
        batch: TrackBatch,
        captures: dict[int, CapturedFrame],
        ready_frame: CapturedFrame,
        tracker_start: float,
        tracker_end: float,
        source_boundary: bool,
    ) -> bool:
        output_frame = captures.get(int(batch.frame_indices[-1]), ready_frame)
        block = ProcessedBlock(
            batch=batch,
            capture_timestamp=output_frame.capture_monotonic,
            enqueue_timestamp=ready_frame.enqueue_monotonic,
            window_ready_timestamp=ready_frame.capture_monotonic,
            tracker_start=tracker_start,
            tracker_end=tracker_end,
            source_path=output_frame.source_path,
            playlist_index=output_frame.playlist_index,
            loop_index=output_frame.loop_index,
            source_boundary=source_boundary,
        )
        try:
            self.output_queue.put(block, timeout=0.1)
        except BufferOverload:
            self._record_event(
                {
                    "event": "overload",
                    "stage": "output_queue",
                    "frame_index": int(batch.frame_indices[0]),
                    "action": "stop_without_silent_drop",
                }
            )
            self.stop_event.set()
            return False
        for frame_index in batch.frame_indices:
            captures.pop(int(frame_index), None)
        memory = sample_memory(tracker_end, int(batch.frame_indices[0]))
        self._record_event(
            {
                "monotonic_s": tracker_end,
                "event": "tracker_block",
                "frame_index": int(batch.frame_indices[0]),
                "capture_timestamp": output_frame.capture_monotonic,
                "enqueue_timestamp": ready_frame.enqueue_monotonic,
                "tracker_start": tracker_start,
                "tracker_end": tracker_end,
                "tracker_ms": (tracker_end - tracker_start) * 1000,
                "end_to_end_latency_ms": (tracker_end - output_frame.capture_monotonic) * 1000,
                "capture_queue_depth": self.capture_queue.depth,
                "output_queue_depth": self.output_queue.depth,
                "capture_rejected_items": self.capture_queue.rejected_items,
                "output_rejected_items": self.output_queue.rejected_items,
                "rss_mb": memory.rss_mb,
                "peak_rss_mb": memory.peak_rss_mb,
                "system_available_mem_mb": memory.system_available_mem_mb,
                "source_path": output_frame.source_path,
                "playlist_index": output_frame.playlist_index,
                "loop_index": output_frame.loop_index,
                "source_boundary": source_boundary,
            }
        )
        return True

    def _finalize_segment(
        self,
        frame_buffer: SlidingFrameBuffer,
        last_processed_start: int | None,
        segment_frames: int,
        captures: dict[int, CapturedFrame],
        ready_frame: CapturedFrame | None,
        source_boundary: bool,
    ) -> None:
        if not segment_frames or ready_frame is None:
            return
        tracker = self.config["tracker"]
        retained = frame_buffer.retained()
        last_seen = int(retained.frame_indices[-1])
        last_window_end = -1 if last_processed_start is None else (
            last_processed_start + int(tracker["window_len"]) - 1
        )
        if last_processed_start is None or last_seen > last_window_end:
            next_start = (
                int(retained.frame_indices[0])
                if last_processed_start is None
                else last_processed_start + int(tracker["step"])
            )
            selection = retained.frame_indices >= next_start
            tail_frames = retained.frames[selection]
            tail_times = retained.timestamps[selection]
            valid_frames = int(tail_frames.shape[0])
            pad = int(tracker["window_len"]) - valid_frames
            interval = (
                float(np.median(np.diff(tail_times)))
                if tail_times.size >= 2
                else 1 / float(self.config["runtime"]["target_fps"])
            )
            frames = np.concatenate(
                [tail_frames, np.repeat(tail_frames[-1:], pad, axis=0)], axis=0
            )
            timestamps = np.concatenate(
                [tail_times, tail_times[-1] + interval * np.arange(1, pad + 1)]
            )
            indices = np.arange(next_start, next_start + int(tracker["window_len"]))
            tracker_start = time.monotonic()
            batch = self.adapter.process_window(
                frames, timestamps, indices, final=True, valid_frames=valid_frames
            )
            tracker_end = time.monotonic()
            self._queue_processed_block(
                batch,
                captures,
                ready_frame,
                tracker_start,
                tracker_end,
                source_boundary,
            )
        else:
            tracker_start = time.monotonic()
            pending = self.adapter.flush_pending()
            tracker_end = time.monotonic()
            if pending is not None:
                self._queue_processed_block(
                    pending,
                    captures,
                    ready_frame,
                    tracker_start,
                    tracker_end,
                    source_boundary,
                )

    def _tracker_worker(self) -> None:
        tracker = self.config["tracker"]
        video = self.config["video"]
        frame_buffer = SlidingFrameBuffer(int(tracker["window_len"]), int(tracker["step"]))
        last_processed_start: int | None = None
        segment_frames = 0
        captures: dict[int, CapturedFrame] = {}
        last_captured: CapturedFrame | None = None
        first_block_in_segment = True
        while not self.stop_event.is_set():
            try:
                captured = self.capture_queue.get(timeout=0.1)
            except LookupError:
                continue
            self.capture_queue.task_done()
            if captured is None:
                self._finalize_segment(
                    frame_buffer,
                    last_processed_start,
                    segment_frames,
                    captures,
                    last_captured,
                    first_block_in_segment,
                )
                self.output_queue.put(None, timeout=0.1)
                return
            if captured.source_boundary and segment_frames:
                self._finalize_segment(
                    frame_buffer,
                    last_processed_start,
                    segment_frames,
                    captures,
                    last_captured,
                    first_block_in_segment,
                )
                self.adapter.reset_stream(captured.frame_index, reason="playlist_source_boundary")
                frame_buffer = SlidingFrameBuffer(
                    int(tracker["window_len"]), int(tracker["step"])
                )
                last_processed_start = None
                segment_frames = 0
                captures.clear()
                first_block_in_segment = True
            captures[captured.frame_index] = captured
            last_captured = captured
            processed = _preprocess_frame(
                captured.frame_rgb,
                video.get("roi"),
                tuple(int(value) for value in tracker["model_resolution"]),
            )
            window = frame_buffer.append(processed, captured.timestamp, captured.frame_index)
            segment_frames += 1
            if window is None:
                continue
            tracker_start = time.monotonic()
            batch = self.adapter.process_window(
                window.frames, window.timestamps, window.frame_indices
            )
            tracker_end = time.monotonic()
            last_processed_start = int(window.frame_indices[0])
            if not self._queue_processed_block(
                batch,
                captures,
                captured,
                tracker_start,
                tracker_end,
                first_block_in_segment,
            ):
                return
            first_block_in_segment = False

    def _writer_worker(self) -> None:
        motion = self.config["motion"]
        thresholds = QualityThresholds(
            min_valid_tracks=int(motion["min_valid_tracks"]),
            min_inlier_ratio=float(motion["min_inlier_ratio"]),
            min_spatial_coverage=float(motion["min_spatial_coverage"]),
            max_fit_rmse_px=float(motion["max_fit_rmse_px"]),
        )
        signal_config = self.config["signal"]
        online_config = signal_config.get("online", {})
        pga_config = self.config["pga"]

        def make_signal_processor() -> OnlineSignalProcessor:
            return OnlineSignalProcessor(
                sample_rate_hz=float(self.config["runtime"]["target_fps"]),
                bandpass_hz=tuple(float(value) for value in signal_config["bandpass_hz"]),
                filter_order=int(online_config.get("filter_order", 4)),
                derivative_method=str(
                    online_config.get(
                        "derivative_method",
                        signal_config.get("derivative_method", "causal_polynomial"),
                    )
                ),
                window_length=int(online_config.get("window_length", 9)),
                polynomial_order=int(online_config.get("polynomial_order", 3)),
            )

        coefficient: float | None = None
        model_path = pga_config.get("model_path")
        if model_path:
            payload = json.loads(Path(str(model_path)).read_text(encoding="utf-8"))
            values = payload.get("coefficient")
            if isinstance(values, list) and len(values) == 1:
                coefficient = float(values[0])

        def make_pga_estimator() -> RunningPGAEstimator:
            return RunningPGAEstimator(
                coefficient,
                allow_uncalibrated_research=bool(
                    pga_config.get("allow_uncalibrated_research", False)
                ),
            )

        signal_processor = make_signal_processor()
        pga_estimator = make_pga_estimator()
        scale_valid = str(self.config["scale"].get("method", "uncalibrated")) != "uncalibrated"
        with (self.output / "tracks.csv").open("w", newline="", encoding="utf-8") as track_handle, (
            self.output / "common_motion.csv"
        ).open("w", newline="", encoding="utf-8") as common_handle, (
            self.output / "online_signal.csv"
        ).open("w", newline="", encoding="utf-8") as signal_handle, (
            self.output / "running_pga.csv"
        ).open("w", newline="", encoding="utf-8") as pga_handle:
            track_writer = csv.writer(track_handle)
            track_writer.writerow(
                ["timestamp", "frame_index", "point_id", "x_px", "y_px", "visible", "reseed_id"]
            )
            common_writer = csv.writer(common_handle)
            common_writer.writerow(
                [
                    "timestamp",
                    "frame_index",
                    "tx_px",
                    "ty_px",
                    "rotation_2d_rad",
                    "scale",
                    "inlier_ratio",
                    "fit_rmse_px",
                    "motion_quality",
                    "quality_reasons",
                ]
            )
            signal_fields = [
                "timestamp",
                "filtered_common_x",
                "filtered_common_y",
                "common_velocity_x",
                "common_velocity_y",
                "common_acceleration_x",
                "common_acceleration_y",
                "local_motion_rms",
                "local_velocity_rms",
                "local_acceleration_rms",
                "quality_state",
                "derivative_method",
                "startup_state",
                "samples_seen",
                "retained_history_samples",
                "filter_state_bytes",
                "reseed_id",
                "reseed_boundary",
            ]
            signal_writer = csv.DictWriter(signal_handle, fieldnames=signal_fields)
            signal_writer.writeheader()
            pga_fields = [
                "timestamp",
                "acceleration_proxy_instant_px_s2",
                "acceleration_proxy_running_px_s2",
                "pga_instant_est_gal",
                "pga_running_est_gal",
                "confidence",
                "scale_state",
                "deployment_prediction_allowed",
                "interpretation",
            ]
            pga_writer = csv.DictWriter(pga_handle, fieldnames=pga_fields)
            pga_writer.writeheader()
            previous = None
            while not self.stop_event.is_set():
                try:
                    batch = self.output_queue.get(timeout=0.1)
                except LookupError:
                    continue
                self.output_queue.task_done()
                if batch is None:
                    break
                block = batch
                tracks = block.batch
                if block.source_boundary and self.timing_records:
                    signal_processor = make_signal_processor()
                    pga_estimator = make_pga_estimator()
                    previous = None
                motion_rows: list[tuple[object | None, object | None, np.ndarray | None]] = []
                for local_index in range(tracks.num_frames):
                    try:
                        estimate = fit_global_transform(
                            tracks.query_xy_px,
                            tracks.xy_px[local_index],
                            tracks.visible[local_index],
                            model=str(motion["global_model"]),
                            use_ransac=bool(motion["use_ransac"]),
                            ransac_threshold_px=float(motion["ransac_threshold_px"]),
                            frame_size=tuple(self.config["tracker"]["model_resolution"]),
                        )
                        decision = assess_motion_quality(estimate, thresholds, previous)
                        predicted = apply_transform(estimate.matrix, tracks.query_xy_px)
                        residual = tracks.xy_px[local_index].astype(np.float64) - predicted
                        residual[~tracks.visible[local_index]] = 0.0
                        motion_rows.append((estimate, decision, residual))
                        if decision.quality.value != "INVALID":
                            previous = estimate
                    except ValueError:
                        motion_rows.append((None, None, None))
                motion_end = time.monotonic()

                signal_states = []
                for local_index, (estimate, decision, residual) in enumerate(motion_rows):
                    if estimate is None or decision is None or residual is None:
                        signal_states.append(None)
                        continue
                    signal_states.append(
                        signal_processor.update(
                            float(tracks.timestamps[local_index]),
                            np.asarray([estimate.tx_px, estimate.ty_px]),
                            residual,
                            decision.quality.value,
                            reseed_id=tracks.reseed_id,
                        )
                    )
                signal_end = time.monotonic()

                pga_states = []
                for state in signal_states:
                    magnitude = (
                        float(
                            np.hypot(
                                state.common_acceleration_x,
                                state.common_acceleration_y,
                            )
                        )
                        if state is not None
                        else float("nan")
                    )
                    pga_states.append(
                        pga_estimator.update(
                            float(
                                state.timestamp
                                if state is not None
                                else tracks.timestamps[len(pga_states)]
                            ),
                            magnitude,
                            quality=state.quality_state if state is not None else "INVALID",
                            scale_valid=scale_valid,
                        )
                    )
                pga_end = time.monotonic()

                for local_index, (timestamp, frame_index) in enumerate(
                    zip(tracks.timestamps, tracks.frame_indices)
                ):
                    for point_index, point_id in enumerate(tracks.point_ids):
                        track_writer.writerow(
                            [
                                timestamp,
                                frame_index,
                                point_id,
                                *tracks.xy_px[local_index, point_index],
                                int(tracks.visible[local_index, point_index]),
                                tracks.reseed_id,
                            ]
                        )
                    estimate, decision, _ = motion_rows[local_index]
                    if estimate is None or decision is None:
                        common_writer.writerow(
                            [timestamp, frame_index, "", "", "", "", "", "", "INVALID", "fit_failed"]
                        )
                    else:
                        common_writer.writerow(
                            [
                                timestamp,
                                frame_index,
                                estimate.tx_px,
                                estimate.ty_px,
                                estimate.rotation_2d_rad,
                                estimate.scale,
                                estimate.inlier_ratio,
                                estimate.fit_rmse_px,
                                decision.quality.value,
                                "|".join(decision.reasons),
                            ]
                        )
                    state = signal_states[local_index]
                    if state is None:
                        signal_writer.writerow(
                            {
                                "timestamp": timestamp,
                                "quality_state": "INVALID",
                                "derivative_method": signal_processor.derivative_method,
                                "startup_state": "FIT_FAILED",
                                "reseed_id": tracks.reseed_id,
                                "reseed_boundary": False,
                            }
                        )
                    else:
                        signal_writer.writerow(state.as_dict())
                    pga_writer.writerow(pga_states[local_index].as_dict())
                for handle in (track_handle, common_handle, signal_handle, pga_handle):
                    handle.flush()
                self.frames_written += tracks.num_frames
                write_end = time.monotonic()
                record = TimingRecord(
                    timestamp=write_end,
                    block_index=len(self.timing_records),
                    capture_timestamp=block.capture_timestamp,
                    enqueue_timestamp=block.enqueue_timestamp,
                    tracker_start=block.tracker_start,
                    tracker_end=block.tracker_end,
                    motion_end=motion_end,
                    signal_end=signal_end,
                    pga_end=pga_end,
                    write_end=write_end,
                    capture_to_tracker_end=(block.tracker_end - block.capture_timestamp) * 1000,
                    capture_to_pga=(pga_end - block.capture_timestamp) * 1000,
                    queue_wait=(block.tracker_start - block.enqueue_timestamp) * 1000,
                    tracker_compute=(block.tracker_end - block.tracker_start) * 1000,
                    postprocess_compute=(write_end - block.tracker_end) * 1000,
                    tracker_ms=(block.tracker_end - block.tracker_start) * 1000,
                    motion_fit_ms=(motion_end - block.tracker_end) * 1000,
                    signal_ms=(signal_end - motion_end) * 1000,
                    pga_ms=(pga_end - signal_end) * 1000,
                    total_pipeline_ms=(write_end - block.tracker_start) * 1000,
                    end_to_end_latency_ms=(pga_end - block.capture_timestamp) * 1000,
                )
                self.timing_records.append(record)
                memory = sample_memory(write_end, record.block_index)
                self._record_event(
                    {
                        "event": "pipeline_block",
                        "block_index": record.block_index,
                        "frame_index": int(tracks.frame_indices[0]),
                        "capture_timestamp": block.capture_timestamp,
                        "enqueue_timestamp": block.enqueue_timestamp,
                        "tracker_start": block.tracker_start,
                        "tracker_end": block.tracker_end,
                        "motion_end": motion_end,
                        "signal_end": signal_end,
                        "pga_end": pga_end,
                        "write_end": write_end,
                        "capture_queue_depth": self.capture_queue.depth,
                        "block_queue_depth": self.output_queue.depth,
                        "writer_queue_depth": 0,
                        "capture_rejected_items": self.capture_queue.rejected_items,
                        "block_rejected_items": self.output_queue.rejected_items,
                        "rss_mb": memory.rss_mb,
                        "peak_rss_mb": memory.peak_rss_mb,
                        "system_available_mem_mb": memory.system_available_mem_mb,
                        "source_path": block.source_path,
                        "playlist_index": block.playlist_index,
                        "loop_index": block.loop_index,
                        "source_boundary": block.source_boundary,
                    }
                )

    def run(self, duration_s: float | None = None) -> dict[str, object]:
        workers = [
            Thread(target=self._capture_worker, args=(duration_s,), name="capture"),
            Thread(target=self._tracker_worker, name="tracker"),
            Thread(target=self._writer_worker, name="writer"),
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
        (self.output / "events.jsonl").write_text(
            "".join(json.dumps(event, sort_keys=True) + "\n" for event in self.events),
            encoding="utf-8",
        )
        queue_metrics = {
            "capture": self.capture_queue.metrics(),
            "output": self.output_queue.metrics(),
        }
        (self.output / "queue_summary.json").write_text(
            json.dumps(queue_metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        tracker_events = [event for event in self.events if event.get("event") == "tracker_block"]
        pipeline_events = [event for event in self.events if event.get("event") == "pipeline_block"]
        write_timing_csv(self.output / "timing.csv", self.timing_records)
        write_timing_csv(self.output / "runtime_timing.csv", self.timing_records)
        memory_records: list[MemoryRecord] = [
            MemoryRecord(
                timestamp=float(event.get("write_end", event.get("monotonic_s", 0.0))),
                block_index=index,
                rss_mb=float(event["rss_mb"]),
                peak_rss_mb=float(event["peak_rss_mb"]),
                system_available_mem_mb=float(event["system_available_mem_mb"]),
            )
            for index, event in enumerate(pipeline_events or tracker_events)
        ]
        if not memory_records:
            memory_records = [sample_memory(time.monotonic(), 0)]
        write_memory_csv(self.output / "memory.csv", memory_records)
        write_memory_csv(self.output / "runtime_memory.csv", memory_records)
        queue_fields = [
            "timestamp",
            "frame_queue_depth",
            "block_queue_depth",
            "writer_queue_depth",
            "capture_queue_depth",
            "tracker_queue_depth",
            "output_queue_depth",
            "dropped_frames",
            "dropped_blocks",
            "overload_state",
        ]
        queue_rows = []
        for event in pipeline_events or tracker_events:
            rejected_capture = int(event.get("capture_rejected_items", 0))
            rejected_output = int(
                event.get("block_rejected_items", event.get("output_rejected_items", 0))
            )
            capture_depth = int(event.get("capture_queue_depth", 0))
            output_depth = int(
                event.get("block_queue_depth", event.get("output_queue_depth", 0))
            )
            queue_rows.append(
                {
                    "timestamp": event.get("write_end", event.get("monotonic_s", 0.0)),
                    "frame_queue_depth": capture_depth,
                    "block_queue_depth": output_depth,
                    "writer_queue_depth": int(event.get("writer_queue_depth", 0)),
                    "capture_queue_depth": capture_depth,
                    "tracker_queue_depth": 0,
                    "output_queue_depth": output_depth,
                    "dropped_frames": rejected_capture,
                    "dropped_blocks": rejected_output,
                    "overload_state": (
                        "OVERLOAD" if rejected_capture or rejected_output else "NORMAL"
                    ),
                }
            )
        for queue_name in ("queue.csv", "runtime_queue.csv"):
            with (self.output / queue_name).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=queue_fields)
                writer.writeheader()
                writer.writerows(queue_rows)
        event_fields = sorted({key for event in self.events for key in event}) or ["event"]
        with (self.output / "runtime_events.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=event_fields)
            writer.writeheader()
            for event in self.events:
                writer.writerow(
                    {
                        key: (
                            json.dumps(value, sort_keys=True)
                            if isinstance(value, (dict, list, tuple))
                            else value
                        )
                        for key, value in event.items()
                    }
                )
        yaml_path = self.output / "config.yaml"
        yaml_path.write_text(yaml.safe_dump(self.config, sort_keys=False), encoding="utf-8")
        repository = Path(__file__).resolve().parents[2]
        environment = environment_snapshot()
        try:
            project_git = git_state(repository)
        except (OSError, subprocess.SubprocessError):
            project_git = {"commit": "unknown", "dirty": True, "status": []}
        try:
            cotracker_git = git_state(self.cotracker_root)
        except (OSError, subprocess.SubprocessError):
            cotracker_git = {"commit": "unknown", "dirty": True, "status": []}
        (self.output / "environment.txt").write_text(
            json.dumps(environment, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (self.output / "device_info.txt").write_text(
            json.dumps(
                {"requested_device": self.device, "platform": environment["platform"]},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (self.output / "git_status.txt").write_text(
            "\n".join(project_git.get("status", [])) + "\n", encoding="utf-8"
        )
        (self.output / "git_status_porcelain.txt").write_text(
            "\n".join(project_git.get("status", [])) + "\n", encoding="utf-8"
        )
        (self.output / "git_commit.txt").write_text(
            str(project_git.get("commit", "unknown")) + "\n", encoding="utf-8"
        )
        try:
            diff = subprocess.run(
                ["git", "diff", "--binary", "HEAD"],
                cwd=repository,
                check=True,
                text=True,
                capture_output=True,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            diff = ""
        (self.output / "git_diff.patch").write_text(diff, encoding="utf-8")
        try:
            cached_diff = subprocess.run(
                ["git", "diff", "--binary", "--cached"],
                cwd=repository,
                check=True,
                text=True,
                capture_output=True,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            cached_diff = ""
        (self.output / "git_diff_cached.patch").write_text(
            cached_diff, encoding="utf-8"
        )
        manifest = {
            "run_id": self.output.name,
            "git_commit": project_git.get("commit", "unknown"),
            "git_dirty": bool(project_git.get("dirty", True)),
            "cotracker_commit": cotracker_git.get("commit", "unknown"),
            "cotracker_dirty": bool(cotracker_git.get("dirty", True)),
            "checkpoint_sha256": sha256_file(self.checkpoint),
            "config_sha256": config_sha256(self.config),
            "input_id": str(self.source),
            "start_time": self.run_start_utc,
            "device": self.device,
            "software_versions": environment["software_versions"],
            "tracker_parameters": self.config["tracker"],
            "motion_parameters": self.config["motion"],
            "scale_parameters": self.config["scale"],
            "signal_parameters": self.config["signal"],
            "pga_model_version": None,
        }
        (self.output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        timing_summary = summarize_timings(self.timing_records)
        captured_frames = sum(
            int(event.get("emitted_frames", 0))
            for event in self.events
            if event.get("event") == "playlist_source_end"
        )
        dropped_frames = int(self.capture_queue.rejected_items)
        dropped_blocks = int(self.output_queue.rejected_items)
        pga_p95 = timing_summary.get("capture_to_pga", {}).get("p95")
        observed_duration_s = (
            self.timing_records[-1].write_end - self.timing_records[0].capture_timestamp
            if self.timing_records
            else 0.0
        )
        queue_slopes: dict[str, float] = {}
        for field in ("frame_queue_depth", "block_queue_depth", "writer_queue_depth"):
            values = np.asarray([float(row[field]) for row in queue_rows], dtype=float)
            queue_slopes[field] = (
                float(np.polyfit(np.arange(values.size, dtype=float), values, 1)[0])
                if values.size >= 2
                else float("nan")
            )
        eligible_10min = observed_duration_s >= 590.0
        acceptance_pass = bool(
            eligible_10min
            and dropped_frames == 0
            and dropped_blocks == 0
            and captured_frames == self.frames_written
            and pga_p95 is not None
            and float(pga_p95) < float(self.config["runtime"]["max_end_to_end_latency_ms"])
            and not any(event.get("event") == "overload" for event in self.events)
        )
        realtime_status = "PASS" if acceptance_pass else "FAIL" if eligible_10min else "NOT_TESTED"
        summary = {
            "queues": queue_metrics,
            "events": len(self.events),
            "stopped_for_overload": any(event.get("event") == "overload" for event in self.events),
            "timing": timing_summary,
            "queue_slopes_per_block": queue_slopes,
            "captured_frames": captured_frames,
            "frames_written": self.frames_written,
            "dropped_frames": dropped_frames,
            "dropped_blocks": dropped_blocks,
            "observed_duration_s": observed_duration_s,
            "target_fps": float(self.config["runtime"]["target_fps"]),
            "realtime_acceptance": realtime_status,
            "realtime_acceptance_scope": (
                "10min_wall_clock" if eligible_10min else "requires_at_least_590s_observed"
            ),
            "pga_est": None,
            "pga_rejection_reason": "deployment_output_rejected_without_geometric_scale",
            "pga_research_output": "see running_pga.csv",
        }
        (self.output / "metrics.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return summary
