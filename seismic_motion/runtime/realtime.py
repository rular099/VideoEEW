"""Bounded three-worker realtime capture, tracking and audit writer."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from threading import Event, Thread
import time
from typing import Any

import numpy as np

from seismic_motion.motion.global_motion import fit_global_transform
from seismic_motion.motion.quality import QualityThresholds, assess_motion_quality
from seismic_motion.tracking.cotracker_adapter import CoTrackerAdapter, CoTrackerAdapterConfig
from seismic_motion.tracking.online_buffer import (
    AuditedBoundedQueue,
    BufferOverload,
    SlidingFrameBuffer,
)
from seismic_motion.tracking.types import TrackBatch

from .pipeline import _preprocess_frame


@dataclass(frozen=True)
class CapturedFrame:
    frame_rgb: np.ndarray
    timestamp: float
    frame_index: int
    capture_monotonic: float


class RealtimeRunner:
    """Run finite bounded queues; overload stops capture instead of silent loss."""

    def __init__(
        self,
        source: str | int,
        config: dict[str, Any],
        *,
        cotracker_root: str,
        checkpoint: str,
        device: str,
        output_directory: str | Path,
    ) -> None:
        tracker = config["tracker"]
        queue_bound = int(config["runtime"]["max_queue_blocks"])
        self.source = source
        self.config = config
        self.output = Path(output_directory)
        self.output.mkdir(parents=True, exist_ok=True)
        self.capture_queue: AuditedBoundedQueue[CapturedFrame | None] = AuditedBoundedQueue(
            maxsize=max(queue_bound * int(tracker["step"]), int(tracker["window_len"])),
            name="capture",
        )
        self.output_queue: AuditedBoundedQueue[TrackBatch | None] = AuditedBoundedQueue(
            maxsize=queue_bound, name="output"
        )
        self.stop_event = Event()
        self.events: list[dict[str, object]] = []
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
        except ImportError as exc:  # pragma: no cover - board environment
            self._record_event({"event": "capture_error", "reason": str(exc)})
            self.stop_event.set()
            return
        capture = cv2.VideoCapture(self.source)
        if not capture.isOpened():
            self._record_event({"event": "capture_error", "reason": "source_open_failed"})
            self.stop_event.set()
            return
        start = time.monotonic()
        frame_index = 0
        source_fps = float(capture.get(cv2.CAP_PROP_FPS)) if isinstance(self.source, str) else 0.0
        while not self.stop_event.is_set():
            if duration_s is not None and time.monotonic() - start >= duration_s:
                break
            if source_fps > 0 and frame_index:
                deadline = start + frame_index / source_fps
                remaining = deadline - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
            ok, frame_bgr = capture.read()
            captured = time.monotonic()
            if not ok:
                break
            frame = CapturedFrame(
                frame_rgb=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
                timestamp=captured,
                frame_index=frame_index,
                capture_monotonic=captured,
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
        capture.release()
        while not self.stop_event.is_set():
            try:
                self.capture_queue.put(None, timeout=0.1)
                break
            except BufferOverload:
                continue

    def _tracker_worker(self) -> None:
        tracker = self.config["tracker"]
        video = self.config["video"]
        frame_buffer = SlidingFrameBuffer(int(tracker["window_len"]), int(tracker["step"]))
        last_processed_start: int | None = None
        frame_count = 0
        while not self.stop_event.is_set():
            try:
                captured = self.capture_queue.get(timeout=0.1)
            except LookupError:
                continue
            self.capture_queue.task_done()
            if captured is None:
                if frame_count:
                    retained = frame_buffer.retained()
                    last_window_end = (
                        -1
                        if last_processed_start is None
                        else last_processed_start + int(tracker["window_len"])
                    )
                    if last_processed_start is None or frame_count > last_window_end:
                        next_start = (
                            int(retained.frame_indices[0])
                            if last_processed_start is None
                            else last_processed_start + int(tracker["step"])
                        )
                        selection = retained.frame_indices >= next_start
                        tail_frames = retained.frames[selection]
                        tail_times = retained.timestamps[selection]
                        valid_frames = tail_frames.shape[0]
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
                        self.output_queue.put(
                            self.adapter.process_window(
                                frames,
                                timestamps,
                                indices,
                                final=True,
                                valid_frames=valid_frames,
                            ),
                            timeout=0.1,
                        )
                    else:
                        pending = self.adapter.flush_pending()
                        if pending is not None:
                            self.output_queue.put(pending, timeout=0.1)
                self.output_queue.put(None, timeout=0.1)
                return
            processed = _preprocess_frame(
                captured.frame_rgb,
                video.get("roi"),
                tuple(int(value) for value in tracker["model_resolution"]),
            )
            window = frame_buffer.append(
                processed, captured.timestamp, captured.frame_index
            )
            frame_count += 1
            if window is None:
                continue
            begin = time.monotonic()
            batch = self.adapter.process_window(
                window.frames, window.timestamps, window.frame_indices
            )
            last_processed_start = int(window.frame_indices[0])
            tracker_ms = (time.monotonic() - begin) * 1000
            try:
                self.output_queue.put(batch, timeout=0.1)
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
                return
            self._record_event(
                {
                    "event": "tracker_block",
                    "frame_index": int(batch.frame_indices[0]),
                    "tracker_ms": tracker_ms,
                    "end_to_end_latency_ms": (time.monotonic() - captured.capture_monotonic)
                    * 1000,
                }
            )

    def _writer_worker(self) -> None:
        motion = self.config["motion"]
        thresholds = QualityThresholds(
            min_valid_tracks=int(motion["min_valid_tracks"]),
            min_inlier_ratio=float(motion["min_inlier_ratio"]),
            min_spatial_coverage=float(motion["min_spatial_coverage"]),
            max_fit_rmse_px=float(motion["max_fit_rmse_px"]),
        )
        with (self.output / "tracks.csv").open("w", newline="", encoding="utf-8") as track_handle, (
            self.output / "common_motion.csv"
        ).open("w", newline="", encoding="utf-8") as common_handle:
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
            previous = None
            while not self.stop_event.is_set():
                try:
                    batch = self.output_queue.get(timeout=0.1)
                except LookupError:
                    continue
                self.output_queue.task_done()
                if batch is None:
                    break
                for local_index, (timestamp, frame_index) in enumerate(
                    zip(batch.timestamps, batch.frame_indices)
                ):
                    for point_index, point_id in enumerate(batch.point_ids):
                        track_writer.writerow(
                            [
                                timestamp,
                                frame_index,
                                point_id,
                                *batch.xy_px[local_index, point_index],
                                int(batch.visible[local_index, point_index]),
                                batch.reseed_id,
                            ]
                        )
                    try:
                        estimate = fit_global_transform(
                            batch.query_xy_px,
                            batch.xy_px[local_index],
                            batch.visible[local_index],
                            model=str(motion["global_model"]),
                            use_ransac=bool(motion["use_ransac"]),
                            ransac_threshold_px=float(motion["ransac_threshold_px"]),
                            frame_size=tuple(self.config["tracker"]["model_resolution"]),
                        )
                        decision = assess_motion_quality(estimate, thresholds, previous)
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
                        if decision.quality.value != "INVALID":
                            previous = estimate
                    except ValueError:
                        common_writer.writerow(
                            [timestamp, frame_index, "", "", "", "", "", "", "INVALID", "fit_failed"]
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
        return {
            "queues": queue_metrics,
            "events": len(self.events),
            "stopped_for_overload": any(event.get("event") == "overload" for event in self.events),
            "pga_est": None,
            "pga_rejection_reason": "scale_invalid_and_model_not_trained",
        }
