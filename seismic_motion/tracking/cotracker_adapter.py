"""Minimal-intrusion sparse adapter around the adjacent CoTracker3 checkout."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .point_selection import select_distributed_corners, validate_manual_points
from .types import TrackBatch


@dataclass(frozen=True)
class CoTrackerAdapterConfig:
    cotracker_root: str | None
    checkpoint: str
    device: str = "cuda"
    num_points: int = 32
    window_len: int = 16
    step: int = 8
    iters: int = 6
    point_mode: str = "corners"
    roi: tuple[int, int, int, int] | None = None
    max_blocks_before_reseed: int = 64


class CoTrackerAdapter:
    """Expose only finalized track steps and cap upstream history by reseeding.

    The public CoTracker online core grows prediction history by `step` frames
    per call. This adapter resets that state after a configured number of
    blocks and seeds the next window with the latest overlapping coordinates.
    Every reset is surfaced through `events` and `TrackBatch.reseed_id`.
    """

    def __init__(
        self,
        config: CoTrackerAdapterConfig,
        manual_points: np.ndarray | None = None,
        event_sink: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        if config.window_len != 2 * config.step:
            raise ValueError("first implementation requires window_len == 2 * step")
        if config.iters != 6:
            raise ValueError(
                "the upstream public predictor hard-codes six iterations; "
                "a different value requires a numerically audited backend"
            )
        if config.max_blocks_before_reseed <= 0:
            raise ValueError("max_blocks_before_reseed must be positive")
        self.config = config
        self.manual_points = None if manual_points is None else np.asarray(manual_points)
        self.event_sink = event_sink
        self._predictor = None
        self._torch = None
        self._query_points: np.ndarray | None = None
        self._initial_query_points: np.ndarray | None = None
        self._point_ids: np.ndarray | None = None
        self._base_frame = 0
        self._blocks_since_reseed = 0
        self._reseed_id = 0
        self._last_tracks: np.ndarray | None = None
        self._last_visibility: np.ndarray | None = None
        self._last_base_frame = 0
        self._last_window_timestamps: np.ndarray | None = None
        self._last_window_indices: np.ndarray | None = None
        self._next_output_frame: int | None = None

    @property
    def events_enabled(self) -> bool:
        return self.event_sink is not None

    @property
    def reseed_id(self) -> int:
        return self._reseed_id

    @property
    def upstream_history_bound_frames(self) -> int:
        return self.config.window_len + self.config.step * self.config.max_blocks_before_reseed

    def _emit_event(self, event: dict[str, object]) -> None:
        if self.event_sink is not None:
            self.event_sink(event)

    def _load_backend(self) -> None:
        if self._predictor is not None:
            return
        root_value = self.config.cotracker_root or os.environ.get("COTRACKER_ROOT")
        if not root_value:
            raise RuntimeError("set tracker.cotracker_root or COTRACKER_ROOT")
        root = Path(root_value).expanduser().resolve()
        if not (root / "cotracker" / "predictor.py").is_file():
            raise RuntimeError(f"not a CoTracker checkout: {root}")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        try:
            import torch
            from cotracker.predictor import CoTrackerOnlinePredictor
        except ImportError as exc:
            raise RuntimeError("PyTorch and the external CoTracker checkout are required") from exc
        checkpoint = Path(self.config.checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise RuntimeError(f"checkpoint not found: {checkpoint}")
        self._torch = torch
        self._predictor = CoTrackerOnlinePredictor(
            checkpoint=str(checkpoint), window_len=self.config.window_len
        ).to(self.config.device).eval()
        if int(self._predictor.step) != self.config.step:
            raise RuntimeError(
                f"configured step {self.config.step} differs from model step {self._predictor.step}"
            )

    def _select_points(self, first_frame: np.ndarray) -> np.ndarray:
        height, width = first_frame.shape[:2]
        if self.manual_points is not None:
            return validate_manual_points(
                self.manual_points, width, height, expected_count=self.config.num_points
            )
        if self.config.point_mode != "corners":
            raise ValueError(f"unsupported point mode: {self.config.point_mode}")
        return select_distributed_corners(
            first_frame,
            self.config.num_points,
            roi=self.config.roi,
        )

    def _queries(self, points: np.ndarray):
        assert self._torch is not None
        values = np.zeros((1, points.shape[0], 3), dtype=np.float32)
        values[0, :, 1:] = points
        return self._torch.from_numpy(values).to(self.config.device)

    def _video_tensor(self, frames: np.ndarray):
        assert self._torch is not None
        values = np.asarray(frames, dtype=np.float32)
        return self._torch.from_numpy(values).to(self.config.device).permute(0, 3, 1, 2)[None]

    def process_window(
        self,
        frames_rgb: np.ndarray,
        timestamps: np.ndarray,
        frame_indices: np.ndarray,
        final: bool = False,
    ) -> TrackBatch:
        self._load_backend()
        assert self._predictor is not None
        assert self._torch is not None
        frames = np.asarray(frames_rgb)
        times = np.asarray(timestamps, dtype=np.float64)
        indices = np.asarray(frame_indices, dtype=np.int64)
        if frames.shape[0] != self.config.window_len:
            raise ValueError("adapter requires complete tracker windows")
        if times.shape != (self.config.window_len,) or indices.shape != times.shape:
            raise ValueError("timestamps/frame_indices do not match the window")
        chunk_start = int(indices[0])
        needs_initialization = self._query_points is None
        needs_reseed = self._blocks_since_reseed >= self.config.max_blocks_before_reseed
        if needs_initialization:
            self._query_points = self._select_points(frames[0])
            self._initial_query_points = self._query_points.copy()
            self._point_ids = np.arange(self._query_points.shape[0], dtype=np.int64)
        elif needs_reseed:
            if self._last_tracks is None:
                raise RuntimeError("cannot reseed without a prior track result")
            local_index = chunk_start - self._last_base_frame
            if not 0 <= local_index < self._last_tracks.shape[0]:
                raise RuntimeError("reseed overlap is not present in prior model output")
            self._query_points = self._last_tracks[local_index].copy()
            self._reseed_id += 1
            self._emit_event(
                {
                    "event": "tracker_reseed",
                    "frame_index": chunk_start,
                    "reseed_id": self._reseed_id,
                    "reason": "bounded_upstream_history",
                }
            )
        video = self._video_tensor(frames)
        if needs_initialization or needs_reseed:
            self._predictor(
                video,
                is_first_step=True,
                queries=self._queries(self._query_points),
                grid_size=0,
            )
            self._base_frame = chunk_start
            self._blocks_since_reseed = 0
        with self._torch.inference_mode():
            tracks_tensor, visibility_tensor = self._predictor(
                video,
                is_first_step=False,
                queries=None,
                grid_size=0,
            )
        tracks = tracks_tensor[0].detach().cpu().numpy().astype(np.float32)
        visibility = visibility_tensor[0].detach().cpu().numpy().astype(bool)
        self._last_tracks = tracks
        self._last_visibility = visibility
        self._last_base_frame = self._base_frame
        self._last_window_timestamps = times.copy()
        self._last_window_indices = indices.copy()
        self._blocks_since_reseed += 1
        stable_end = int(indices[-1]) + 1 if final else chunk_start + self.config.step
        output_start = chunk_start if self._next_output_frame is None else self._next_output_frame
        output_end = stable_end
        local_start = output_start - self._base_frame
        local_end = output_end - self._base_frame
        if local_start < 0 or local_end > tracks.shape[0]:
            raise RuntimeError("upstream result does not cover the finalized output step")
        assert self._point_ids is not None
        assert self._initial_query_points is not None
        result_tracks = tracks[local_start:local_end]
        result_visibility = visibility[local_start:local_end]
        window_offset = output_start - chunk_start
        result_times = times[window_offset : window_offset + output_end - output_start]
        result_indices = indices[window_offset : window_offset + output_end - output_start]
        batch = TrackBatch(
            timestamps=result_times,
            frame_indices=result_indices,
            point_ids=self._point_ids,
            xy_px=result_tracks,
            visible=result_visibility,
            confidence=np.full(result_visibility.shape, np.nan, dtype=np.float32),
            query_xy_px=self._initial_query_points,
            reseed_id=self._reseed_id,
        )
        self._next_output_frame = output_end
        return batch

    def flush_pending(self) -> TrackBatch | None:
        """Emit the final overlap without another model call at end of stream."""

        if (
            self._last_tracks is None
            or self._last_visibility is None
            or self._last_window_timestamps is None
            or self._last_window_indices is None
            or self._next_output_frame is None
        ):
            return None
        final_end = int(self._last_window_indices[-1]) + 1
        if self._next_output_frame >= final_end:
            return None
        output_start = self._next_output_frame
        local_start = output_start - self._last_base_frame
        local_end = final_end - self._last_base_frame
        window_start = int(self._last_window_indices[0])
        window_offset = output_start - window_start
        count = final_end - output_start
        assert self._point_ids is not None
        assert self._initial_query_points is not None
        batch = TrackBatch(
            timestamps=self._last_window_timestamps[window_offset : window_offset + count],
            frame_indices=self._last_window_indices[window_offset : window_offset + count],
            point_ids=self._point_ids,
            xy_px=self._last_tracks[local_start:local_end],
            visible=self._last_visibility[local_start:local_end],
            confidence=np.full(
                self._last_visibility[local_start:local_end].shape,
                np.nan,
                dtype=np.float32,
            ),
            query_xy_px=self._initial_query_points,
            reseed_id=self._reseed_id,
        )
        self._next_output_frame = final_end
        return batch
