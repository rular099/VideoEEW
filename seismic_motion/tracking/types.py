"""Stable data contracts at the tracker boundary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TrackBatch:
    """A time-contiguous block of sparse tracks in source-image pixels."""

    timestamps: np.ndarray
    frame_indices: np.ndarray
    point_ids: np.ndarray
    xy_px: np.ndarray
    visible: np.ndarray
    confidence: np.ndarray
    query_xy_px: np.ndarray
    reseed_id: int = 0

    def __post_init__(self) -> None:
        timestamps = np.asarray(self.timestamps, dtype=np.float64)
        frame_indices = np.asarray(self.frame_indices, dtype=np.int64)
        point_ids = np.asarray(self.point_ids, dtype=np.int64)
        xy = np.asarray(self.xy_px, dtype=np.float32)
        visible = np.asarray(self.visible, dtype=bool)
        confidence = np.asarray(self.confidence, dtype=np.float32)
        query_xy = np.asarray(self.query_xy_px, dtype=np.float32)
        frames = timestamps.shape[0]
        points = point_ids.shape[0]
        if timestamps.ndim != 1 or frame_indices.shape != (frames,):
            raise ValueError("timestamps and frame_indices must be one-dimensional")
        if frames and np.any(np.diff(timestamps) <= 0):
            raise ValueError("timestamps must be strictly increasing")
        if xy.shape != (frames, points, 2):
            raise ValueError(f"xy_px must have shape [T,N,2], got {xy.shape}")
        if visible.shape != (frames, points):
            raise ValueError("visible must have shape [T,N]")
        if confidence.shape != (frames, points):
            raise ValueError("confidence must have shape [T,N]")
        if query_xy.shape != (points, 2):
            raise ValueError("query_xy_px must have shape [N,2]")
        if not np.isfinite(timestamps).all():
            raise ValueError("timestamps contain non-finite values")
        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "frame_indices", frame_indices)
        object.__setattr__(self, "point_ids", point_ids)
        object.__setattr__(self, "xy_px", xy)
        object.__setattr__(self, "visible", visible)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "query_xy_px", query_xy)

    @property
    def num_frames(self) -> int:
        return int(self.timestamps.shape[0])

    @property
    def num_points(self) -> int:
        return int(self.point_ids.shape[0])

    def save_npz(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            timestamps=self.timestamps,
            frame_indices=self.frame_indices,
            point_ids=self.point_ids,
            xy_px=self.xy_px,
            visible=self.visible,
            confidence=self.confidence,
            query_xy_px=self.query_xy_px,
            reseed_id=np.asarray(self.reseed_id, dtype=np.int64),
        )


def concatenate_track_batches(batches: list[TrackBatch]) -> TrackBatch:
    if not batches:
        raise ValueError("at least one track batch is required")
    point_ids = batches[0].point_ids
    query_xy = batches[0].query_xy_px
    for batch in batches[1:]:
        if not np.array_equal(batch.point_ids, point_ids):
            raise ValueError("point identities differ between batches")
    return TrackBatch(
        timestamps=np.concatenate([batch.timestamps for batch in batches]),
        frame_indices=np.concatenate([batch.frame_indices for batch in batches]),
        point_ids=point_ids,
        xy_px=np.concatenate([batch.xy_px for batch in batches], axis=0),
        visible=np.concatenate([batch.visible for batch in batches], axis=0),
        confidence=np.concatenate([batch.confidence for batch in batches], axis=0),
        query_xy_px=query_xy,
        reseed_id=max(batch.reseed_id for batch in batches),
    )

