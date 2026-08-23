"""Bounded frame windows and overload-visible worker queues."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from queue import Empty, Full, Queue
from threading import Lock
from typing import Generic, TypeVar

import numpy as np


T = TypeVar("T")


class BufferOverload(RuntimeError):
    """Raised instead of silently dropping data from a full bounded queue."""


@dataclass(frozen=True)
class FrameWindow:
    frames: np.ndarray
    timestamps: np.ndarray
    frame_indices: np.ndarray


class SlidingFrameBuffer:
    """Keep exactly one tracker window and emit it every configured step."""

    def __init__(self, window_len: int, step: int) -> None:
        if window_len < 2:
            raise ValueError("window_len must be at least 2")
        if not 1 <= step <= window_len:
            raise ValueError("step must be in [1, window_len]")
        self.window_len = int(window_len)
        self.step = int(step)
        self._frames: deque[np.ndarray] = deque(maxlen=window_len)
        self._timestamps: deque[float] = deque(maxlen=window_len)
        self._indices: deque[int] = deque(maxlen=window_len)
        self._since_emit = 0
        self._last_timestamp: float | None = None
        self.total_frames = 0

    @property
    def size(self) -> int:
        return len(self._frames)

    @property
    def capacity(self) -> int:
        return self.window_len

    def append(
        self, frame: np.ndarray, timestamp: float, frame_index: int | None = None
    ) -> FrameWindow | None:
        timestamp = float(timestamp)
        if not np.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        if self._last_timestamp is not None and timestamp <= self._last_timestamp:
            raise ValueError("frame timestamps must be strictly increasing")
        array = np.asarray(frame)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("frame must have shape [H,W,3]")
        index = self.total_frames if frame_index is None else int(frame_index)
        self._frames.append(array)
        self._timestamps.append(timestamp)
        self._indices.append(index)
        self._last_timestamp = timestamp
        self.total_frames += 1
        self._since_emit += 1
        first_window = self.total_frames == self.window_len
        stepped_window = self.total_frames > self.window_len and self._since_emit >= self.step
        if first_window or stepped_window:
            self._since_emit = 0
            return self.snapshot()
        return None

    def snapshot(self) -> FrameWindow:
        if len(self._frames) != self.window_len:
            raise RuntimeError("buffer does not yet contain a complete window")
        return FrameWindow(
            frames=np.stack(tuple(self._frames)),
            timestamps=np.asarray(self._timestamps, dtype=np.float64),
            frame_indices=np.asarray(self._indices, dtype=np.int64),
        )


class AuditedBoundedQueue(Generic[T]):
    """A bounded queue that counts and exposes every rejected enqueue."""

    def __init__(self, maxsize: int, name: str) -> None:
        if maxsize <= 0:
            raise ValueError("maxsize must be positive")
        self.name = name
        self._queue: Queue[T] = Queue(maxsize=maxsize)
        self._lock = Lock()
        self.rejected_items = 0
        self.max_observed_depth = 0

    @property
    def depth(self) -> int:
        return self._queue.qsize()

    @property
    def capacity(self) -> int:
        return self._queue.maxsize

    def put(self, item: T, timeout: float = 0.0) -> None:
        try:
            self._queue.put(item, block=timeout > 0, timeout=max(0.0, timeout))
        except Full as exc:
            with self._lock:
                self.rejected_items += 1
            raise BufferOverload(
                f"{self.name} queue is full ({self.depth}/{self.capacity}); item rejected"
            ) from exc
        with self._lock:
            self.max_observed_depth = max(self.max_observed_depth, self.depth)

    def get(self, timeout: float | None = None) -> T:
        try:
            return self._queue.get(block=timeout is not None, timeout=timeout or 0.0)
        except Empty as exc:
            raise LookupError(f"{self.name} queue is empty") from exc

    def task_done(self) -> None:
        self._queue.task_done()

    def metrics(self) -> dict[str, int | str]:
        return {
            "name": self.name,
            "depth": self.depth,
            "capacity": self.capacity,
            "max_observed_depth": self.max_observed_depth,
            "rejected_items": self.rejected_items,
        }

