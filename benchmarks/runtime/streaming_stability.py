#!/usr/bin/env python3
"""Fast-forward bounded-buffer stability test over a long logical stream."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import psutil


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from seismic_motion.tracking.online_buffer import (  # noqa: E402
    AuditedBoundedQueue,
    BufferOverload,
    SlidingFrameBuffer,
)


def run(duration_s: float, fps: float, height: int, width: int) -> dict[str, object]:
    total_frames = int(round(duration_s * fps))
    buffer = SlidingFrameBuffer(window_len=16, step=8)
    queue = AuditedBoundedQueue(maxsize=2, name="tracker")
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    process = psutil.Process()
    rss_samples = []
    emitted = 0
    explicit_overloads = 0
    for index in range(total_frames):
        window = buffer.append(frame, index / fps, index)
        if window is not None:
            emitted += 1
            queue.put(window.frame_indices.copy())
            if emitted % 31 == 0:
                try:
                    queue.put(window.frame_indices.copy())
                    queue.put(window.frame_indices.copy())
                except BufferOverload:
                    explicit_overloads += 1
            while queue.depth:
                queue.get()
                queue.task_done()
        if index % max(1, int(fps)) == 0:
            rss_samples.append(process.memory_info().rss / (1024 * 1024))
    rss = np.asarray(rss_samples, dtype=float)
    warmup = min(30, max(0, rss.size // 4))
    tail = rss[warmup:]
    slope = (
        float(np.polyfit(np.arange(tail.size) / 60, tail, 1)[0])
        if tail.size >= 2
        else float("nan")
    )
    return {
        "logical_duration_s": duration_s,
        "fps": fps,
        "total_frames": total_frames,
        "emitted_windows": emitted,
        "frame_buffer_capacity": buffer.capacity,
        "frame_buffer_final_size": buffer.size,
        "queue": queue.metrics(),
        "explicit_overload_events_observed": explicit_overloads,
        "rss_mb": {
            "initial": float(rss[0]),
            "final": float(rss[-1]),
            "minimum": float(np.min(rss)),
            "maximum": float(np.max(rss)),
            "post_warmup_slope_mb_per_min": slope,
        },
        "pass": bool(buffer.size <= buffer.capacity and queue.depth == 0 and slope < 1.0),
        "scope": "bounded frame/queue fast-forward; tracker tensor cap is tested separately by reseed",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-s", type=float, default=600)
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    metrics = run(args.duration_s, args.fps, args.height, args.width)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    if not metrics["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

