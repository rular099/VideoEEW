"""Stage timing records and percentile summaries."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
import resource
import sys

import numpy as np


@dataclass(frozen=True)
class TimingRecord:
    timestamp: float
    block_index: int
    capture_timestamp: float = float("nan")
    enqueue_timestamp: float = float("nan")
    tracker_start: float = float("nan")
    tracker_end: float = float("nan")
    motion_end: float = float("nan")
    signal_end: float = float("nan")
    pga_end: float = float("nan")
    write_end: float = float("nan")
    capture_to_tracker_end: float = float("nan")
    capture_to_pga: float = float("nan")
    queue_wait: float = float("nan")
    tracker_compute: float = float("nan")
    postprocess_compute: float = float("nan")
    capture_ms: float = 0.0
    preprocess_ms: float = 0.0
    encoder_ms: float = float("nan")
    tracker_ms: float = 0.0
    motion_fit_ms: float = 0.0
    signal_ms: float = 0.0
    pga_ms: float = 0.0
    total_pipeline_ms: float = 0.0
    end_to_end_latency_ms: float = float("nan")


@dataclass(frozen=True)
class MemoryRecord:
    timestamp: float
    block_index: int
    rss_mb: float
    peak_rss_mb: float
    system_available_mem_mb: float = float("nan")


def sample_memory(timestamp: float, block_index: int) -> MemoryRecord:
    """Collect process memory without making psutil a mandatory dependency."""

    rss_mb = float("nan")
    available_mb = float("nan")
    try:
        import psutil

        process = psutil.Process()
        rss_mb = process.memory_info().rss / (1024**2)
        available_mb = psutil.virtual_memory().available / (1024**2)
    except ImportError:
        pass
    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        peak /= 1024.0
    else:  # macOS reports bytes; Linux reports KiB.
        peak /= 1024.0**2
    if not np.isfinite(rss_mb):
        rss_mb = peak
    return MemoryRecord(
        timestamp=float(timestamp),
        block_index=int(block_index),
        rss_mb=float(rss_mb),
        peak_rss_mb=float(peak),
        system_available_mem_mb=float(available_mb),
    )


def summarize_timings(records: list[TimingRecord]) -> dict[str, dict[str, float]]:
    if not records:
        return {}
    summary: dict[str, dict[str, float]] = {}
    for field in (
        "capture_to_tracker_end",
        "capture_to_pga",
        "queue_wait",
        "tracker_compute",
        "postprocess_compute",
        "capture_ms",
        "preprocess_ms",
        "tracker_ms",
        "motion_fit_ms",
        "signal_ms",
        "pga_ms",
        "total_pipeline_ms",
        "end_to_end_latency_ms",
    ):
        values = np.asarray([getattr(record, field) for record in records], dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            summary[field] = {
                "mean": float(np.mean(values)),
                "p50": float(np.percentile(values, 50)),
                "p95": float(np.percentile(values, 95)),
                "p99": float(np.percentile(values, 99)),
                "max": float(np.max(values)),
            }
    return summary


def write_timing_csv(path: str | Path, records: list[TimingRecord]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(TimingRecord.__dataclass_fields__)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def write_memory_csv(path: str | Path, records: list[MemoryRecord]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(MemoryRecord.__dataclass_fields__)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))
