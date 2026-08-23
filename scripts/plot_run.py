#!/usr/bin/env python3
"""Generate the compact, deterministic plots expected by an audit bundle."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import tempfile

import numpy as np


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _column(rows: list[dict[str, str]], name: str) -> np.ndarray:
    output = []
    for row in rows:
        try:
            value = float(row.get(name, ""))
        except (TypeError, ValueError):
            value = float("nan")
        output.append(value)
    return np.asarray(output, dtype=float)


def _save(fig, output: Path) -> None:
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    fig.clf()


def plot_run(run_directory: Path) -> list[Path]:
    cache_root = Path(tempfile.gettempdir()) / "videoeew-matplotlib-cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run = run_directory.resolve()
    plots = run / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    signals_path = run / "filtered_signals.npz"
    common = None
    relative_time = None
    if signals_path.is_file():
        with np.load(signals_path) as values:
            timestamps = np.asarray(values["timestamps"], dtype=float)
            common_key = (
                "filtered_common_xy_px"
                if "filtered_common_xy_px" in values
                else "common_xy_px"
            )
            common = np.asarray(values[common_key], dtype=float)
            acceleration = (
                np.asarray(values["acceleration_proxy_px_s2"], dtype=float)
                if "acceleration_proxy_px_s2" in values
                else None
            )
        relative_time = timestamps - timestamps[0]
        panel_count = 2 if acceleration is not None else 1
        fig, axes = plt.subplots(panel_count, 1, figsize=(9, 5.4), sharex=True)
        axes = np.atleast_1d(axes)
        axes[0].plot(relative_time, common[:, 0], label="common x", linewidth=1.0)
        axes[0].plot(relative_time, common[:, 1], label="common y", linewidth=1.0)
        axes[0].set_ylabel("displacement (px)")
        axes[0].legend(loc="upper right", ncols=2)
        axes[0].grid(alpha=0.25)
        if acceleration is not None:
            axes[1].plot(relative_time, acceleration[:, 0], label="proxy x", linewidth=0.9)
            axes[1].plot(relative_time, acceleration[:, 1], label="proxy y", linewidth=0.9)
            axes[1].set_ylabel("acceleration proxy (px/s²)")
            axes[1].legend(loc="upper right", ncols=2)
            axes[1].grid(alpha=0.25)
        axes[-1].set_xlabel("time since video start (s)")
        path = plots / "motion_timeseries.png"
        _save(fig, path)
        plt.close(fig)
        written.append(path)

        if timestamps.size >= 4:
            sample_rate = 1 / np.median(np.diff(timestamps))
            frequencies = np.fft.rfftfreq(timestamps.size, d=1 / sample_rate)
            fig, ax = plt.subplots(figsize=(8.5, 4.4))
            for channel, label in enumerate(("common x", "common y")):
                spectrum = np.abs(np.fft.rfft(common[:, channel] - np.mean(common[:, channel])))
                ax.plot(frequencies, spectrum, label=label, linewidth=1.0)
            ax.set_xlim(0, min(10.0, sample_rate / 2))
            ax.set_xlabel("frequency (Hz)")
            ax.set_ylabel("amplitude (px)")
            ax.grid(alpha=0.25)
            ax.legend()
            path = plots / "spectrum.png"
            _save(fig, path)
            plt.close(fig)
            written.append(path)

    residual_path = run / "residual_motion.npz"
    if common is not None and relative_time is not None and residual_path.is_file():
        with np.load(residual_path) as values:
            residual = np.asarray(values["residual_xy_px"], dtype=float)
        common_magnitude = np.linalg.norm(common, axis=1)
        residual_rms = np.sqrt(np.nanmean(np.sum(np.square(residual), axis=2), axis=1))
        count = min(relative_time.size, residual_rms.size)
        fig, ax = plt.subplots(figsize=(8.5, 4.4))
        ax.plot(relative_time[:count], common_magnitude[:count], label="common magnitude")
        ax.plot(relative_time[:count], residual_rms[:count], label="local residual RMS")
        ax.set_xlabel("time since video start (s)")
        ax.set_ylabel("motion (px)")
        ax.grid(alpha=0.25)
        ax.legend()
        path = plots / "common_vs_residual.png"
        _save(fig, path)
        plt.close(fig)
        written.append(path)

    timing = _rows(run / "timing.csv")
    if timing:
        block = np.arange(len(timing))
        fig, ax = plt.subplots(figsize=(8.5, 4.4))
        columns = ("preprocess_ms", "tracker_ms", "motion_fit_ms", "signal_ms", "total_pipeline_ms")
        for name in columns:
            values = _column(timing, name)
            if np.isfinite(values).any():
                ax.plot(block, values, label=name, linewidth=0.9)
        ax.set_xlabel("block index")
        ax.set_ylabel("time (ms)")
        ax.grid(alpha=0.25)
        ax.legend(ncols=2, fontsize=8)
        path = plots / "latency.png"
        _save(fig, path)
        plt.close(fig)
        written.append(path)

    queue = _rows(run / "queue.csv")
    if queue:
        row_index = np.arange(len(queue))
        fig, ax = plt.subplots(figsize=(8.5, 4.0))
        for name in ("capture_queue_depth", "tracker_queue_depth", "output_queue_depth"):
            values = _column(queue, name)
            if np.isfinite(values).any():
                ax.step(row_index, values, where="post", label=name)
        ax.set_xlabel("sample index")
        ax.set_ylabel("queue depth")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        path = plots / "queue_depth.png"
        _save(fig, path)
        plt.close(fig)
        written.append(path)

    memory = _rows(run / "memory.csv")
    if memory:
        block = np.arange(len(memory))
        fig, ax = plt.subplots(figsize=(8.5, 4.0))
        ax.plot(block, _column(memory, "rss_mb"), label="RSS")
        ax.plot(block, _column(memory, "peak_rss_mb"), label="peak RSS")
        ax.set_xlabel("block index")
        ax.set_ylabel("memory (MB)")
        ax.grid(alpha=0.25)
        ax.legend()
        path = plots / "memory.png"
        _save(fig, path)
        plt.close(fig)
        written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    for output in plot_run(Path(args.run)):
        print(output)


if __name__ == "__main__":
    main()
