#!/usr/bin/env python3
"""Plot compact diagnostics for a summarized PGA evaluation run."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import tempfile

import numpy as np


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    run = Path(args.run)
    output = run / "plots"
    output.mkdir(parents=True, exist_ok=True)
    cache = Path(tempfile.gettempdir()) / "videoeew-matplotlib-cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    predictions = _rows(run / "pga_predictions.csv")
    truth = np.asarray([float(row["PGA_true"]) for row in predictions])
    algorithms = [
        name
        for name in ("single_coefficient", "ridge", "huber")
        if name in predictions[0]
    ]
    figure, axis = plt.subplots(figsize=(5.4, 5.0))
    limits = [max(1.0, float(np.min(truth)) * 0.7), float(np.max(truth)) * 1.5]
    for algorithm in algorithms:
        estimate = np.asarray([float(row[algorithm]) for row in predictions])
        axis.scatter(truth, estimate, label=algorithm, alpha=0.85)
    axis.plot(limits, limits, color="black", linestyle="--", linewidth=1, label="ideal")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(limits)
    axis.set_ylim(limits)
    axis.set_xlabel("instrument PGA truth (gal)")
    axis.set_ylabel("out-of-fold estimate (gal)")
    axis.grid(alpha=0.25, which="both")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "pga_true_vs_estimate.png", dpi=180)
    plt.close(figure)

    alignment = _rows(run / "alignment_summary.csv")
    record_ids = [row["record_id"] for row in alignment]
    x = np.arange(len(record_ids))
    acceleration = np.asarray([float(row["acceleration_correlation"]) for row in alignment])
    displacement = np.asarray([float(row["displacement_correlation"]) for row in alignment])
    figure, axis = plt.subplots(figsize=(9.0, 4.3))
    axis.bar(x - 0.2, acceleration, 0.4, label="acceleration proxy")
    axis.bar(x + 0.2, displacement, 0.4, label="displacement")
    axis.axhline(0.4, color="black", linestyle="--", linewidth=1, label="training gate")
    axis.set_xticks(x, record_ids)
    axis.set_xlabel("record ID")
    axis.set_ylabel("maximum candidate correlation")
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.2, axis="y")
    axis.legend(fontsize=8, ncols=3)
    figure.tight_layout()
    figure.savefig(output / "alignment_correlation.png", dpi=180)
    plt.close(figure)

    runtime = _rows(run / "runtime_summary.csv")
    record_ids = [row["record_id"] for row in runtime]
    p95 = np.asarray([float(row["tracker_p95_ms"]) for row in runtime])
    figure, axis = plt.subplots(figsize=(9.0, 4.3))
    axis.bar(np.arange(len(record_ids)), p95)
    axis.axhline(266.7, color="black", linestyle="--", linewidth=1, label="30 FPS step-8 budget")
    axis.set_xticks(np.arange(len(record_ids)), record_ids)
    axis.set_xlabel("record ID")
    axis.set_ylabel("tracker p95 (ms/block)")
    axis.grid(alpha=0.2, axis="y")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "tracker_p95_by_record.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
