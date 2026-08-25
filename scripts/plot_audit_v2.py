#!/usr/bin/env python3
"""Generate the fixed-name next-stage audit figures without inventing data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _number(row: dict[str, str], name: str) -> float:
    try:
        return float(row.get(name, "nan"))
    except (TypeError, ValueError):
        return float("nan")


def _save_placeholder(path: Path, title: str, reason: str) -> None:
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.axis("off")
    axis.set_title(title)
    axis.text(0.5, 0.5, reason, ha="center", va="center", wrap=True)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _pga_scatter(path: Path, output: Path, title: str) -> bool:
    rows = [row for row in _rows(path) if row.get("included") == "True"]
    truth = np.asarray([_number(row, "true_pga_gal") for row in rows])
    estimate = np.asarray([_number(row, "predicted_pga_gal") for row in rows])
    finite = np.isfinite(truth) & np.isfinite(estimate)
    if not finite.any():
        _save_placeholder(output, title, "NOT_EVALUABLE: no finite included predictions")
        return False
    figure, axis = plt.subplots(figsize=(5.5, 5))
    axis.scatter(truth[finite], estimate[finite], s=35, alpha=0.8)
    upper = max(float(np.max(truth[finite])), float(np.max(estimate[finite])), 1.0)
    axis.plot([0, upper], [0, upper], color="black", linestyle="--", linewidth=1)
    axis.set(xlabel="True PGA (gal)", ylabel="Estimated PGA (gal)", title=title)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return True


def _error_plot(rows: list[dict[str, str]], group: str, output: Path) -> bool:
    included = [row for row in rows if row.get("included") == "True"]
    grouped: dict[str, list[float]] = {}
    for row in included:
        name = row.get(group, "UNKNOWN") or "UNKNOWN"
        error = _number(row, "abs_error_gal")
        if np.isfinite(error):
            grouped.setdefault(name, []).append(error)
    if not grouped or (group != "record_id" and set(grouped) == {"UNKNOWN"}):
        _save_placeholder(
            output,
            f"PGA absolute error by {group}",
            "NOT_EVALUABLE: grouping metadata is UNKNOWN",
        )
        return False
    labels = sorted(grouped)
    values = [float(np.mean(grouped[label])) for label in labels]
    figure, axis = plt.subplots(figsize=(max(7, 0.35 * len(labels)), 4.5))
    axis.bar(np.arange(len(labels)), values)
    axis.set_xticks(np.arange(len(labels)), labels, rotation=90 if len(labels) > 8 else 0)
    axis.set_ylabel("Mean absolute error (gal)")
    axis.set_title(f"PGA error by {group}")
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return True


def generate(args: argparse.Namespace) -> dict[str, str]:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    status: dict[str, str] = {}
    pga_dir = Path(args.pga_dir) if args.pga_dir else None
    pga_layout = (
        ("pga_eval_all.csv", "pga_true_vs_estimate_all.png", "PGA: ALL"),
        (
            "pga_eval_video_quality.csv",
            "pga_true_vs_estimate_video_quality.png",
            "PGA: VIDEO-QUALITY-ONLY",
        ),
        ("pga_eval_posthoc_aligned.csv", "pga_true_vs_estimate_posthoc.png", "PGA: POST-HOC"),
    )
    for source_name, output_name, title in pga_layout:
        generated = bool(pga_dir) and _pga_scatter(
            pga_dir / source_name, output / output_name, title
        )
        if not pga_dir:
            _save_placeholder(output / output_name, title, "NOT_MEASURED")
        status[output_name] = "GENERATED" if generated else "NOT_EVALUABLE_PLACEHOLDER"
    all_rows = _rows(pga_dir / "pga_eval_all.csv") if pga_dir else []
    for group, output_name in (
        ("record_id", "pga_error_by_record.png"),
        ("event_id", "pga_error_by_event.png"),
        ("camera_id", "pga_error_by_camera.png"),
        ("site_id", "pga_error_by_site.png"),
    ):
        generated = _error_plot(all_rows, group, output / output_name)
        status[output_name] = "GENERATED" if generated else "NOT_EVALUABLE_PLACEHOLDER"

    offline_dir = Path(args.offline_pga_dir) if args.offline_pga_dir else None
    causal_rows = {
        row.get("record_id", ""): row for row in all_rows if row.get("included") == "True"
    }
    offline_rows = {
        row.get("record_id", ""): row
        for row in (_rows(offline_dir / "pga_eval_all.csv") if offline_dir else [])
        if row.get("included") == "True"
    }
    common = sorted(set(causal_rows).intersection(offline_rows))
    offline_values = np.asarray(
        [_number(offline_rows[key], "predicted_pga_gal") for key in common]
    )
    causal_values = np.asarray(
        [_number(causal_rows[key], "predicted_pga_gal") for key in common]
    )
    finite = np.isfinite(offline_values) & np.isfinite(causal_values)
    name = "offline_vs_causal_pga.png"
    if finite.any():
        figure, axis = plt.subplots(figsize=(5.5, 5))
        axis.scatter(offline_values[finite], causal_values[finite])
        upper = max(float(np.max(offline_values[finite])), float(np.max(causal_values[finite])))
        axis.plot([0, upper], [0, upper], "k--", linewidth=1)
        axis.set(xlabel="Offline zero-phase estimate (gal)", ylabel="Causal estimate (gal)")
        axis.grid(alpha=0.2)
        figure.tight_layout()
        figure.savefig(output / name, dpi=180)
        plt.close(figure)
        status[name] = "GENERATED"
    else:
        _save_placeholder(output / name, "Offline versus causal PGA", "NOT_EVALUABLE: no matched causal run")
        status[name] = "NOT_EVALUABLE_PLACEHOLDER"

    runtime = Path(args.runtime_dir) if args.runtime_dir else None
    timing = _rows(runtime / "runtime_timing.csv") if runtime else []
    queue = _rows(runtime / "runtime_queue.csv") if runtime else []
    memory = _rows(runtime / "runtime_memory.csv") if runtime else []
    for output_name, rows, columns, ylabel in (
        (
            "runtime_latency_timeseries.png",
            timing,
            ("capture_to_pga", "tracker_compute", "postprocess_compute"),
            "Latency (ms)",
        ),
        (
            "runtime_queue_depth.png",
            queue,
            ("frame_queue_depth", "block_queue_depth", "writer_queue_depth"),
            "Queue depth",
        ),
        ("runtime_memory.png", memory, ("rss_mb", "peak_rss_mb"), "Memory (MB)"),
    ):
        if rows:
            figure, axis = plt.subplots(figsize=(9, 4))
            for column in columns:
                values = np.asarray([_number(row, column) for row in rows])
                if np.isfinite(values).any():
                    axis.plot(values, label=column, linewidth=1)
            axis.set(xlabel="Audit row", ylabel=ylabel)
            axis.legend()
            axis.grid(alpha=0.2)
            figure.tight_layout()
            figure.savefig(output / output_name, dpi=180)
            plt.close(figure)
            status[output_name] = "GENERATED"
        else:
            _save_placeholder(output / output_name, output_name, "NOT_MEASURED")
            status[output_name] = "NOT_MEASURED_PLACEHOLDER"

    alignment = Path(args.alignment_dir) if args.alignment_dir else None
    null_rows = _rows(alignment / "null_max_corr_distribution.csv") if alignment else []
    observed_rows = _rows(alignment / "alignment_significance.csv") if alignment else []
    name = "alignment_observed_vs_null.png"
    null_values = np.asarray([_number(row, "max_correlation") for row in null_rows])
    null_values = null_values[np.isfinite(null_values)]
    if null_values.size:
        figure, axis = plt.subplots(figsize=(7, 4.5))
        axis.hist(null_values, bins=40, alpha=0.75, label="null max correlation")
        for row in observed_rows:
            axis.axvline(_number(row, "observed_max_correlation"), color="red", alpha=0.5)
        axis.set(xlabel="Maximum searched correlation", ylabel="Count")
        axis.grid(alpha=0.2)
        figure.tight_layout()
        figure.savefig(output / name, dpi=180)
        plt.close(figure)
        status[name] = "GENERATED"
    else:
        _save_placeholder(output / name, "Observed versus null alignment", "NOT_MEASURED")
        status[name] = "NOT_MEASURED_PLACEHOLDER"

    stress = Path(args.stress_dir) if args.stress_dir else None
    stress_rows = _rows(stress / "tracking_stress_summary.csv") if stress else []
    stress_plots = (
        ("translation_amplitude_px", "point_rmse_px", "tracking_rmse_vs_amplitude.png"),
        ("translation_amplitude_px", "common_point_rmse_px", "common_error_vs_amplitude.png"),
        ("translation_amplitude_px", "local_residual_rmse_px", "local_error_vs_amplitude.png"),
        ("rotation_amplitude_deg", "rotation_rmse_deg", "rotation_error_vs_rotation.png"),
    )
    for x_name, y_name, output_name in stress_plots:
        x = np.asarray([_number(row, x_name) for row in stress_rows])
        y = np.asarray([_number(row, y_name) for row in stress_rows])
        finite = np.isfinite(x) & np.isfinite(y)
        if finite.any():
            figure, axis = plt.subplots(figsize=(6, 4.5))
            axis.scatter(x[finite], y[finite], s=18, alpha=0.6)
            axis.set(xlabel=x_name, ylabel=y_name)
            axis.grid(alpha=0.2)
            figure.tight_layout()
            figure.savefig(output / output_name, dpi=180)
            plt.close(figure)
            status[output_name] = "GENERATED"
        else:
            _save_placeholder(output / output_name, output_name, "NOT_MEASURED")
            status[output_name] = "NOT_MEASURED_PLACEHOLDER"

    reseed = Path(args.reseed_dir) if args.reseed_dir else None
    reseed_rows = _rows(reseed / "reseed_boundary.csv") if reseed else []
    values = np.asarray([_number(row, "acceleration_spike_ratio") for row in reseed_rows])
    values = values[np.isfinite(values)]
    name = "reseed_jump_distribution.png"
    if values.size:
        figure, axis = plt.subplots(figsize=(6, 4.5))
        axis.hist(values, bins=min(30, max(5, values.size)))
        axis.axvline(3, color="red", linestyle="--", label="review threshold")
        axis.set(xlabel="Acceleration spike / local median", ylabel="Count")
        axis.legend()
        axis.grid(alpha=0.2)
        figure.tight_layout()
        figure.savefig(output / name, dpi=180)
        plt.close(figure)
        status[name] = "GENERATED"
    else:
        _save_placeholder(output / name, "Reseed jump distribution", "NOT_MEASURED")
        status[name] = "NOT_MEASURED_PLACEHOLDER"
    (output / "plot_manifest.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pga-dir")
    parser.add_argument("--offline-pga-dir")
    parser.add_argument("--runtime-dir")
    parser.add_argument("--alignment-dir")
    parser.add_argument("--stress-dir")
    parser.add_argument("--reseed-dir")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(generate(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
