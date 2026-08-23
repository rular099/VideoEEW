"""Leakage-safe cross-validation for simple empirical coarse-PGA models."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from seismic_motion.pga.model import (
    EmpiricalPGAModel,
    evaluate_predictions,
    group_kfold_indices,
)


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _safe_group_metrics(truth: np.ndarray, estimate: np.ndarray) -> dict[str, float | int]:
    ratio = np.maximum(
        np.maximum(estimate, 1e-12) / truth,
        truth / np.maximum(estimate, 1e-12),
    )
    payload: dict[str, float | int] = {
        "count": int(truth.size),
        "mae": float(np.mean(np.abs(estimate - truth))),
        "rmse": float(np.sqrt(np.mean(np.square(estimate - truth)))),
        "median_multiplicative_error": float(np.median(ratio)),
        "fraction_within_factor_2": float(np.mean(ratio <= 2.0)),
    }
    return payload


def _bin_indices(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.digitize(values, edges[1:-1], right=False), 0, edges.size - 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="CSV with features, target and group")
    parser.add_argument("--config", default="configs/pga_train.yaml")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    input_rows = _read_rows(args.dataset)
    row_filter = config.get("row_filter", {})
    minimum_alignment = float(row_filter.get("minimum_alignment_correlation", 0.0))
    reject_quality_flags = tuple(str(value) for value in row_filter.get("reject_quality_flags", []))
    rows = []
    excluded_rows: list[dict[str, str]] = []
    for row in input_rows:
        correlation = float(row.get("alignment_correlation", "inf"))
        flags = str(row.get("quality_flags", ""))
        accepted = correlation >= minimum_alignment and not any(
            rejected in flags for rejected in reject_quality_flags
        )
        (rows if accepted else excluded_rows).append(row)
    if not rows:
        raise ValueError("row_filter rejected every dataset row")
    feature_names = tuple(config["features"])
    target_name = str(config["target"])
    group_name = str(config["split"]["group_column"])
    features = np.asarray(
        [[float(row[name]) for name in feature_names] for row in rows], dtype=np.float64
    )
    target = np.asarray([float(row[target_name]) for row in rows], dtype=np.float64)
    groups = np.asarray([row[group_name] for row in rows], dtype=str)
    folds = group_kfold_indices(
        groups,
        folds=int(config["split"]["folds"]),
        seed=int(config["split"]["seed"]),
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    algorithm_metrics: dict[str, dict[str, float]] = {}
    all_predictions: dict[str, np.ndarray] = {}
    algorithm_layout: dict[str, tuple[tuple[str, ...], list[int]]] = {}
    for algorithm in config["algorithms"]:
        predictions = np.full(target.shape, np.nan)
        names = (
            ("common_peak_acceleration_px_s2",)
            if algorithm == "single_coefficient"
            else feature_names
        )
        columns = (
            [feature_names.index("common_peak_acceleration_px_s2")]
            if algorithm == "single_coefficient"
            else list(range(len(feature_names)))
        )
        algorithm_layout[str(algorithm)] = (names, columns)
        for train, test in folds:
            model = EmpiricalPGAModel(
                feature_names=names,
                algorithm=algorithm,
                alpha=float(config["model"]["ridge_alpha"]),
                log_target=bool(config["model"]["log_target"]),
                requires_scale=bool(config["model"]["requires_scale"]),
            )
            model.fit(features[train][:, columns], target[train])
            predictions[test] = model.predict(
                features[test][:, columns],
                scale_valid=False,
                allow_uncalibrated_evaluation=True,
            )
        algorithm_metrics[algorithm] = evaluate_predictions(target, predictions)
        all_predictions[algorithm] = predictions
    primary = str(config["primary_algorithm"])
    primary_names, primary_columns = algorithm_layout[primary]
    final_model = EmpiricalPGAModel(
        feature_names=primary_names,
        algorithm=primary,
        alpha=float(config["model"]["ridge_alpha"]),
        log_target=bool(config["model"]["log_target"]),
        requires_scale=bool(config["model"]["requires_scale"]),
    ).fit(
        features[:, primary_columns],
        target,
        metadata={
            "group_column": group_name,
            "unique_groups": int(np.unique(groups).size),
            "offline_uncalibrated_evaluation": True,
            "deployment_prediction_allowed": False,
            "selection_policy": "predeclared simple baseline; not selected on test-fold performance",
        },
    )
    final_model.save(output / "pga_model.json")
    with (output / "pga_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["row_index", group_name, "PGA_true", *all_predictions, "confidence", "scale_state"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(target.size):
            writer.writerow(
                {
                    "row_index": index,
                    group_name: groups[index],
                    "PGA_true": target[index],
                    **{name: prediction[index] for name, prediction in all_predictions.items()},
                    "confidence": "LOW_RESEARCH_ONLY",
                    "scale_state": "UNCALIBRATED",
                }
            )
    metrics = {
        "target": target_name,
        "group_column": group_name,
        "rows": len(rows),
        "input_rows": len(input_rows),
        "excluded_rows": len(excluded_rows),
        "excluded_record_ids": [row.get("record_id", "") for row in excluded_rows],
        "row_filter": row_filter,
        "unique_groups": int(np.unique(groups).size),
        "algorithms": algorithm_metrics,
        "deployment_prediction_allowed": False,
        "reason": "geometric scale missing; results are offline setup-specific research evaluation",
    }
    bin_edges = np.asarray(config.get("pga_bins_gal", [0, 50, 100, 200, 400, 1000]), dtype=float)
    if bin_edges.ndim != 1 or bin_edges.size < 3 or np.any(np.diff(bin_edges) <= 0):
        raise ValueError("pga_bins_gal must be a strictly increasing vector")
    truth_bins = _bin_indices(target, bin_edges)
    metrics["pga_bin_edges_gal"] = bin_edges.tolist()
    group_columns = [
        str(name)
        for name in config.get(
            "report_group_columns",
            ["record_id", "camera_id", "site_id", "quality_flags", "alignment_status"],
        )
        if rows and name in rows[0]
    ]
    group_rows: list[dict[str, object]] = []
    for algorithm, predictions in all_predictions.items():
        predicted_bins = _bin_indices(predictions, bin_edges)
        confusion = np.zeros((bin_edges.size - 1, bin_edges.size - 1), dtype=int)
        for truth_bin, prediction_bin in zip(truth_bins, predicted_bins):
            confusion[truth_bin, prediction_bin] += 1
        with (output / f"pga_confusion_{algorithm}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(["true_bin\\pred_bin", *range(confusion.shape[1])])
            for index, values in enumerate(confusion):
                writer.writerow([index, *values])
        for bin_index in range(bin_edges.size - 1):
            selection = truth_bins == bin_index
            if np.any(selection):
                group_rows.append(
                    {
                        "algorithm": algorithm,
                        "group_column": "PGA_true_bin",
                        "group_value": f"[{bin_edges[bin_index]},{bin_edges[bin_index + 1]})",
                        **_safe_group_metrics(target[selection], predictions[selection]),
                    }
                )
        for column in group_columns:
            values = np.asarray([row[column] for row in rows], dtype=str)
            for value in sorted(np.unique(values)):
                selection = values == value
                group_rows.append(
                    {
                        "algorithm": algorithm,
                        "group_column": column,
                        "group_value": value,
                        **_safe_group_metrics(target[selection], predictions[selection]),
                    }
                )
    if group_rows:
        with (output / "pga_metrics_by_group.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(group_rows[0]))
            writer.writeheader()
            writer.writerows(group_rows)
    (output / "pga_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
