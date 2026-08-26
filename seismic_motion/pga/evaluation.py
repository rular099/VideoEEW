"""Truth-blind subset construction and auditable PGA research evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .model import EmpiricalPGAModel, evaluate_predictions, group_kfold_indices


ALGORITHMS = ("median", "single_coefficient", "log_linear", "ridge", "huber")


def read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _finite(row: dict[str, str], name: str) -> float | None:
    try:
        value = float(row.get(name, ""))
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def video_quality_decision(
    row: dict[str, str], quality_config: dict[str, Any]
) -> tuple[bool, str, str]:
    """Apply only fields available from video/runtime at deployment time."""

    failures: list[str] = []
    unknown: list[str] = []
    rules: tuple[tuple[str, Callable[[float, float], bool], str], ...] = (
        ("mean_active_track_count", lambda value, threshold: value >= threshold, "min_tracks"),
        ("mean_visibility", lambda value, threshold: value >= threshold, "min_visibility"),
        ("mean_inlier_ratio", lambda value, threshold: value >= threshold, "min_inlier_ratio"),
        ("mean_fit_rmse_px", lambda value, threshold: value <= threshold, "max_fit_rmse_px"),
        ("estimated_missing_frames", lambda value, threshold: value <= threshold, "max_missing_frames"),
        ("timestamp_rms_jitter_s", lambda value, threshold: value <= threshold, "max_timestamp_jitter_s"),
    )
    for column, comparison, config_name in rules:
        threshold = quality_config.get(config_name)
        if threshold is None:
            continue
        value = _finite(row, column)
        if value is None:
            unknown.append(column)
        elif not comparison(value, float(threshold)):
            failures.append(f"{column}_OUT_OF_RANGE")
    flags = str(row.get("quality_flags", ""))
    if "INVALID" in flags:
        failures.append("MOTION_QUALITY_INVALID")
    for optional in ("spatial_coverage", "motion_model_condition_number", "blur_metric", "exposure_instability"):
        if _finite(row, optional) is None:
            unknown.append(optional)
    if failures:
        return False, "REJECTED", "|".join(sorted(set(failures)))
    state = "PASS_WITH_UNKNOWN_METRICS" if unknown else "PASS"
    reason = "UNKNOWN:" + "|".join(sorted(set(unknown))) if unknown else ""
    return True, state, reason


def _algorithm_layout(
    algorithm: str, feature_names: tuple[str, ...]
) -> tuple[tuple[str, ...], list[int]]:
    if algorithm == "median":
        return (), []
    if algorithm == "single_coefficient":
        name = "common_peak_acceleration_px_s2"
        return (name,), [feature_names.index(name)]
    return feature_names, list(range(len(feature_names)))


def cross_validated_predictions(
    features: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    feature_names: tuple[str, ...],
    *,
    folds: int,
    seed: int,
    ridge_alpha: float,
) -> dict[str, np.ndarray]:
    unique_groups = np.unique(groups)
    actual_folds = min(int(folds), int(unique_groups.size))
    if actual_folds < 2:
        raise ValueError("at least two groups are required for cross-validation")
    splits = group_kfold_indices(groups, folds=actual_folds, seed=seed)
    outputs: dict[str, np.ndarray] = {}
    for algorithm in ALGORITHMS:
        names, columns = _algorithm_layout(algorithm, feature_names)
        prediction = np.full(target.shape, np.nan, dtype=np.float64)
        for train, test in splits:
            model = EmpiricalPGAModel(
                feature_names=names,
                algorithm=algorithm,  # type: ignore[arg-type]
                alpha=ridge_alpha,
                log_target=algorithm in {"log_linear", "ridge", "huber"},
                requires_scale=True,
            ).fit(features[train][:, columns], target[train])
            prediction[test] = model.predict(
                features[test][:, columns],
                scale_valid=False,
                allow_uncalibrated_evaluation=True,
            )
        outputs[algorithm] = prediction
    return outputs


def bootstrap_intervals(
    truth: np.ndarray,
    estimate: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    metrics: dict[str, list[float]] = {}
    for _ in range(iterations):
        indices = rng.integers(0, truth.size, size=truth.size)
        sample = evaluate_predictions(truth[indices], estimate[indices])
        for name, value in sample.items():
            if np.isfinite(value):
                metrics.setdefault(name, []).append(float(value))
    output = {}
    for name, values in metrics.items():
        array = np.asarray(values, dtype=np.float64)
        output[name] = {
            "lower_95": float(np.percentile(array, 2.5)),
            "median": float(np.percentile(array, 50)),
            "upper_95": float(np.percentile(array, 97.5)),
            "bootstrap_iterations": int(iterations),
        }
    return output


def evaluate_subsets(
    input_rows: list[dict[str, str]],
    config: dict[str, Any],
    output_directory: str | Path,
) -> dict[str, Any]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    expected_feature_version = str(config["feature_version"])
    feature_versions = {
        str(row.get("feature_version", "MISSING")) for row in input_rows
    }
    if feature_versions != {expected_feature_version}:
        raise ValueError(
            "dataset feature versions do not match config: "
            f"expected {expected_feature_version}, observed {sorted(feature_versions)}"
        )
    feature_names = tuple(str(value) for value in config["features"])
    target_name = str(config["target"])
    target = np.asarray(
        [float(row[target_name]) for row in input_rows], dtype=np.float64
    )
    features = np.asarray(
        [[float(row[name]) for name in feature_names] for row in input_rows],
        dtype=np.float64,
    )
    finite_model_input = np.isfinite(target) & (target > 0) & np.isfinite(features).all(axis=1)
    quality_decisions = [
        video_quality_decision(row, config["video_quality"]) for row in input_rows
    ]
    quality_mask = np.asarray([decision[0] for decision in quality_decisions], dtype=bool)
    posthoc_threshold = float(config["posthoc"]["exploratory_min_correlation"])
    posthoc_mask = np.asarray(
        [
            (_finite(row, "alignment_correlation") or -np.inf) >= posthoc_threshold
            for row in input_rows
        ],
        dtype=bool,
    )
    subset_masks = {
        "all": finite_model_input,
        "video_quality": finite_model_input & quality_mask,
        "posthoc_aligned": finite_model_input & posthoc_mask,
    }
    subset_files = {
        "all": "pga_eval_all.csv",
        "video_quality": "pga_eval_video_quality.csv",
        "posthoc_aligned": "pga_eval_posthoc_aligned.csv",
    }
    all_metrics: dict[str, Any] = {}
    bootstrap: dict[str, Any] = {}
    for subset_name, mask in subset_masks.items():
        indices = np.flatnonzero(mask)
        metrics_payload: dict[str, Any] = {
            "input_rows": len(input_rows),
            "included_rows": int(indices.size),
            "selection_uses_strong_motion": subset_name == "posthoc_aligned",
            "interpretation": (
                "RESEARCH_DIAGNOSTIC_ONLY_NOT_DEPLOYABLE"
                if subset_name == "posthoc_aligned"
                else "PRIMARY_TRUTH_BLIND_SELECTION"
            ),
        }
        predictions: dict[str, np.ndarray] = {}
        if indices.size >= 2:
            subset_groups = np.asarray(
                [str(input_rows[index].get("record_id", index)) for index in indices]
            )
            predictions = cross_validated_predictions(
                features[indices],
                target[indices],
                subset_groups,
                feature_names,
                folds=int(config["split"]["folds"]),
                seed=int(config["split"]["seed"]),
                ridge_alpha=float(config["model"]["ridge_alpha"]),
            )
            metrics_payload["algorithms"] = {
                name: evaluate_predictions(target[indices], values)
                for name, values in predictions.items()
            }
            primary = str(config["primary_algorithm"])
            bootstrap[subset_name] = bootstrap_intervals(
                target[indices],
                predictions[primary],
                iterations=int(config["bootstrap"]["iterations"]),
                seed=int(config["bootstrap"]["seed"]),
            )
        else:
            metrics_payload["algorithms"] = {}
            metrics_payload["not_evaluable_reason"] = "fewer_than_two_included_records"
        all_metrics[subset_name] = metrics_payload

        primary_predictions = np.full(len(input_rows), np.nan, dtype=np.float64)
        if predictions:
            primary_predictions[indices] = predictions[str(config["primary_algorithm"])]
        fields = [
            "record_id",
            "event_id",
            "camera_id",
            "site_id",
            "fps",
            "duration",
            "scale_state",
            "quality_state",
            "true_pga_gal",
            "predicted_pga_gal",
            "abs_error_gal",
            "multiplicative_error",
            "included",
            "exclusion_reason",
            "causal",
            "interpretation",
            *[f"predicted_{name}_gal" for name in ALGORITHMS],
        ]
        with (output / subset_files[subset_name]).open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            prediction_by_algorithm = {
                name: dict(zip(indices.tolist(), values.tolist()))
                for name, values in predictions.items()
            }
            for index, row in enumerate(input_rows):
                included = bool(mask[index])
                predicted = primary_predictions[index]
                reasons: list[str] = []
                if not finite_model_input[index]:
                    reasons.append("NONFINITE_MODEL_INPUT")
                if subset_name == "video_quality" and not quality_mask[index]:
                    reasons.append(quality_decisions[index][2] or "VIDEO_QUALITY_REJECTED")
                if subset_name == "posthoc_aligned" and not posthoc_mask[index]:
                    reasons.append("EXPLORATORY_CORRELATION_BELOW_THRESHOLD")
                ratio = (
                    max(predicted / target[index], target[index] / max(predicted, 1e-12))
                    if included and np.isfinite(predicted)
                    else float("nan")
                )
                writer.writerow(
                    {
                        "record_id": row.get("record_id", "UNKNOWN"),
                        "event_id": "UNKNOWN",
                        "camera_id": "UNKNOWN",
                        "site_id": "UNKNOWN",
                        "fps": row.get("effective_fps", "UNKNOWN"),
                        "duration": (
                            (_finite(row, "window_end") or 0.0)
                            - (_finite(row, "window_start") or 0.0)
                        ),
                        "scale_state": row.get("scale_state", "UNCALIBRATED"),
                        "quality_state": quality_decisions[index][1],
                        "true_pga_gal": target[index],
                        "predicted_pga_gal": predicted,
                        "abs_error_gal": (
                            abs(predicted - target[index]) if np.isfinite(predicted) else ""
                        ),
                        "multiplicative_error": ratio,
                        "included": included,
                        "exclusion_reason": "|".join(reasons),
                        "causal": row.get("causal", "UNKNOWN"),
                        "interpretation": metrics_payload["interpretation"],
                        **{
                            f"predicted_{name}_gal": prediction_by_algorithm.get(name, {}).get(index, "")
                            for name in ALGORITHMS
                        },
                    }
                )
    causal_values = {str(row.get("causal", "UNKNOWN")) for row in input_rows}
    quality_indices = np.flatnonzero(subset_masks["video_quality"])
    if quality_indices.size >= 2:
        primary = str(config["primary_algorithm"])
        names, columns = _algorithm_layout(primary, feature_names)
        final_model = EmpiricalPGAModel(
            feature_names=names,
            algorithm=primary,  # type: ignore[arg-type]
            alpha=float(config["model"]["ridge_alpha"]),
            log_target=primary in {"log_linear", "ridge", "huber"},
            requires_scale=True,
        ).fit(
            features[quality_indices][:, columns],
            target[quality_indices],
            metadata={
                "subset": "VIDEO_QUALITY_ONLY",
                "selection_uses_strong_motion": False,
                "metadata_relationships": "UNKNOWN",
                "deployment_prediction_allowed": False,
                "interpretation": "RESEARCH_ONLY_UNCALIBRATED",
                "causal_input_values": sorted(causal_values),
            },
        )
        final_model.save(output / "pga_model_research.json")
    group_status = {
        "record_group_cv": "PROVISIONAL_UNKNOWN_RECORD_RELATIONSHIPS",
        "event_group_cv": "NOT_EVALUABLE_METADATA_UNKNOWN",
        "camera_group_cv": "NOT_EVALUABLE_METADATA_UNKNOWN",
        "site_group_cv": "NOT_EVALUABLE_METADATA_UNKNOWN",
    }
    payload = {
        "feature_version": expected_feature_version,
        "input_feature_versions": sorted(feature_versions),
        "target": target_name,
        "primary_algorithm": config["primary_algorithm"],
        "subsets": all_metrics,
        "group_cv_status": group_status,
        "causal_input_values": sorted(causal_values),
        "causal_pga_status": "PASS" if causal_values == {"1"} else "FAIL",
        "deployment_prediction_allowed": False,
        "scientific_validity": "RESEARCH_ONLY",
        "geometric_scale": "UNCALIBRATED",
    }
    (output / "pga_metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "pga_bootstrap_ci.json").write_text(
        json.dumps(bootstrap, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload
