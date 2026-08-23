"""Simple auditable empirical PGA models and leakage-safe group splits."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
from scipy import stats


Algorithm = Literal["median", "single_coefficient", "log_linear", "ridge", "huber"]


def group_kfold_indices(
    groups: Iterable[str], folds: int = 5, seed: int = 0
) -> list[tuple[np.ndarray, np.ndarray]]:
    values = np.asarray(list(groups), dtype=str)
    unique = np.unique(values)
    if folds < 2 or unique.size < folds:
        raise ValueError("fold count must be at least 2 and no larger than unique groups")
    rng = np.random.default_rng(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    fold_groups = np.array_split(shuffled, folds)
    splits = []
    for test_groups in fold_groups:
        test = np.flatnonzero(np.isin(values, test_groups))
        train = np.flatnonzero(~np.isin(values, test_groups))
        if np.intersect1d(values[train], values[test]).size:
            raise RuntimeError("group leakage detected")
        splits.append((train, test))
    return splits


@dataclass
class EmpiricalPGAModel:
    feature_names: tuple[str, ...]
    algorithm: Algorithm = "ridge"
    alpha: float = 1.0
    log_target: bool = True
    requires_scale: bool = True
    model_version: str = "videoeew-pga-v1"
    coefficient: np.ndarray | None = None
    intercept: float = 0.0
    feature_mean: np.ndarray | None = None
    feature_scale: np.ndarray | None = None
    training_metadata: dict[str, object] | None = None

    def _prepare(self, features: np.ndarray, fit: bool) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim == 1:
            values = values[:, None]
        if values.shape[1] != len(self.feature_names):
            raise ValueError("feature column count does not match feature_names")
        if not np.isfinite(values).all():
            raise ValueError("features contain non-finite values")
        if fit:
            self.feature_mean = np.mean(values, axis=0)
            self.feature_scale = np.std(values, axis=0)
            self.feature_scale[self.feature_scale < 1e-12] = 1.0
        if self.feature_mean is None or self.feature_scale is None:
            raise RuntimeError("model has not been fitted")
        return (values - self.feature_mean) / self.feature_scale

    def fit(
        self,
        features: np.ndarray,
        pga_true: np.ndarray,
        *,
        metadata: dict[str, object] | None = None,
    ) -> "EmpiricalPGAModel":
        target = np.asarray(pga_true, dtype=np.float64)
        if target.ndim != 1 or np.any(~np.isfinite(target)) or np.any(target <= 0):
            raise ValueError("PGA target must be a finite positive vector")
        if self.algorithm == "median":
            self.coefficient = np.empty(0, dtype=np.float64)
            self.intercept = float(np.median(target))
            self.feature_mean = np.empty(0, dtype=np.float64)
            self.feature_scale = np.empty(0, dtype=np.float64)
            self.log_target = False
            self.training_metadata = metadata or {}
            return self
        design = self._prepare(features, fit=True)
        response = np.log(target) if self.log_target else target
        if self.algorithm == "single_coefficient":
            if design.shape[1] != 1:
                raise ValueError("single_coefficient requires exactly one feature")
            raw = np.asarray(features, dtype=np.float64).reshape(-1)
            coefficient = float(np.dot(raw, target) / max(np.dot(raw, raw), 1e-12))
            self.coefficient = np.asarray([coefficient])
            self.intercept = 0.0
            self.log_target = False
        else:
            augmented = np.column_stack([np.ones(design.shape[0]), design])
            effective_alpha = self.alpha if self.algorithm in {"ridge", "huber"} else 0.0
            penalty = np.eye(augmented.shape[1]) * effective_alpha
            penalty[0, 0] = 0
            weights = np.ones(design.shape[0])
            iterations = 30 if self.algorithm == "huber" else 1
            parameters = np.zeros(augmented.shape[1])
            for _ in range(iterations):
                weighted = augmented * np.sqrt(weights[:, None])
                weighted_response = response * np.sqrt(weights)
                matrix = weighted.T @ weighted + penalty
                vector = weighted.T @ weighted_response
                parameters = (
                    np.linalg.lstsq(matrix, vector, rcond=None)[0]
                    if self.algorithm == "log_linear"
                    else np.linalg.solve(matrix, vector)
                )
                if self.algorithm == "huber":
                    residual = response - augmented @ parameters
                    scale = 1.4826 * np.median(np.abs(residual - np.median(residual)))
                    threshold = max(1.345 * scale, 1e-8)
                    weights = np.minimum(1.0, threshold / np.maximum(np.abs(residual), 1e-12))
            self.intercept = float(parameters[0])
            self.coefficient = parameters[1:]
        self.training_metadata = metadata or {}
        return self

    def predict(
        self,
        features: np.ndarray,
        *,
        scale_valid: bool = True,
        allow_uncalibrated_evaluation: bool = False,
    ) -> np.ndarray:
        if self.coefficient is None:
            raise RuntimeError("model has not been fitted")
        if self.requires_scale and not scale_valid and not allow_uncalibrated_evaluation:
            raise RuntimeError("PGA prediction rejected because geometric scale is invalid")
        if self.algorithm == "median":
            count = np.asarray(features).shape[0]
            prediction = np.full(count, self.intercept, dtype=np.float64)
        elif self.algorithm == "single_coefficient":
            values = np.asarray(features, dtype=np.float64).reshape(-1, 1)
            prediction = values[:, 0] * self.coefficient[0]
        else:
            design = self._prepare(features, fit=False)
            prediction = self.intercept + design @ self.coefficient
            if self.log_target:
                prediction = np.exp(prediction)
        return np.maximum(prediction, 0.0)

    def save(self, path: str | Path) -> None:
        if self.coefficient is None:
            raise RuntimeError("cannot save an unfitted model")
        payload = asdict(self)
        for key in ("coefficient", "feature_mean", "feature_scale"):
            value = payload[key]
            payload[key] = None if value is None else np.asarray(value).tolist()
        Path(path).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "EmpiricalPGAModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        payload["feature_names"] = tuple(payload["feature_names"])
        for key in ("coefficient", "feature_mean", "feature_scale"):
            if payload[key] is not None:
                payload[key] = np.asarray(payload[key], dtype=np.float64)
        return cls(**payload)


def evaluate_predictions(pga_true: np.ndarray, pga_est: np.ndarray) -> dict[str, float]:
    truth = np.asarray(pga_true, dtype=np.float64)
    estimate = np.asarray(pga_est, dtype=np.float64)
    if truth.shape != estimate.shape or truth.ndim != 1 or np.any(truth <= 0):
        raise ValueError("truth and estimate must be same-shape positive vectors")
    safe_estimate = np.maximum(estimate, 1e-12)
    ratio = np.maximum(safe_estimate / truth, truth / safe_estimate)
    pearson = (
        float(np.corrcoef(truth, estimate)[0, 1])
        if truth.size >= 2 and np.std(truth) > 0 and np.std(estimate) > 0
        else float("nan")
    )
    spearman = (
        float(stats.spearmanr(truth, estimate).statistic)
        if truth.size >= 2 and np.std(truth) > 0 and np.std(estimate) > 0
        else float("nan")
    )
    return {
        "mae": float(np.mean(np.abs(estimate - truth))),
        "rmse": float(np.sqrt(np.mean(np.square(estimate - truth)))),
        "log_pga_mae": float(np.mean(np.abs(np.log(safe_estimate) - np.log(truth)))),
        "median_multiplicative_error": float(np.median(ratio)),
        "fraction_within_factor_1p5": float(np.mean(ratio <= 1.5)),
        "fraction_within_factor_2": float(np.mean(ratio <= 2.0)),
        "pearson": pearson,
        "spearman": spearman,
    }
