"""Selection-bias diagnostics for maximum axis/sign/lag/domain alignment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np

from .alignment import AlignmentResult, estimate_time_offset


@dataclass(frozen=True)
class AlignmentNullSummary:
    record_id: str
    observed_max_correlation: float
    empirical_p_value: float
    iterations: int
    null_method: str
    interpretation: str = "RESEARCH_DIAGNOSTIC_ONLY_NOT_DEPLOYABLE"


def phase_randomized_surrogate(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    samples = np.asarray(values, dtype=np.float64)
    if samples.ndim == 1:
        samples = samples[:, None]
        squeeze = True
    else:
        squeeze = False
    output = np.empty_like(samples)
    for channel in range(samples.shape[1]):
        spectrum = np.fft.rfft(samples[:, channel])
        phases = rng.uniform(0, 2 * np.pi, size=spectrum.size)
        phases[0] = 0.0
        if samples.shape[0] % 2 == 0:
            phases[-1] = 0.0
        surrogate = np.abs(spectrum) * np.exp(1j * phases)
        surrogate[0] = spectrum[0]
        if samples.shape[0] % 2 == 0:
            surrogate[-1] = spectrum[-1]
        output[:, channel] = np.fft.irfft(surrogate, n=samples.shape[0])
    return output[:, 0] if squeeze else output


def maximum_domain_alignment(
    visual_timestamps: np.ndarray,
    visual_domains: Mapping[str, np.ndarray],
    sensor_timestamps: np.ndarray,
    sensor_domains: Mapping[str, np.ndarray],
    *,
    offset_range_s: tuple[float, float],
    step_s: float | None = None,
) -> tuple[str, AlignmentResult]:
    best_name = ""
    best: AlignmentResult | None = None
    for name in sorted(set(visual_domains).intersection(sensor_domains)):
        candidate = estimate_time_offset(
            visual_timestamps,
            visual_domains[name],
            sensor_timestamps,
            sensor_domains[name],
            offset_range_s=offset_range_s,
            step_s=step_s,
            min_correlation=2.0,
        )
        if best is None or candidate.correlation > best.correlation:
            best_name = name
            best = candidate
    if best is None:
        raise ValueError("visual and sensor domains do not overlap")
    return best_name, best


def alignment_null_test(
    record_id: str,
    visual_timestamps: np.ndarray,
    visual_domains: Mapping[str, np.ndarray],
    sensor_timestamps: np.ndarray,
    sensor_domains: Mapping[str, np.ndarray],
    *,
    offset_range_s: tuple[float, float],
    iterations: int = 1000,
    seed: int = 0,
    null_method: str = "circular_shift",
    step_s: float | None = None,
) -> tuple[AlignmentNullSummary, list[dict[str, float | int | str]], dict[str, object]]:
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if null_method not in {"circular_shift", "phase_randomized"}:
        raise ValueError("unsupported alignment null method")
    observed_domain, observed = maximum_domain_alignment(
        visual_timestamps,
        visual_domains,
        sensor_timestamps,
        sensor_domains,
        offset_range_s=offset_range_s,
        step_s=step_s,
    )
    rng = np.random.default_rng(seed)
    distribution: list[dict[str, float | int | str]] = []
    sensor_length = min(np.asarray(values).shape[0] for values in sensor_domains.values())
    minimum_shift = max(1, int(round(0.05 * sensor_length)))
    for iteration in range(iterations):
        if null_method == "circular_shift":
            shift = int(rng.integers(minimum_shift, max(minimum_shift + 1, sensor_length)))
            surrogate_domains = {
                name: np.roll(np.asarray(values), shift, axis=0)
                for name, values in sensor_domains.items()
            }
        else:
            shift = 0
            surrogate_domains = {
                name: phase_randomized_surrogate(np.asarray(values), rng)
                for name, values in sensor_domains.items()
            }
        domain, candidate = maximum_domain_alignment(
            visual_timestamps,
            visual_domains,
            sensor_timestamps,
            surrogate_domains,
            offset_range_s=offset_range_s,
            step_s=step_s,
        )
        distribution.append(
            {
                "record_id": record_id,
                "iteration": iteration,
                "null_method": null_method,
                "circular_shift_samples": shift,
                "max_correlation": candidate.correlation,
                "selected_domain": domain,
                "selected_offset_s": candidate.offset_s,
                "selected_visual_channel": candidate.visual_channel,
                "selected_sensor_channel": candidate.sensor_channel,
                "selected_polarity": candidate.polarity,
            }
        )
    null_values = np.asarray([float(row["max_correlation"]) for row in distribution])
    p_value = float((1 + np.sum(null_values >= observed.correlation)) / (iterations + 1))
    summary = AlignmentNullSummary(
        record_id=str(record_id),
        observed_max_correlation=float(observed.correlation),
        empirical_p_value=p_value,
        iterations=int(iterations),
        null_method=null_method,
    )
    candidate = {
        "record_id": str(record_id),
        "observed_domain": observed_domain,
        **asdict(observed),
        "search_min_s": float(offset_range_s[0]),
        "search_max_s": float(offset_range_s[1]),
        "search_includes_axis_sign_lag_domain": True,
    }
    return summary, distribution, candidate


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must be a one-dimensional vector in [0,1]")
    order = np.argsort(values)
    ranked = values[order] * values.size / np.arange(1, values.size + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty_like(ranked)
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted
