"""Causal running motion proxy and explicitly research-only PGA estimates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class RunningPGAState:
    timestamp: float
    acceleration_proxy_instant_px_s2: float
    acceleration_proxy_running_px_s2: float
    pga_instant_est_gal: float | None
    pga_running_est_gal: float | None
    confidence: str
    scale_state: str
    deployment_prediction_allowed: bool
    interpretation: str

    def as_dict(self) -> dict[str, float | str | bool | None]:
        return asdict(self)


class RunningPGAEstimator:
    """Maintain monotone running peaks with no future input.

    A Gal-valued estimate is emitted without a geometric scale only when the
    caller explicitly opts into research-only evaluation.  It is never marked
    deployment-valid in that mode.
    """

    def __init__(
        self,
        coefficient_gal_per_px_s2: float | None = None,
        *,
        allow_uncalibrated_research: bool = False,
    ) -> None:
        if coefficient_gal_per_px_s2 is not None and coefficient_gal_per_px_s2 < 0:
            raise ValueError("PGA coefficient must be non-negative")
        self.coefficient = coefficient_gal_per_px_s2
        self.allow_uncalibrated_research = bool(allow_uncalibrated_research)
        self._running_proxy = 0.0
        self._running_pga = 0.0
        self._last_timestamp: float | None = None

    def update(
        self,
        timestamp: float,
        feature_vector_t: Mapping[str, float] | float,
        *,
        quality: str,
        scale_valid: bool,
    ) -> RunningPGAState:
        if self._last_timestamp is not None and timestamp <= self._last_timestamp:
            raise ValueError("running PGA timestamps must be strictly increasing")
        self._last_timestamp = float(timestamp)
        if isinstance(feature_vector_t, Mapping):
            x = float(feature_vector_t["common_acceleration_magnitude_px_s2"])
        else:
            x = float(feature_vector_t)
        proxy = abs(x) if np.isfinite(x) else float("nan")
        if np.isfinite(proxy):
            self._running_proxy = max(self._running_proxy, proxy)

        research_uncalibrated = not scale_valid and self.allow_uncalibrated_research
        may_emit_gal = self.coefficient is not None and (scale_valid or research_uncalibrated)
        pga_instant = None
        pga_running = None
        if may_emit_gal and np.isfinite(proxy):
            pga_instant = proxy * float(self.coefficient)
            self._running_pga = max(self._running_pga, pga_instant)
            pga_running = self._running_pga
        deployment_allowed = bool(scale_valid and self.coefficient is not None)
        interpretation = (
            "DEPLOYMENT_CALIBRATED"
            if deployment_allowed
            else "RESEARCH_ONLY_UNCALIBRATED"
            if research_uncalibrated and self.coefficient is not None
            else "PIXEL_PROXY_ONLY"
        )
        confidence = "INVALID" if quality == "INVALID" or not np.isfinite(proxy) else (
            "LOW_RESEARCH_ONLY" if not deployment_allowed else str(quality)
        )
        return RunningPGAState(
            timestamp=float(timestamp),
            acceleration_proxy_instant_px_s2=proxy,
            acceleration_proxy_running_px_s2=self._running_proxy,
            pga_instant_est_gal=pga_instant,
            pga_running_est_gal=pga_running,
            confidence=confidence,
            scale_state="VALID" if scale_valid else "UNCALIBRATED",
            deployment_prediction_allowed=deployment_allowed,
            interpretation=interpretation,
        )
