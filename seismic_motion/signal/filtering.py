"""Causal runtime and zero-phase evaluation filters with explicit modes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class FilterResult:
    values: np.ndarray
    mode: str
    bandpass_hz: tuple[float, float]
    sample_rate_hz: float
    algorithmic_latency_samples: int


def detrend(values: np.ndarray) -> np.ndarray:
    samples = np.asarray(values, dtype=np.float64)
    return signal.detrend(samples, axis=0, type="linear")


def design_bandpass(
    sample_rate_hz: float,
    bandpass_hz: tuple[float, float],
    order: int = 4,
) -> np.ndarray:
    low, high = (float(value) for value in bandpass_hz)
    nyquist = sample_rate_hz / 2
    if not 0 < low < high < nyquist:
        raise ValueError(
            f"bandpass must satisfy 0 < low < high < Nyquist ({nyquist:g} Hz)"
        )
    return signal.butter(order, [low, high], btype="bandpass", fs=sample_rate_hz, output="sos")


def bandpass_filter(
    values: np.ndarray,
    sample_rate_hz: float,
    bandpass_hz: tuple[float, float] = (0.3, 8.0),
    *,
    causal: bool,
    order: int = 4,
    apply_detrend: bool = True,
) -> FilterResult:
    samples = np.asarray(values, dtype=np.float64)
    working = detrend(samples) if apply_detrend else samples.copy()
    sos = design_bandpass(sample_rate_hz, bandpass_hz, order=order)
    if causal:
        filtered = signal.sosfilt(sos, working, axis=0)
        mode = "causal_sos"
        latency = 0
    else:
        minimum = 3 * (2 * sos.shape[0] + 1)
        if working.shape[0] <= minimum:
            raise ValueError(f"zero-phase filtering requires more than {minimum} samples")
        filtered = signal.sosfiltfilt(sos, working, axis=0)
        mode = "offline_zero_phase"
        latency = 0
    return FilterResult(filtered, mode, bandpass_hz, float(sample_rate_hz), latency)


class CausalBandpassFilter:
    """Stateful SOS filter for bounded-latency runtime blocks."""

    def __init__(
        self,
        sample_rate_hz: float,
        bandpass_hz: tuple[float, float] = (0.3, 8.0),
        order: int = 4,
    ) -> None:
        self.sos = design_bandpass(sample_rate_hz, bandpass_hz, order=order)
        self._state: np.ndarray | None = None
        self.samples_seen = 0

    def process(self, values: np.ndarray) -> np.ndarray:
        samples = np.asarray(values, dtype=np.float64)
        if samples.ndim == 1:
            samples = samples[:, None]
            squeeze = True
        else:
            squeeze = False
        if self._state is None:
            initial = signal.sosfilt_zi(self.sos)
            self._state = np.repeat(initial[:, :, None], samples.shape[1], axis=2)
            self._state *= samples[0][None, None, :]
        filtered, self._state = signal.sosfilt(
            self.sos, samples, axis=0, zi=self._state
        )
        self.samples_seen += int(samples.shape[0])
        return filtered[:, 0] if squeeze else filtered

    @property
    def state_nbytes(self) -> int:
        return 0 if self._state is None else int(self._state.nbytes)
