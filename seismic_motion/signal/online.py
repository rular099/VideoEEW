"""Strictly causal bounded-state common/local motion processing."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .derivatives import CausalDerivativeEstimator
from .filtering import CausalBandpassFilter


@dataclass(frozen=True)
class OnlineSignalState:
    timestamp: float
    filtered_common_x: float
    filtered_common_y: float
    common_velocity_x: float
    common_velocity_y: float
    common_acceleration_x: float
    common_acceleration_y: float
    local_motion_rms: float
    local_velocity_rms: float
    local_acceleration_rms: float
    quality_state: str
    derivative_method: str
    startup_state: str
    samples_seen: int
    retained_history_samples: int
    filter_state_bytes: int
    reseed_id: int
    reseed_boundary: bool

    def as_dict(self) -> dict[str, float | int | str | bool]:
        return asdict(self)


class OnlineSignalProcessor:
    """Update online motion signals without look-ahead or unbounded history.

    The Butterworth bandpass already contains a causal high-pass stage, so this
    path deliberately does not run full-record linear detrending.  Local point
    identities and shape must remain stable within one processor instance.
    """

    def __init__(
        self,
        *,
        sample_rate_hz: float,
        bandpass_hz: tuple[float, float] = (0.3, 8.0),
        filter_order: int = 4,
        derivative_method: str = "causal_polynomial",
        window_length: int = 9,
        polynomial_order: int = 3,
    ) -> None:
        self.sample_rate_hz = float(sample_rate_hz)
        self.bandpass_hz = tuple(float(value) for value in bandpass_hz)
        self.derivative_method = derivative_method
        self.window_length = int(window_length)
        self.polynomial_order = int(polynomial_order)
        self._common_filter = CausalBandpassFilter(
            self.sample_rate_hz, self.bandpass_hz, order=filter_order
        )
        self._local_filter = CausalBandpassFilter(
            self.sample_rate_hz, self.bandpass_hz, order=filter_order
        )
        self._common_derivative = CausalDerivativeEstimator(
            method=derivative_method,
            window_length=window_length,
            polynomial_order=polynomial_order,
        )
        self._local_derivative = CausalDerivativeEstimator(
            method=derivative_method,
            window_length=window_length,
            polynomial_order=polynomial_order,
        )
        self._last_reseed_id: int | None = None

    @property
    def retained_history_samples(self) -> int:
        return max(
            self._common_derivative.retained_samples,
            self._local_derivative.retained_samples,
        )

    def update(
        self,
        timestamp: float,
        common_motion: np.ndarray,
        local_motion: np.ndarray,
        quality: str,
        *,
        reseed_id: int = 0,
    ) -> OnlineSignalState:
        common = np.asarray(common_motion, dtype=np.float64)
        local = np.asarray(local_motion, dtype=np.float64)
        if common.shape != (2,):
            raise ValueError("common_motion must have shape [2]")
        if local.ndim != 2 or local.shape[1] != 2 or local.shape[0] == 0:
            raise ValueError("local_motion must have shape [N,2] with N > 0")
        if not np.isfinite(common).all() or not np.isfinite(local).all():
            raise ValueError("online motion inputs must be finite")

        filtered_common = self._common_filter.process(common[None, :])[0]
        filtered_local = self._local_filter.process(local.reshape(1, -1))[0].reshape(local.shape)
        common_state = self._common_derivative.update(timestamp, filtered_common)
        local_state = self._local_derivative.update(timestamp, filtered_local)
        reseed_boundary = self._last_reseed_id is not None and reseed_id != self._last_reseed_id
        self._last_reseed_id = int(reseed_id)

        def rms_vector(values: np.ndarray) -> float:
            if not np.isfinite(values).all():
                return float("nan")
            return float(np.sqrt(np.mean(np.sum(np.square(values), axis=1))))

        return OnlineSignalState(
            timestamp=float(timestamp),
            filtered_common_x=float(filtered_common[0]),
            filtered_common_y=float(filtered_common[1]),
            common_velocity_x=float(common_state.velocity[0]),
            common_velocity_y=float(common_state.velocity[1]),
            common_acceleration_x=float(common_state.acceleration[0]),
            common_acceleration_y=float(common_state.acceleration[1]),
            local_motion_rms=rms_vector(filtered_local),
            local_velocity_rms=rms_vector(local_state.velocity),
            local_acceleration_rms=rms_vector(local_state.acceleration),
            quality_state=str(quality),
            derivative_method=self.derivative_method,
            startup_state=common_state.startup_state,
            samples_seen=common_state.samples_seen,
            retained_history_samples=self.retained_history_samples,
            filter_state_bytes=self._common_filter.state_nbytes + self._local_filter.state_nbytes,
            reseed_id=int(reseed_id),
            reseed_boundary=reseed_boundary,
        )
