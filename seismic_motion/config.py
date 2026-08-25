"""Versioned configuration loading and validation."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigError(ValueError):
    """Raised when a runtime configuration is incomplete or inconsistent."""


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping and validate cross-field runtime invariants."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ConfigError("configuration root must be a mapping")
    resolved = copy.deepcopy(loaded)
    validate_config(resolved)
    return resolved


def validate_config(config: Mapping[str, Any]) -> None:
    required = {"video", "tracker", "motion", "signal", "runtime", "scale", "pga"}
    missing = sorted(required.difference(config))
    if missing:
        raise ConfigError(f"missing configuration sections: {', '.join(missing)}")

    tracker = config["tracker"]
    runtime = config["runtime"]
    signal = config["signal"]
    if int(tracker["window_len"]) < 2:
        raise ConfigError("tracker.window_len must be at least 2")
    if int(tracker["step"]) <= 0 or int(tracker["step"]) > int(tracker["window_len"]):
        raise ConfigError("tracker.step must be in [1, window_len]")
    if "source_timestamp_future_context_frames" in tracker:
        minimum, maximum = (
            int(value) for value in tracker["source_timestamp_future_context_frames"]
        )
        expected = (int(tracker["window_len"]) - int(tracker["step"]), int(tracker["window_len"]) - 1)
        if (minimum, maximum) != expected:
            raise ConfigError(
                "tracker.source_timestamp_future_context_frames must explicitly "
                f"match the finalized-block range {list(expected)}"
            )
    if int(tracker["num_points"]) <= 0:
        raise ConfigError("tracker.num_points must be positive")
    if float(runtime["target_fps"]) <= 0:
        raise ConfigError("runtime.target_fps must be positive")
    if int(runtime["max_queue_blocks"]) <= 0:
        raise ConfigError("runtime.max_queue_blocks must be positive")
    low, high = (float(value) for value in signal["bandpass_hz"])
    if not 0 <= low < high:
        raise ConfigError("signal.bandpass_hz must be increasing and non-negative")


def canonical_json(config: Mapping[str, Any]) -> str:
    """Return deterministic JSON used for hashing and manifests."""

    return json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def config_sha256(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()
