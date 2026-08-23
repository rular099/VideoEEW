#!/usr/bin/env python3
"""Validate an exported feature encoder in a separate ONNXRuntime process."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--metrics", required=True)
    args = parser.parse_args()
    import onnx
    import onnxruntime as ort

    onnx_path = Path(args.onnx)
    onnx.checker.check_model(onnx.load(str(onnx_path)))
    with np.load(args.reference) as reference:
        inputs = np.asarray(reference["rgb_0_255"])
        expected = np.asarray(reference["normalized_features"])
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"rgb_0_255": inputs})[0]
    absolute = np.abs(actual - expected)
    relative = absolute / np.maximum(np.abs(expected), 1e-6)
    metrics_path = Path(args.metrics)
    metrics = (
        json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics_path.is_file()
        else {}
    )
    metrics.update(
        {
            "onnx_checker_passed": True,
            "onnxruntime_checked": True,
            "onnxruntime_version": ort.__version__,
            "numpy_version": np.__version__,
            "max_abs_error": float(np.max(absolute)),
            "mean_abs_error": float(np.mean(absolute)),
            "max_relative_error_eps_1e-6": float(np.max(relative)),
            "mean_relative_error_eps_1e-6": float(np.mean(relative)),
        }
    )
    metrics.pop("onnxruntime_reason", None)
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
