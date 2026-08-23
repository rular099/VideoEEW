#!/usr/bin/env python3
"""Measure numerical equivalence of the Einsum-to-MatMul candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from seismic_motion.deployment.correlation import (
    correlation_batched_matmul,
    correlation_einsum,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    generator = torch.Generator(device=args.device).manual_seed(7)
    features = torch.randn(
        (1, 16, 32, 9, 9, 128), generator=generator, device=args.device
    )
    support = torch.randn(
        (1, 32, 7, 7, 128), generator=generator, device=args.device
    )
    with torch.inference_mode():
        expected = correlation_einsum(features, support)
        actual = correlation_batched_matmul(features, support)
    absolute = torch.abs(actual - expected)
    relative = absolute / torch.clamp_min(torch.abs(expected), 1e-6)
    timings: dict[str, dict[str, float]] = {}
    for name, function in (
        ("einsum", correlation_einsum),
        ("batched_matmul", correlation_batched_matmul),
    ):
        samples = []
        with torch.inference_mode():
            for _ in range(args.iterations):
                begin = time.perf_counter()
                function(features, support)
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize()
                samples.append((time.perf_counter() - begin) * 1000)
        timings[name] = {
            "mean_ms": float(np.mean(samples)),
            "p95_ms": float(np.percentile(samples, 95)),
        }
    payload = {
        "device": args.device,
        "dtype": str(features.dtype),
        "correlation_shape": list(features.shape),
        "support_shape": list(support.shape),
        "output_shape": list(actual.shape),
        "max_abs_error": float(torch.max(absolute)),
        "mean_abs_error": float(torch.mean(absolute)),
        "max_relative_error_eps_1e-6": float(torch.max(relative)),
        "mean_relative_error_eps_1e-6": float(torch.mean(relative)),
        "timing": timings,
        "track_output_error_px": None,
        "track_output_error_reason": "candidate is not substituted into the full tracker until ONNX/RKNN conversion is available",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
