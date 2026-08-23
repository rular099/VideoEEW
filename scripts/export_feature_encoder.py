#!/usr/bin/env python3
"""Export only the first-priority fixed-shape feature encoder to ONNX."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

from seismic_motion.deployment.feature_encoder import FeatureEncoderExportWrapper


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cotracker-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    root = Path(args.cotracker_root).resolve()
    sys.path.insert(0, str(root))
    from cotracker.models.build_cotracker import build_cotracker

    model = build_cotracker(
        checkpoint=str(Path(args.checkpoint).resolve()), offline=False, window_len=16
    ).to(args.device).eval()
    wrapper = FeatureEncoderExportWrapper(model.fnet).to(args.device).eval()
    generator = torch.Generator(device=args.device).manual_seed(0)
    example = torch.rand((1, 3, 384, 512), generator=generator, device=args.device) * 255
    with torch.inference_mode():
        eager = wrapper(example)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        example,
        str(output),
        input_names=["rgb_0_255"],
        output_names=["normalized_features"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes=None,
        dynamo=False,
    )
    metrics: dict[str, object] = {
        "onnx_path": str(output.resolve()),
        "input_shape": list(example.shape),
        "output_shape": list(eager.shape),
        "opset": args.opset,
        "fixed_shape": True,
        "onnxruntime_checked": False,
    }
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
        actual = session.run(None, {"rgb_0_255": example.cpu().numpy()})[0]
        expected = eager.cpu().numpy()
        metrics.update(
            {
                "onnxruntime_checked": True,
                "max_abs_error": float(np.max(np.abs(actual - expected))),
                "mean_abs_error": float(np.mean(np.abs(actual - expected))),
            }
        )
    except ImportError:
        metrics["onnxruntime_reason"] = "onnxruntime is not installed"
    metrics_path = output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

