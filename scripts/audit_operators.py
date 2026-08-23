#!/usr/bin/env python3
"""Write source and optional ONNX operator inventories."""

from __future__ import annotations

import argparse
from pathlib import Path

from seismic_motion.deployment.operator_audit import (
    load_onnx_operators,
    scan_python_operators,
    write_operator_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cotracker-root", required=True)
    parser.add_argument("--onnx", action="append", default=[])
    parser.add_argument("--output", default="reports/onnx_ops.csv")
    args = parser.parse_args()
    root = Path(args.cotracker_root)
    source_paths = [
        root / "cotracker/models/core/cotracker/cotracker3_online.py",
        root / "cotracker/models/core/model_utils.py",
        root / "cotracker/models/core/cotracker/blocks.py",
    ]
    rows = scan_python_operators(source_paths)
    for onnx_path in args.onnx:
        rows.extend(load_onnx_operators(onnx_path))
    write_operator_csv(args.output, rows)
    print(Path(args.output).resolve())


if __name__ == "__main__":
    main()

