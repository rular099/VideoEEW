#!/usr/bin/env python3
"""Analyze cached run directories for tracker reseed discontinuities."""

from __future__ import annotations

import argparse
from pathlib import Path

from seismic_motion.tracking.reseed_analysis import analyze_reseed_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = analyze_reseed_run(Path(args.run), Path(args.output))
    print(f"analyzed reseed boundaries: {len(rows)}")


if __name__ == "__main__":
    main()
