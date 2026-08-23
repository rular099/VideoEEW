"""Compare extracted visual acceleration with a strong-motion text record."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from seismic_motion.pga.alignment import estimate_time_offset
from seismic_motion.pga.records import load_strong_motion_txt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual-signals", required=True)
    parser.add_argument("--strong-motion", required=True)
    parser.add_argument("--max-offset-s", type=float, default=2.0)
    parser.add_argument("--min-correlation", type=float, default=0.25)
    parser.add_argument("--output")
    args = parser.parse_args()
    visual = np.load(args.visual_signals)
    record = load_strong_motion_txt(args.strong_motion)
    result = estimate_time_offset(
        visual["timestamps"],
        visual["acceleration_proxy_px_s2"],
        record.timestamps_s,
        np.stack([record.ew_gal, record.ns_gal], axis=1),
        max_offset_s=args.max_offset_s,
        min_correlation=args.min_correlation,
    )
    payload = {
        **asdict(result),
        "offset_definition": "compare visual(t) with sensor(t + offset_s)",
        "sensor_channels": ["EW-gal", "NS-gal"],
        "visual_units": "px/s^2 proxy",
        "sensor_units": "gal",
        "pga_horizontal_vector_gal": record.pga_gal("horizontal_vector"),
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()

