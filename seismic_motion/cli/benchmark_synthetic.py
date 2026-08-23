"""Generate a deterministic Phase C sequence and its audit metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def _load_generator():
    root = Path(__file__).resolve().parents[2]
    benchmark_root = root / "benchmarks" / "synthetic"
    if str(benchmark_root) not in sys.path:
        sys.path.insert(0, str(benchmark_root))
    from generator import generate_sequence

    return generate_sequence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="translation_rotation_local")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--amplitude-px", type=float, default=0.5)
    parser.add_argument("--frequency-hz", type=float, default=2.0)
    parser.add_argument("--rotation-deg", type=float, default=0.1)
    parser.add_argument("--duration-s", type=float, default=4.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    generate_sequence = _load_generator()
    sequence = generate_sequence(
        args.case,
        fps=args.fps,
        duration_s=args.duration_s,
        translation_amplitude_px=args.amplitude_px,
        translation_frequency_hz=args.frequency_hz,
        rotation_amplitude_deg=args.rotation_deg,
    )
    sequence.save(args.output)
    print(
        json.dumps(
            {
                "case": sequence.case,
                "frames": int(sequence.frames_rgb.shape[0]),
                "points": int(sequence.tracks_xy_px.shape[1]),
                "fps": sequence.fps,
                "output": str(Path(args.output).resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

