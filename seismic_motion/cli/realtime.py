"""Run the bounded realtime camera/file pipeline."""

from __future__ import annotations

import argparse
import json

from seismic_motion.config import load_config
from seismic_motion.runtime.realtime import RealtimeRunner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--camera", required=True)
    parser.add_argument("--cotracker-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--duration-s", type=float)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = int(args.camera) if args.camera.isdigit() else args.camera
    runner = RealtimeRunner(
        source,
        load_config(args.config),
        cotracker_root=args.cotracker_root,
        checkpoint=args.checkpoint,
        device=args.device,
        output_directory=args.output,
    )
    print(json.dumps(runner.run(args.duration_s), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

