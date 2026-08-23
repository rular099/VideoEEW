"""Run the bounded realtime camera/file pipeline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

from seismic_motion.config import load_config
from seismic_motion.runtime.realtime import RealtimeRunner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--camera")
    source_group.add_argument("--playlist", help="YAML file containing a versioned sources list")
    parser.add_argument("--cotracker-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--duration-s", type=float)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.playlist:
        payload = yaml.safe_load(Path(args.playlist).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("sources"), list):
            raise ValueError("playlist YAML must contain a sources list")
        source = [os.path.expandvars(str(value)) for value in payload["sources"]]
        if not source:
            raise ValueError("playlist sources must not be empty")
    else:
        assert args.camera is not None
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
