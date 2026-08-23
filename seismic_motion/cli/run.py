"""Run the complete offline PC pipeline on one video."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from seismic_motion.config import load_config
from seismic_motion.runtime.pipeline import run_offline_video, write_run_artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--cotracker-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps", type=float)
    parser.add_argument("--output")
    args = parser.parse_args()
    config = load_config(args.config)
    run_id = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    output = Path(args.output or Path("runs") / run_id)
    result = run_offline_video(
        args.video,
        config,
        cotracker_root=args.cotracker_root,
        checkpoint=args.checkpoint,
        device=args.device,
        fps_override=args.fps,
    )
    write_run_artifacts(output, result, config)
    print(output.resolve())


if __name__ == "__main__":
    main()

