"""Generate ALL, VIDEO-QUALITY-ONLY and POST-HOC PGA evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from seismic_motion.pga.evaluation import evaluate_subsets, read_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", default="configs/pga_eval_v2.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    result = evaluate_subsets(read_rows(args.dataset), config, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
