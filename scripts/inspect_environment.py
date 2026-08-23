#!/usr/bin/env python3
"""Print or save the exact software environment used for a run."""

from __future__ import annotations

import argparse
import json

from seismic_motion.diagnostics.provenance import environment_snapshot, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()
    snapshot = environment_snapshot()
    if args.output:
        write_json(args.output, snapshot)
    print(json.dumps(snapshot, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

