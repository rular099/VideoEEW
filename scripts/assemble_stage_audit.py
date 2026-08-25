#!/usr/bin/env python3
"""Assemble heterogeneous next-stage runs into one auditable stage directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess

import yaml

from seismic_motion.diagnostics.provenance import environment_snapshot, git_state


ARTIFACTS = {
    "pga": (
        "pga_eval_all.csv",
        "pga_eval_video_quality.csv",
        "pga_eval_posthoc_aligned.csv",
        "pga_metrics.json",
        "pga_bootstrap_ci.json",
        "pga_model_research.json",
    ),
    "runtime": (
        "runtime_timing.csv",
        "runtime_queue.csv",
        "runtime_memory.csv",
        "runtime_events.csv",
    ),
    "alignment": (
        "alignment_candidates.csv",
        "null_max_corr_distribution.csv",
        "alignment_significance.csv",
    ),
    "stress": (
        "tracking_stress_summary.csv",
        "common_motion_metrics.csv",
        "local_motion_metrics.csv",
        "rotation_metrics.csv",
        "stress_manifest.json",
        "stress_failures.csv",
    ),
    "reseed": ("reseed_boundary.csv", "reseed_summary.json"),
}


def _json(path: Path | None) -> dict[str, object]:
    if path is None or not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _copy_group(source: Path | None, names: tuple[str, ...], output: Path) -> None:
    for name in names:
        destination = output / name
        if source is not None and (source / name).is_file():
            shutil.copy2(source / name, destination)
        else:
            destination.write_text("NOT_MEASURED\n", encoding="utf-8")


def assemble(args: argparse.Namespace) -> Path:
    repository = Path(__file__).resolve().parents[1]
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    sources = {
        "pga": Path(args.pga_dir).resolve() if args.pga_dir else None,
        "runtime": Path(args.runtime_dir).resolve() if args.runtime_dir else None,
        "alignment": Path(args.alignment_dir).resolve() if args.alignment_dir else None,
        "stress": Path(args.stress_dir).resolve() if args.stress_dir else None,
        "reseed": Path(args.reseed_dir).resolve() if args.reseed_dir else None,
    }
    for group, names in ARTIFACTS.items():
        _copy_group(sources[group], names, output)
    plots = output / "plots"
    plots.mkdir(exist_ok=True)
    if args.plots_dir:
        for path in Path(args.plots_dir).glob("*"):
            if path.is_file() and path.suffix.lower() in {".png", ".json"}:
                shutil.copy2(path, plots / path.name)

    state = git_state(repository)
    commit = str(state.get("commit", "unknown"))
    status_lines = [str(value) for value in state.get("status", [])]
    (output / "git_commit.txt").write_text(commit + "\n", encoding="utf-8")
    (output / "git_status_porcelain.txt").write_text(
        "\n".join(status_lines) + "\n", encoding="utf-8"
    )
    for arguments, name in (
        (["git", "diff", "--binary", "HEAD"], "git_diff.patch"),
        (["git", "diff", "--binary", "--cached"], "git_diff_cached.patch"),
    ):
        diff = subprocess.run(
            arguments, cwd=repository, check=True, text=True, capture_output=True
        ).stdout
        (output / name).write_text(diff, encoding="utf-8")

    effective_config = Path(args.effective_config)
    shutil.copy2(effective_config, output / "effective_config.yaml")
    shutil.copy2(effective_config, output / "config.yaml")
    configuration = yaml.safe_load(effective_config.read_text(encoding="utf-8"))
    (output / "signal_config.yaml").write_text(
        yaml.safe_dump(configuration.get("signal", {}), sort_keys=False), encoding="utf-8"
    )
    (output / "runtime_config.yaml").write_text(
        yaml.safe_dump(configuration.get("runtime", {}), sort_keys=False), encoding="utf-8"
    )
    shutil.copy2(args.model_config, output / "model_config.yaml")
    shutil.copy2(args.deployment_config, output / "deployment_config.yaml")
    shutil.copy2(repository / "reports/IMPLEMENTATION_STATUS.md", output / "IMPLEMENTATION_STATUS.md")
    (output / "RK3588_STATUS.md").write_text(
        "# RK3588 status\n\nBLOCKED\n\nNo accessible RK3588 device was available. "
        "No board latency, memory, thermal, RKNN conversion, or realtime claim is made.\n",
        encoding="utf-8",
    )

    pga_metrics = _json(sources["pga"] / "pga_metrics.json" if sources["pga"] else None)
    runtime_metrics = _json(sources["runtime"] / "metrics.json" if sources["runtime"] else None)
    stress_manifest = _json(
        sources["stress"] / "stress_manifest.json" if sources["stress"] else None
    )
    reseed_summary = _json(
        sources["reseed"] / "reseed_summary.json" if sources["reseed"] else None
    )
    combined = {
        "pc_30_fps_realtime": (
            runtime_metrics.get("realtime_acceptance", "NOT_TESTED")
            if abs(float(runtime_metrics.get("target_fps", 0.0)) - 30.0) < 0.1
            else "NOT_TESTED"
        ),
        "pc_50_fps_realtime": (
            runtime_metrics.get("realtime_acceptance", "NOT_TESTED")
            if abs(float(runtime_metrics.get("target_fps", 0.0)) - 50.0) < 0.1
            else "NOT_TESTED"
        ),
        "rk3588_realtime": "BLOCKED",
        "causal_pga_status": pga_metrics.get("causal_pga_status", "NOT_TESTED"),
        "scientific_validity": pga_metrics.get("scientific_validity", "RESEARCH_ONLY"),
        "geometric_scale": pga_metrics.get("geometric_scale", "UNCALIBRATED"),
        "pga": pga_metrics,
        "runtime": runtime_metrics,
        "stress": stress_manifest,
        "reseed": reseed_summary,
    }
    (output / "metrics.json").write_text(
        json.dumps(combined, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    environment = environment_snapshot()
    (output / "environment.txt").write_text(
        json.dumps(environment, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "device_info.txt").write_text(
        json.dumps(
            {
                "development_host": environment["platform"],
                "runtime_evidence": str(sources["runtime"] or "NOT_MEASURED"),
                "rk3588": "BLOCKED_NO_DEVICE",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    source_payload = {name: str(path) if path else "NOT_MEASURED" for name, path in sources.items()}
    (output / "input_manifest.json").write_text(
        json.dumps(source_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "run_id": output.name,
        "git_commit": commit,
        "git_dirty": bool(state.get("dirty", True)),
        "device": runtime_metrics.get("device", "server-242-or-NOT_MEASURED"),
        "input_id": source_payload,
        "signal_parameters": configuration.get("signal", {}),
        "tracker_parameters": configuration.get("tracker", {}),
        "motion_parameters": configuration.get("motion", {}),
        "scale_parameters": configuration.get("scale", {}),
        "pga_model_version": "videoeew-pga-v2-research",
        "change_summary": "strict causal online signal, unbiased subsets, realtime audit, reseed and stress",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pga-dir")
    parser.add_argument("--runtime-dir")
    parser.add_argument("--alignment-dir")
    parser.add_argument("--stress-dir")
    parser.add_argument("--reseed-dir")
    parser.add_argument("--plots-dir")
    parser.add_argument("--effective-config", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--deployment-config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(assemble(args))


if __name__ == "__main__":
    main()
