#!/usr/bin/env python3
"""Materialize reviewable common/local/signal artifacts from synthetic truth."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from seismic_motion.motion.quality import QualityThresholds
from seismic_motion.motion.residual_motion import decompose_tracks
from seismic_motion.signal.features import extract_motion_features
from seismic_motion.signal.filtering import bandpass_filter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    data = np.load(args.input)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    timestamps = data["timestamps"]
    tracks = data["tracks_xy_px"]
    visible = data["visibility"]
    reference = data["reference_xy_px"]
    decomposition = decompose_tracks(
        reference,
        tracks,
        visible,
        model="similarity",
        ransac_threshold_px=0.15,
        frame_size=data["frames_rgb"].shape[1:3],
        quality_thresholds=QualityThresholds(
            min_spatial_coverage=0.05, max_fit_rmse_px=0.15
        ),
    )
    np.savez_compressed(
        output / "tracks.npz",
        timestamps=timestamps,
        tracks_xy_px=tracks,
        visibility=visible,
        reference_xy_px=reference,
    )
    np.savez_compressed(
        output / "residual_motion.npz",
        timestamps=timestamps,
        residual_xy_px=decomposition.residual_xy_px,
        inlier_mask=decomposition.inlier_mask,
    )
    common_header = [
        "timestamp",
        "tx_px",
        "ty_px",
        "rotation_2d_rad",
        "scale",
        "shear",
        "inlier_ratio",
        "fit_rmse_px",
        "spatial_coverage",
        "num_valid_tracks",
        "num_inliers",
        "condition_number",
        "motion_quality",
    ]
    with (output / "common_motion.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(common_header)
        for index, estimate in enumerate(decomposition.estimates):
            if estimate is None:
                writer.writerow([timestamps[index], *("" for _ in range(11)), "INVALID"])
            else:
                writer.writerow(
                    [
                        timestamps[index],
                        estimate.tx_px,
                        estimate.ty_px,
                        estimate.rotation_2d_rad,
                        estimate.scale,
                        estimate.shear,
                        estimate.inlier_ratio,
                        estimate.fit_rmse_px,
                        estimate.spatial_coverage,
                        estimate.num_valid_tracks,
                        estimate.num_inliers,
                        estimate.condition_number,
                        decomposition.quality[index],
                    ]
                )
    with (output / "motion_quality.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "quality", "reasons"])
        for timestamp, quality, reasons in zip(
            timestamps, decomposition.quality, decomposition.quality_reasons
        ):
            writer.writerow([timestamp, quality, "|".join(reasons)])
    common_xy = decomposition.common_parameters[:, :2]
    filtered = bandpass_filter(common_xy, 30.0, (0.3, 8.0), causal=False)
    np.savez_compressed(
        output / "filtered_signals.npz",
        timestamps=timestamps,
        common_xy_px=filtered.values,
        filter_mode=np.asarray(filtered.mode),
    )
    inlier_ratio = decomposition.common_parameters[:, 5]
    fit_rmse = decomposition.common_parameters[:, 6]
    features = extract_motion_features(
        timestamps,
        filtered.values,
        decomposition.residual_xy_px,
        visible,
        inlier_ratio=inlier_ratio,
        fit_rmse_px=fit_rmse,
        quality_flags=np.unique(decomposition.quality),
        causal=False,
    )
    with (output / "features.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(features))
        writer.writeheader()
        writer.writerow(features)


if __name__ == "__main__":
    main()

