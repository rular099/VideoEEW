"""Strong-motion text records and video/sensor dataset discovery."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class StrongMotionRecord:
    timestamps_s: np.ndarray
    ew_gal: np.ndarray
    ns_gal: np.ndarray
    ud_gal: np.ndarray
    source_path: str

    def pga_gal(self, definition: str = "horizontal_vector") -> float:
        if definition == "horizontal_vector":
            values = np.hypot(self.ew_gal, self.ns_gal)
        elif definition == "max_horizontal_component":
            return float(max(np.max(np.abs(self.ew_gal)), np.max(np.abs(self.ns_gal))))
        elif definition == "three_component_vector":
            values = np.sqrt(self.ew_gal**2 + self.ns_gal**2 + self.ud_gal**2)
        else:
            raise ValueError(f"unknown PGA definition: {definition}")
        return float(np.max(values))


@dataclass(frozen=True)
class DatasetPair:
    record_id: str
    video_path: str
    strong_motion_paths: tuple[str, ...]
    pairing_status: str


def load_strong_motion_txt(path: str | Path) -> StrongMotionRecord:
    source = Path(path)
    rows: list[list[float]] = []
    with source.open("r", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader, None)
        if not header or len(header) < 4:
            raise ValueError(f"invalid strong-motion header: {source}")
        for row in reader:
            if not row or len(row) < 4:
                continue
            try:
                rows.append([float(value) for value in row[:4]])
            except ValueError:
                continue
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] != 4:
        raise ValueError(f"no usable strong-motion samples: {source}")
    timestamps = values[:, 0]
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"strong-motion timestamps are not increasing: {source}")
    return StrongMotionRecord(
        timestamps_s=timestamps,
        ew_gal=values[:, 1],
        ns_gal=values[:, 2],
        ud_gal=values[:, 3],
        source_path=str(source.resolve()),
    )


def discover_dataset_pairs(data_root: str | Path) -> list[DatasetPair]:
    root = Path(data_root)
    video_root = root / "视频"
    sensor_root = root / "强震仪"
    videos = {
        path.stem: path
        for path in video_root.iterdir()
        if path.is_file() and not path.name.startswith("._") and path.suffix.lower() in {".avi", ".mp4", ".mov", ".mkv"}
    }
    sensor_files = [
        path
        for path in sensor_root.glob("*.txt")
        if path.is_file() and not path.name.startswith("._")
    ]
    sensor_by_stem: dict[str, list[Path]] = {}
    for path in sensor_files:
        base = path.stem
        # Split records such as 54-1/54-2 are retained together under video 54.
        record_id = base.rsplit("-", 1)[0] if base.rsplit("-", 1)[-1].isdigit() else base
        sensor_by_stem.setdefault(record_id, []).append(path)
    all_ids = sorted(
        set(videos) | set(sensor_by_stem),
        key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value),
    )
    pairs: list[DatasetPair] = []
    for record_id in all_ids:
        video = videos.get(record_id)
        sensors = tuple(str(path.resolve()) for path in sorted(sensor_by_stem.get(record_id, [])))
        if video is not None and sensors:
            status = "paired" if len(sensors) == 1 else "paired_split_sensor"
        elif video is None:
            status = "missing_video"
        else:
            status = "missing_strong_motion"
        pairs.append(
            DatasetPair(
                record_id=record_id,
                video_path=str(video.resolve()) if video is not None else "",
                strong_motion_paths=sensors,
                pairing_status=status,
            )
        )
    return pairs
