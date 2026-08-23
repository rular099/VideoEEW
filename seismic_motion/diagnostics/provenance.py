"""Small, dependency-light provenance helpers used by every run."""

from __future__ import annotations

import hashlib
import importlib.metadata
from functools import lru_cache
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


@lru_cache(maxsize=32)
def _sha256_file_cached(
    resolved_path: str, size: int, modification_time_ns: int, chunk_size: int
) -> str:
    _ = (size, modification_time_ns)  # Included in the cache key.
    digest = hashlib.sha256()
    with Path(resolved_path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    source = Path(path).resolve()
    stat = source.stat()
    return _sha256_file_cached(str(source), stat.st_size, stat.st_mtime_ns, chunk_size)


def git_state(repository: str | Path) -> dict[str, Any]:
    root = Path(repository)

    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=root, check=True, text=True, capture_output=True
        )
        return result.stdout.strip()

    status = run("status", "--porcelain=v1")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(status),
        "status": status.splitlines(),
    }


def software_versions(distributions: Iterable[str] | None = None) -> dict[str, str]:
    names = distributions or (
        "numpy",
        "scipy",
        "PyYAML",
        "opencv-python",
        "opencv-python-headless",
        "pandas",
        "psutil",
        "scikit-learn",
        "torch",
        "torchvision",
        "onnx",
        "onnxruntime",
    )
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    return versions


def environment_snapshot() -> dict[str, Any]:
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "software_versions": software_versions(),
    }


def write_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
