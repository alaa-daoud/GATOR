"""Dataset registry and metadata helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DatasetEntry:
    """Metadata for a registered dataset."""

    name: str
    root: Path
    description: str | None = None


def register_dataset(name: str, root: str | Path, description: str | None = None) -> DatasetEntry:
    """Register a dataset entry.

    TODO: Persist registry entries and validate directory layout.
    """
    return DatasetEntry(name=name, root=Path(root), description=description)


def load_datasets_config(path: str | Path) -> dict[str, Any]:
    """Load dataset configuration YAML."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def list_videos(
    dataset_name: str,
    config_path: str | Path = "./configs/datasets.yaml",
    extensions: tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv"),
) -> list[Path]:
    """Return video file paths for a dataset name."""
    config = load_datasets_config(config_path)
    datasets = config.get("datasets", {})
    if dataset_name not in datasets:
        available = ", ".join(sorted(datasets.keys()))
        raise KeyError(f"Dataset '{dataset_name}' not found. Available: {available}")

    dataset_root = Path(datasets[dataset_name]).expanduser()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_root}")

    videos = [
        path
        for path in dataset_root.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    ]
    return sorted(videos)
