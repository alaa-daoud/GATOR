"""Dataset registry and metadata helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetEntry:
    """Metadata for a registered dataset."""

    name: str
    root: Path
    description: str


def register_dataset(name: str, root: str | Path, description: str) -> DatasetEntry:
    """Register a dataset entry.

    TODO: Persist registry entries and validate directory layout.
    """
    return DatasetEntry(name=name, root=Path(root), description=description)
