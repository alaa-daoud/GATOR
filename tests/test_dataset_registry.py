"""Tests for dataset registry."""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

from traffic_risk.io.dataset_registry import list_videos


def test_list_videos_missing_path(tmp_path: Path) -> None:
    """Raise a clear error when dataset path is missing."""
    config_path = tmp_path / "datasets.yaml"
    payload = {"datasets": {"missing": str(tmp_path / "does-not-exist")}}
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Dataset path not found"):
        list_videos("missing", config_path=config_path)
