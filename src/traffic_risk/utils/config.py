"""Configuration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict


class PipelineSettings(BaseSettings):
    """Settings derived from environment variables."""

    data_dir: Path = Path("./data")
    config_path: Path = Path("./configs/default.yaml")
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_prefix="TRAFFIC_RISK_",
        env_file=".env",
        extra="ignore",
    )


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file.

    TODO: Enforce deterministic settings and schema validation.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}
