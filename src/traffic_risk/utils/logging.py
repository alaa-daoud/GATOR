"""Logging configuration."""

from __future__ import annotations

import logging


def configure_logging(level: str) -> None:
    """Configure basic logging."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
