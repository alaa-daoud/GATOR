"""Command-line interface for the traffic risk pipeline."""

from __future__ import annotations

import argparse

from traffic_risk.utils.config import PipelineSettings, load_config
from traffic_risk.utils.logging import configure_logging


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Traffic risk visual feature extraction pipeline"
    )
    parser.add_argument(
        "--config",
        default="./configs/default.yaml",
        help="Path to a YAML configuration file.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (e.g., INFO, DEBUG).",
    )
    return parser


def main() -> None:
    """Run the (placeholder) pipeline."""
    parser = build_parser()
    args = parser.parse_args()

    settings = PipelineSettings()
    configure_logging(args.log_level or settings.log_level)
    config = load_config(args.config or settings.config_path)

    # TODO: wire up video ingestion, detection, tracking, and feature extraction.
    print("Loaded config:", config)


if __name__ == "__main__":
    main()
