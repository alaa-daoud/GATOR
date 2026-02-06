"""Plot utilities for extracted traffic features."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_features_table(path: str | Path) -> pd.DataFrame:
    """Load features table from parquet or CSV."""
    feature_path = Path(path)
    if feature_path.suffix == ".parquet":
        table = pd.read_parquet(feature_path)
    else:
        table = pd.read_csv(feature_path)
    return table.sort_values(["frame_idx", "timestamp_s"], kind="stable").reset_index(drop=True)


def plot_feature_summaries(
    features_table: pd.DataFrame,
    *,
    outdir: str | Path,
) -> list[Path]:
    """Create summary plots for speed, visibility, and vehicle count."""
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []

    speed_path = output_dir / "speed_over_time.png"
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(features_table["timestamp_s"], features_table["mean_speed_px_s"], color="tab:blue")
    ax.set_title("Mean Speed over Time")
    ax.set_xlabel("timestamp_s")
    ax.set_ylabel("mean_speed_px_s")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(speed_path, dpi=150)
    plt.close(fig)
    paths.append(speed_path)

    visibility_path = output_dir / "visibility_histogram.png"
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(features_table["visibility_luma"], bins=20, color="tab:orange", edgecolor="black")
    ax.set_title("Visibility Luminance Histogram")
    ax.set_xlabel("visibility_luma")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(visibility_path, dpi=150)
    plt.close(fig)
    paths.append(visibility_path)

    count_path = output_dir / "vehicle_count_over_time.png"
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(features_table["timestamp_s"], features_table["vehicle_count"], color="tab:green")
    ax.set_title("Vehicle Count over Time")
    ax.set_xlabel("timestamp_s")
    ax.set_ylabel("vehicle_count")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(count_path, dpi=150)
    plt.close(fig)
    paths.append(count_path)

    return paths
