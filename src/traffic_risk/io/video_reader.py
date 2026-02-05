"""Video ingestion utilities."""

from __future__ import annotations

from pathlib import Path


def open_video(path: str | Path) -> Path:
    """Validate and return a video path.

    TODO: Replace with OpenCV video reader wrapper that yields frames.
    """
    video_path = Path(path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    return video_path
