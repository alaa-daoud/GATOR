"""Video ingestion utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


def open_video(path: str | Path) -> cv2.VideoCapture:
    """Open a video path with OpenCV."""
    video_path = Path(path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    return capture


def iter_frames(
    path: str | Path,
    *,
    sample_every: int = 1,
    max_frames: int | None = None,
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield frames from a video as (frame_index, timestamp_sec, frame)."""
    if sample_every < 1:
        raise ValueError("sample_every must be >= 1")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be >= 1")

    capture = open_video(path)
    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    frame_index = 0
    yielded = 0

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            if frame_index % sample_every == 0:
                timestamp = frame_index / fps if fps > 0 else 0.0
                yield frame_index, timestamp, frame
                yielded += 1
                if max_frames is not None and yielded >= max_frames:
                    break
            frame_index += 1
    finally:
        capture.release()
