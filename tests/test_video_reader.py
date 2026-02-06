"""Tests for video reader utilities."""

from __future__ import annotations

from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from traffic_risk.io.video_reader import iter_frames


def _write_test_video(path: Path, frame_count: int = 5) -> None:
    frame_size = (64, 64)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, frame_size)
    for index in range(frame_count):
        frame = np.full((frame_size[1], frame_size[0], 3), index * 10, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_iter_frames_sampling_and_max(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.avi"
    _write_test_video(video_path, frame_count=6)

    frames = list(iter_frames(video_path, sample_every=2, max_frames=2))

    assert len(frames) == 2
    assert frames[0][0] == 0
    assert frames[1][0] == 2
