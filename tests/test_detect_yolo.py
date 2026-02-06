"""Tests for YOLOv8 detection wrapper."""

from __future__ import annotations

from typing import Iterator

import pytest

np = pytest.importorskip("numpy")

from traffic_risk.perception.detect_yolo import detect_frames


class _FakeBoxes:
    def __init__(self) -> None:
        self.xyxy = np.array([[0, 0, 10, 10], [5, 5, 15, 15]])
        self.conf = np.array([0.9, 0.8])
        self.cls = np.array([0, 1])


class _FakeResult:
    def __init__(self) -> None:
        self.boxes = _FakeBoxes()
        self.names = {0: "car", 1: "person"}


class _FakeModel:
    def __call__(self, frame, conf=0.25, iou=0.45, device=None, verbose=False):
        return [_FakeResult()]


def _frames() -> Iterator[tuple[int, float, np.ndarray]]:
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    yield 0, 0.0, frame


def test_detect_frames_filters_classes() -> None:
    frames_iter = _frames()
    setattr(frames_iter, "class_whitelist", ["car"])

    results = list(detect_frames(_FakeModel(), frames_iter))

    assert len(results) == 1
    assert len(results[0].detections) == 1
    assert results[0].detections[0].class_name == "car"
