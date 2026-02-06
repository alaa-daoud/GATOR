"""Tests for lightweight IoU tracker."""

from __future__ import annotations

from traffic_risk.perception.track import track_detections
from traffic_risk.perception.types import Detection, DetectionFrame


def _det(x1: float, y1: float, x2: float, y2: float, label: str = "car") -> Detection:
    return Detection(
        bbox_xyxy=(x1, y1, x2, y2),
        confidence=0.9,
        class_id=2,
        class_name=label,
    )


def test_track_ids_remain_consistent_across_frames() -> None:
    frames = [
        DetectionFrame(frame_idx=0, timestamp=0.0, detections=[_det(0, 0, 10, 10), _det(100, 100, 110, 110)]),
        DetectionFrame(frame_idx=1, timestamp=0.1, detections=[_det(1, 1, 11, 11), _det(101, 101, 111, 111)]),
        DetectionFrame(frame_idx=2, timestamp=0.2, detections=[_det(2, 2, 12, 12), _det(102, 102, 112, 112)]),
    ]

    tracked = list(track_detections(frames, iou_threshold=0.3, max_age=2))

    ids_per_frame = [[det.track_id for det in frame.detections] for frame in tracked]
    assert ids_per_frame[0] == [0, 1]
    assert ids_per_frame[1] == [0, 1]
    assert ids_per_frame[2] == [0, 1]


def test_track_persists_brief_occlusion_then_recovers_id() -> None:
    frames = [
        DetectionFrame(frame_idx=0, timestamp=0.0, detections=[_det(0, 0, 10, 10)]),
        DetectionFrame(frame_idx=1, timestamp=0.1, detections=[]),
        DetectionFrame(frame_idx=2, timestamp=0.2, detections=[_det(1, 1, 11, 11)]),
    ]

    tracked = list(track_detections(frames, iou_threshold=0.3, max_age=2))

    assert [det.track_id for det in tracked[0].detections] == [0]
    assert tracked[1].detections == []
    assert [det.track_id for det in tracked[2].detections] == [0]
