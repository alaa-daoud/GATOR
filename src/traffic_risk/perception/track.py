"""Tracking integration utilities."""

from __future__ import annotations

from dataclasses import dataclass

from traffic_risk.perception.types import Detection


@dataclass(frozen=True)
class Track:
    """Track state for a single object."""

    track_id: int
    detection: Detection
    age: int = 0


def _iou(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    """Compute IoU for two xyxy boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union == 0:
        return 0.0
    return inter_area / union


def update_tracks(
    detections: list[Detection],
    previous_tracks: list[Track] | None = None,
    iou_threshold: float = 0.3,
) -> list[Track]:
    """Update tracking state using greedy IoU association.

    TODO: Add Kalman filtering for motion prediction.
    """
    if previous_tracks is None:
        return [Track(track_id=index, detection=det) for index, det in enumerate(detections)]

    used_detections: set[int] = set()
    updated_tracks: list[Track] = []
    next_id = max((track.track_id for track in previous_tracks), default=-1) + 1

    for track in previous_tracks:
        best_iou = 0.0
        best_idx: int | None = None
        for idx, det in enumerate(detections):
            if idx in used_detections:
                continue
            iou_score = _iou(track.detection.bbox_xyxy, det.bbox_xyxy)
            if iou_score > best_iou:
                best_iou = iou_score
                best_idx = idx

        if best_idx is not None and best_iou >= iou_threshold:
            used_detections.add(best_idx)
            updated_tracks.append(
                Track(track_id=track.track_id, detection=detections[best_idx], age=0)
            )
        else:
            updated_tracks.append(
                Track(track_id=track.track_id, detection=track.detection, age=track.age + 1)
            )

    for idx, det in enumerate(detections):
        if idx not in used_detections:
            updated_tracks.append(Track(track_id=next_id, detection=det))
            next_id += 1

    return updated_tracks
