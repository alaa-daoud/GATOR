"""Tracking integration utilities."""

from __future__ import annotations

from dataclasses import dataclass

from traffic_risk.perception.detect_yolo import Detection


@dataclass(frozen=True)
class Track:
    """Track state for a single object."""

    track_id: int
    detection: Detection


def update_tracks(detections: list[Detection]) -> list[Track]:
    """Update tracking state.

    TODO: Implement tracker integration (e.g., ByteTrack/DeepSORT).
    """
    return [Track(track_id=index, detection=det) for index, det in enumerate(detections)]
