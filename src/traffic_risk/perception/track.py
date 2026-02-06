"""Lightweight multi-object tracking utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

from traffic_risk.perception.types import Detection, DetectionFrame


@dataclass(frozen=True)
class TrackState:
    """Internal state for an active track."""

    track_id: int
    last_bbox: tuple[float, float, float, float]
    last_seen_frame: int
    age: int
    hit_count: int


@dataclass(frozen=True)
class TrackedDetection:
    """A detection associated with a track id."""

    track_id: int
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str


@dataclass(frozen=True)
class TrackedFrame:
    """Tracked detections for a frame."""

    frame_idx: int
    timestamp: float
    detections: list[TrackedDetection]


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
    if union <= 0:
        return 0.0
    return inter_area / union


def _greedy_match(
    tracks: list[TrackState],
    detections: list[Detection],
    iou_threshold: float,
) -> tuple[list[tuple[int, int]], set[int], set[int]]:
    """Greedy IoU matching between existing tracks and current detections."""
    candidate_pairs: list[tuple[float, int, int]] = []
    for track_idx, track in enumerate(tracks):
        for det_idx, detection in enumerate(detections):
            score = _iou(track.last_bbox, detection.bbox_xyxy)
            if score >= iou_threshold:
                candidate_pairs.append((score, track_idx, det_idx))

    candidate_pairs.sort(key=lambda item: item[0], reverse=True)

    matches: list[tuple[int, int]] = []
    used_tracks: set[int] = set()
    used_dets: set[int] = set()
    for _score, track_idx, det_idx in candidate_pairs:
        if track_idx in used_tracks or det_idx in used_dets:
            continue
        used_tracks.add(track_idx)
        used_dets.add(det_idx)
        matches.append((track_idx, det_idx))

    unmatched_tracks = set(range(len(tracks))) - used_tracks
    unmatched_dets = set(range(len(detections))) - used_dets
    return matches, unmatched_tracks, unmatched_dets


def track_detections(
    detection_frames_iter: Iterable[DetectionFrame],
    iou_threshold: float = 0.3,
    max_age: int = 10,
) -> Iterator[TrackedFrame]:
    """Track detections frame-to-frame with greedy IoU matching.

    Tracks persist for up to ``max_age`` missed frames to handle short occlusions.
    """
    active_tracks: dict[int, TrackState] = {}
    next_track_id = 0

    for frame in detection_frames_iter:
        track_list = list(active_tracks.values())
        matches, unmatched_tracks, unmatched_dets = _greedy_match(
            track_list, frame.detections, iou_threshold
        )

        updated_tracks: dict[int, TrackState] = {}
        tracked_detections: list[TrackedDetection] = []

        for track_idx, det_idx in matches:
            track = track_list[track_idx]
            detection = frame.detections[det_idx]
            new_state = TrackState(
                track_id=track.track_id,
                last_bbox=detection.bbox_xyxy,
                last_seen_frame=frame.frame_idx,
                age=0,
                hit_count=track.hit_count + 1,
            )
            updated_tracks[new_state.track_id] = new_state
            tracked_detections.append(
                TrackedDetection(
                    track_id=new_state.track_id,
                    bbox_xyxy=detection.bbox_xyxy,
                    confidence=detection.confidence,
                    class_id=detection.class_id,
                    class_name=detection.class_name,
                )
            )

        for track_idx in unmatched_tracks:
            track = track_list[track_idx]
            new_age = track.age + 1
            if new_age <= max_age:
                updated_tracks[track.track_id] = TrackState(
                    track_id=track.track_id,
                    last_bbox=track.last_bbox,
                    last_seen_frame=track.last_seen_frame,
                    age=new_age,
                    hit_count=track.hit_count,
                )

        for det_idx in unmatched_dets:
            detection = frame.detections[det_idx]
            state = TrackState(
                track_id=next_track_id,
                last_bbox=detection.bbox_xyxy,
                last_seen_frame=frame.frame_idx,
                age=0,
                hit_count=1,
            )
            updated_tracks[state.track_id] = state
            tracked_detections.append(
                TrackedDetection(
                    track_id=state.track_id,
                    bbox_xyxy=detection.bbox_xyxy,
                    confidence=detection.confidence,
                    class_id=detection.class_id,
                    class_name=detection.class_name,
                )
            )
            next_track_id += 1

        tracked_detections.sort(key=lambda det: det.track_id)
        active_tracks = updated_tracks
        yield TrackedFrame(
            frame_idx=frame.frame_idx,
            timestamp=frame.timestamp,
            detections=tracked_detections,
        )
