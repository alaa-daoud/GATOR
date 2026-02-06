"""Feature extraction from tracked detections and raw video frames."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from traffic_risk.features.geometry import bbox_centroid, euclidean_distance, min_pairwise_distance
from traffic_risk.features.visibility import frame_luminance_bgr
from traffic_risk.io.video_reader import iter_frames
from traffic_risk.perception.detect_yolo import DEFAULT_CLASSES, detect_frames
from traffic_risk.perception.track import TrackedFrame, track_detections


@dataclass(frozen=True)
class FrameFeatureRow:
    """Tabular feature row for a single frame."""

    video_id: str
    frame_idx: int
    timestamp_s: float
    vehicle_count: int
    mean_speed_px_s: float
    min_distance_px: float
    visibility_luma: float
    fps: float
    sample_every: int


def extract_features_from_tracked_frames(
    tracked_frames: Iterable[TrackedFrame],
    visibility_by_frame_idx: dict[int, float],
    *,
    video_id: str,
    fps: float,
    sample_every: int,
) -> pd.DataFrame:
    """Compute per-frame features from tracked detections and visibility."""
    prev_centroid_by_track: dict[int, tuple[float, float]] = {}
    prev_timestamp_by_track: dict[int, float] = {}
    rows: list[FrameFeatureRow] = []

    for frame in tracked_frames:
        centroids: list[tuple[float, float]] = []
        speeds: list[float] = []

        for detection in frame.detections:
            centroid = bbox_centroid(detection.bbox_xyxy)
            centroids.append(centroid)

            if detection.track_id in prev_centroid_by_track:
                dt = frame.timestamp - prev_timestamp_by_track[detection.track_id]
                if dt > 0:
                    distance = euclidean_distance(centroid, prev_centroid_by_track[detection.track_id])
                    speeds.append(distance / dt)

            prev_centroid_by_track[detection.track_id] = centroid
            prev_timestamp_by_track[detection.track_id] = frame.timestamp

        rows.append(
            FrameFeatureRow(
                video_id=video_id,
                frame_idx=frame.frame_idx,
                timestamp_s=frame.timestamp,
                vehicle_count=len(frame.detections),
                mean_speed_px_s=sum(speeds) / len(speeds) if speeds else 0.0,
                min_distance_px=min_pairwise_distance(centroids),
                visibility_luma=visibility_by_frame_idx.get(frame.frame_idx, 0.0),
                fps=fps,
                sample_every=sample_every,
            )
        )

    frame_table = pd.DataFrame([row.__dict__ for row in rows])
    if frame_table.empty:
        frame_table = pd.DataFrame(
            columns=[
                "video_id",
                "frame_idx",
                "timestamp_s",
                "vehicle_count",
                "mean_speed_px_s",
                "min_distance_px",
                "visibility_luma",
                "fps",
                "sample_every",
            ]
        )
    return frame_table.sort_values(["frame_idx", "timestamp_s"], kind="stable").reset_index(drop=True)


def run_feature_extraction(
    video_path: str | Path,
    *,
    model: str = "yolov8n.pt",
    conf: float = 0.25,
    iou: float = 0.45,
    sample_every: int = 1,
    class_whitelist: list[str] | None = None,
) -> pd.DataFrame:
    """Run detect->track->feature extraction for a single video."""
    video_id = Path(video_path).stem
    sampled_frames = list(iter_frames(video_path, sample_every=sample_every))
    visibility = {
        frame_idx: frame_luminance_bgr(frame) for frame_idx, _timestamp, frame in sampled_frames
    }

    class _FrameSource:
        def __init__(self) -> None:
            self.video_path = str(video_path)
            self.sample_every = sample_every
            self.class_whitelist = class_whitelist or list(DEFAULT_CLASSES)
            self.cache_path = None

        def __iter__(self):
            return iter(sampled_frames)

    detection_frames = list(detect_frames(model, _FrameSource(), conf=conf, iou=iou))
    tracked_frames = list(track_detections(detection_frames))

    fps = 0.0
    if len(sampled_frames) >= 2:
        dt = sampled_frames[1][1] - sampled_frames[0][1]
        fps = (1.0 / dt) if dt > 0 else 0.0

    return extract_features_from_tracked_frames(
        tracked_frames,
        visibility,
        video_id=video_id,
        fps=fps,
        sample_every=sample_every,
    )


def export_features(
    feature_table: pd.DataFrame,
    *,
    outdir: str | Path,
    video_id: str,
    file_format: str,
) -> list[Path]:
    """Export feature table to CSV/Parquet with deterministic schema/order."""
    ordered_columns = [
        "video_id",
        "frame_idx",
        "timestamp_s",
        "vehicle_count",
        "mean_speed_px_s",
        "min_distance_px",
        "visibility_luma",
        "fps",
        "sample_every",
    ]
    stable_table = feature_table.loc[:, ordered_columns].sort_values(
        ["frame_idx", "timestamp_s"], kind="stable"
    )

    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    if file_format in {"csv", "both"}:
        csv_path = output_dir / f"{video_id}_features.csv"
        stable_table.to_csv(csv_path, index=False)
        outputs.append(csv_path)

    if file_format in {"parquet", "both"}:
        parquet_path = output_dir / f"{video_id}_features.parquet"
        stable_table.to_parquet(parquet_path, index=False)
        outputs.append(parquet_path)

    return outputs
