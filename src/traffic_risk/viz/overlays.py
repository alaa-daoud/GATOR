"""Overlay rendering for tracked detections."""

from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd


def draw_tracked_detections(
    frame: object,
    detections: list[dict[str, object]],
) -> object:
    """Draw tracked bounding boxes and labels on a frame."""
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox_xyxy"]
        track_id = detection["track_id"]
        class_name = detection.get("class_name", "vehicle")

        pt1 = (int(x1), int(y1))
        pt2 = (int(x2), int(y2))
        cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"id={track_id} {class_name}",
            (int(x1), max(int(y1) - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return frame


def load_tracked_lookup(tracks_path: str | Path) -> dict[int, list[dict[str, object]]]:
    """Load tracked detections parquet/pickle keyed by frame index."""
    path = Path(tracks_path)
    if path.suffix == ".parquet":
        records = pd.read_parquet(path).to_dict("records")
    else:
        records = pd.read_pickle(path)

    lookup: dict[int, list[dict[str, object]]] = {}
    for row in records:
        lookup[int(row["frame_idx"])] = list(row.get("detections", []))
    return lookup


def render_annotated_outputs(
    *,
    video_path: str | Path,
    tracks_path: str | Path,
    outdir: str | Path,
    sample_every: int = 1,
    make_video: bool = True,
) -> list[Path]:
    """Render annotated frames and optional video from tracked detections."""
    tracked_lookup = load_tracked_lookup(tracks_path)
    out_dir = Path(outdir)
    frames_dir = out_dir / "annotated_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 10.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    writer = None
    video_out_path = out_dir / "annotated.mp4"
    if make_video and width > 0 and height > 0:
        writer = cv2.VideoWriter(
            str(video_out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

    frame_idx = 0
    generated: list[Path] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_idx % sample_every != 0:
                frame_idx += 1
                continue

            detections = tracked_lookup.get(frame_idx, [])
            annotated = draw_tracked_detections(frame, detections)

            frame_path = frames_dir / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(frame_path), annotated)
            generated.append(frame_path)

            if writer is not None:
                writer.write(annotated)

            frame_idx += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    if make_video and video_out_path.exists():
        generated.append(video_out_path)
    return generated
