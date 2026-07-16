"""YOLOv8 detection helpers."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd
from ultralytics import YOLO

from traffic_risk.perception.types import Detection, DetectionFrame

DEFAULT_CLASSES = ("car", "truck", "bus", "motorcycle")


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _resolve_model(model: str | YOLO, device: str | None = None) -> YOLO:
    if isinstance(model, str):
        return YOLO(model)
    if device:
        model.to(device)
    return model


def _video_hash(video_path: Path) -> str:
    data = f"{video_path.resolve()}::{video_path.stat().st_mtime}".encode("utf-8")
    return sha256(data).hexdigest()


def _cache_key(
    *,
    video_hash: str,
    model_name: str,
    conf: float,
    iou: float,
    classes: tuple[str, ...],
    sample_every: int,
) -> str:
    payload = {
        "video_hash": video_hash,
        "model_name": model_name,
        "conf": conf,
        "iou": iou,
        "classes": classes,
        "sample_every": sample_every,
    }
    raw = repr(payload).encode("utf-8")
    return sha256(raw).hexdigest()


def _load_cache(cache_path: Path) -> dict[str, Any] | None:
    if not cache_path.exists():
        return None
    if cache_path.suffix == ".parquet":
        meta_path = cache_path.with_suffix(".meta.json")
        if not meta_path.exists():
            return None
        meta = pd.read_json(meta_path, typ="series").to_dict()
        df = pd.read_parquet(cache_path)
        return {"key": meta.get("key"), "rows": df.to_dict("records")}

    return pd.read_pickle(cache_path)


def _save_cache(cache_path: Path, key: str, rows: list[dict[str, Any]]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.suffix == ".parquet":
        df = pd.DataFrame(rows)
        df.to_parquet(cache_path, index=False)
        meta_path = cache_path.with_suffix(".meta.json")
        pd.Series({"key": key}).to_json(meta_path)
        return
    pd.to_pickle({"key": key, "rows": rows}, cache_path)


def detect_frames(
    model: str | YOLO,
    frames_iter: Iterable[tuple[int, float, np.ndarray]],
    conf: float = 0.25,
    iou: float = 0.45,
    device: str | None = None,
) -> Iterator[DetectionFrame]:
    """Run YOLOv8 on frames and yield detections per frame."""
    model_instance = _resolve_model(model, device=device)
    model_name = model if isinstance(model, str) else getattr(
        model_instance, "ckpt_path", model_instance.__class__.__name__
    )

    class_whitelist = getattr(frames_iter, "class_whitelist", DEFAULT_CLASSES)
    classes = tuple(class_whitelist)

    video_path = getattr(frames_iter, "video_path", None)
    cache_path = getattr(frames_iter, "cache_path", None)
    sample_every = getattr(frames_iter, "sample_every", 1)

    if cache_path and video_path:
        cache_path = Path(cache_path)
        video_hash = _video_hash(Path(video_path))
        key = _cache_key(
            video_hash=video_hash,
            model_name=str(model_name),
            conf=conf,
            iou=iou,
            classes=classes,
            sample_every=sample_every,
        )
        cached = _load_cache(cache_path)
        if cached and cached.get("key") == key:
            for row in cached.get("rows", []):
                detections = [
                    Detection(
                        bbox_xyxy=tuple(det["bbox_xyxy"]),
                        confidence=det["confidence"],
                        class_id=det["class_id"],
                        class_name=det["class_name"],
                    )
                    for det in row["detections"]
                ]
                yield DetectionFrame(
                    frame_idx=row["frame_idx"],
                    timestamp=row["timestamp"],
                    detections=detections,
                )
            return

    rows: list[dict[str, Any]] = []
    for frame_idx, timestamp, frame in frames_iter:
        results = model_instance(
            frame, conf=conf, iou=iou, device=device, verbose=False
        )
        result = results[0]
        boxes = result.boxes
        xyxy = _to_numpy(boxes.xyxy)
        confs = _to_numpy(boxes.conf)
        class_ids = _to_numpy(boxes.cls).astype(int)

        detections: list[Detection] = []
        for bbox, score, class_id in zip(xyxy, confs, class_ids, strict=False):
            class_name = result.names.get(int(class_id), str(class_id))
            if class_name not in classes:
                continue
            detections.append(
                Detection(
                    bbox_xyxy=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                    confidence=float(score),
                    class_id=int(class_id),
                    class_name=class_name,
                )
            )

        detection_frame = DetectionFrame(
            frame_idx=frame_idx, timestamp=timestamp, detections=detections
        )
        rows.append(
            {
                "frame_idx": detection_frame.frame_idx,
                "timestamp": detection_frame.timestamp,
                "detections": [asdict(det) for det in detection_frame.detections],
            }
        )
        yield detection_frame

    if cache_path and video_path:
        _save_cache(cache_path, key, rows)
