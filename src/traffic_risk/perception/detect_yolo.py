"""YOLOv8 detection helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    """Single detection result."""

    frame_id: int
    bbox_xyxy: tuple[float, float, float, float]
    score: float
    label: str


def load_model(weights_path: str) -> str:
    """Load YOLOv8 weights.

    TODO: Use ultralytics.YOLO to load weights.
    """
    return weights_path


def run_inference(model: str, frame: object) -> list[Detection]:
    """Run detection on a frame.

    TODO: Implement actual model inference.
    """
    _ = model
    _ = frame
    return []
