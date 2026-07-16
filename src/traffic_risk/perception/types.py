"""Shared perception types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    """Single detection result."""

    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str


@dataclass(frozen=True)
class DetectionFrame:
    """Detections for a single frame."""

    frame_idx: int
    timestamp: float
    detections: list[Detection]
