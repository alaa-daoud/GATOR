"""Geometry utilities for tracked-vehicle feature extraction."""

from __future__ import annotations

import math
from itertools import combinations


def bbox_centroid(bbox_xyxy: tuple[float, float, float, float]) -> tuple[float, float]:
    """Return the bbox centroid in pixel coordinates."""
    x1, y1, x2, y2 = bbox_xyxy
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def euclidean_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Compute Euclidean distance between 2D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def min_pairwise_distance(points: list[tuple[float, float]]) -> float:
    """Return the minimum pairwise point distance.

    Returns 0.0 when fewer than two points are provided.
    """
    if len(points) < 2:
        return 0.0
    return min(euclidean_distance(a, b) for a, b in combinations(points, 2))
