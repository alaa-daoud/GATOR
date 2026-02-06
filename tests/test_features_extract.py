"""Tests for feature extraction and visibility helpers."""

from __future__ import annotations

import math

import pytest

pytest.importorskip("pandas")

from traffic_risk.features.extract import extract_features_from_tracked_frames
from traffic_risk.features.visibility import frame_luminance_bgr
from traffic_risk.perception.track import TrackedDetection, TrackedFrame


def test_frame_luminance_bgr_on_synthetic_frame() -> None:
    frame = [
        [[0, 0, 0], [255, 255, 255]],
        [[0, 0, 255], [255, 0, 0]],
    ]
    # luma values: 0, 255, 76.245, 29.07 -> average 90.07875
    value = frame_luminance_bgr(frame)
    assert math.isclose(value, 90.07875, rel_tol=1e-6)


def test_extract_features_speed_and_min_distance() -> None:
    tracked_frames = [
        TrackedFrame(
            frame_idx=0,
            timestamp=0.0,
            detections=[
                TrackedDetection(0, (0.0, 0.0, 2.0, 2.0), 0.9, 2, "car"),
                TrackedDetection(1, (4.0, 0.0, 6.0, 2.0), 0.8, 7, "truck"),
            ],
        ),
        TrackedFrame(
            frame_idx=1,
            timestamp=1.0,
            detections=[
                TrackedDetection(0, (1.0, 0.0, 3.0, 2.0), 0.9, 2, "car"),
                TrackedDetection(1, (5.0, 0.0, 7.0, 2.0), 0.8, 7, "truck"),
            ],
        ),
    ]
    visibility = {0: 100.0, 1: 120.0}

    table = extract_features_from_tracked_frames(
        tracked_frames,
        visibility,
        video_id="sample",
        fps=1.0,
        sample_every=1,
    )

    assert list(table["frame_idx"]) == [0, 1]
    assert list(table["vehicle_count"]) == [2, 2]
    assert table.loc[0, "mean_speed_px_s"] == 0.0
    assert math.isclose(table.loc[1, "mean_speed_px_s"], 1.0, rel_tol=1e-9)
    assert math.isclose(table.loc[0, "min_distance_px"], 4.0, rel_tol=1e-9)
    assert math.isclose(table.loc[1, "min_distance_px"], 4.0, rel_tol=1e-9)
    assert list(table["visibility_luma"]) == [100.0, 120.0]
