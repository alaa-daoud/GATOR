"""Feature extraction pipeline."""

from __future__ import annotations

from traffic_risk.perception.track import Track


def extract_features(tracks: list[Track]) -> list[dict[str, float]]:
    """Extract features from tracks.

    TODO: Implement feature extraction (velocity, heading, geometry, visibility).
    """
    _ = tracks
    return []
