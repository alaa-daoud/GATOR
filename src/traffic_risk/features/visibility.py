"""Visibility features computed from frame luminance."""

from __future__ import annotations


# ITU-R BT.601 luma coefficients for BGR inputs.
_B_COEFF = 0.114
_G_COEFF = 0.587
_R_COEFF = 0.299


def frame_luminance_bgr(frame: object) -> float:
    """Compute average luminance for a BGR frame.

    The function is intentionally generic and works with nested iterables or
    numpy-like arrays of shape (H, W, C>=3).
    """
    total = 0.0
    count = 0
    for row in frame:  # type: ignore[assignment]
        for pixel in row:
            b = float(pixel[0])
            g = float(pixel[1])
            r = float(pixel[2])
            total += _B_COEFF * b + _G_COEFF * g + _R_COEFF * r
            count += 1
    if count == 0:
        return 0.0
    return total / count
