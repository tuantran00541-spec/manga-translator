from __future__ import annotations

import math

BOX_RULE = ('Each box names its corners, normalised to 0-1000 on the image: "x1","y1" top-left and "x2","y2" '
            "bottom-right, x across the width and y down the height. ")
BOX_KEYS = ("x1", "y1", "x2", "y2")


def scaled_box(raw, width: int, height: int) -> tuple[float, float, float, float] | None:
    if not isinstance(raw, dict):
        return None
    try:
        corners = [float(raw[key]) for key in BOX_KEYS]
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in corners):
        return None
    x1, y1, x2, y2 = (max(0.0, min(1000.0, v)) for v in corners)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1 * width / 1000, y1 * height / 1000, x2 * width / 1000, y2 * height / 1000
