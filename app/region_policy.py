from __future__ import annotations

import numpy as np


def page_preserve_regions(page: dict | None) -> list[dict]:
    if not isinstance(page, dict):
        return []
    value = page.get("preserve_regions")
    if isinstance(value, list):
        return value
    legacy = page.get("excluded_regions")
    return legacy if isinstance(legacy, list) else []


def geometry_center_in_regions(geometry: dict | None, regions) -> bool:
    if not isinstance(geometry, dict) or not regions:
        return False
    try:
        x1, y1, x2, y2 = (float(geometry[k]) for k in ("x1", "y1", "x2", "y2"))
    except (KeyError, TypeError, ValueError):
        return False
    if x2 <= x1 or y2 <= y1:
        return False
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    for region in regions:
        if not isinstance(region, dict):
            continue
        try:
            rx1, ry1, rx2, ry2 = (float(region[k]) for k in ("x1", "y1", "x2", "y2"))
        except (KeyError, TypeError, ValueError):
            continue
        if rx2 > rx1 and ry2 > ry1 and rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
            return True
    return False


def geometry_in_preserve_regions(page: dict, geometry: dict | None) -> bool:
    return geometry_center_in_regions(geometry, page_preserve_regions(page))


def text_object_in_preserve_region(page: dict, obj: dict) -> bool:
    return geometry_in_preserve_regions(page, obj.get("region") if isinstance(obj, dict) else None)


def subtract_regions_from_mask(mask: np.ndarray | None, regions) -> np.ndarray | None:
    if mask is None:
        return None
    result = np.ascontiguousarray(mask.copy())
    h, w = result.shape[:2]
    for region in regions or []:
        if not isinstance(region, dict):
            continue
        try:
            x1 = max(0, min(w, int(region["x1"])))
            y1 = max(0, min(h, int(region["y1"])))
            x2 = max(0, min(w, int(region["x2"])))
            y2 = max(0, min(h, int(region["y2"])))
        except (KeyError, TypeError, ValueError):
            continue
        if x2 > x1 and y2 > y1:
            result[y1:y2, x1:x2] = 0
    return result
