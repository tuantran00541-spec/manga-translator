from __future__ import annotations

import cv2
import numpy as np

from app.mask_store import decode_mask_value
from app.render.balloon_layout import box_bounds
from app.text_objects import source_box_ids


def user_position(record: dict) -> bool:
    saved = box_bounds(record.get("auto_geometry"))
    return bool(record.get("manual") or record.get("geometry_overridden") or record.get("origin") == "manual"
                or (saved and not record.get("seam_owner") and box_bounds(record.get("region")) != saved))


def detector_position(obj: dict, source_boxes: dict) -> bool:
    refs = source_box_ids(obj)
    manual_source = bool(refs) and all(ref in source_boxes and user_position(source_boxes[ref]) for ref in refs)
    return bool((refs or obj.get("auto_generated") or obj.get("origin") == "detector" or obj.get("seam_owner"))
                and not (user_position(obj) or manual_source))


def _pixel_bounds(mask, origin):
    rows, cols = np.nonzero(mask)
    if rows.size < 4:
        return None
    x, y = origin
    return (x + int(cols.min()), y + int(rows.min()), x + int(cols.max()) + 1, y + int(rows.max()) + 1)


def _visible_ink(original, anchor):
    x1, y1, x2, y2 = anchor
    left, top = max(0, x1 - 2), max(0, y1 - 2)
    right, bottom = min(original.width, x2 + 2), min(original.height, y2 + 2)
    if left >= right or top >= bottom:
        return False
    crop = np.asarray(original.crop((left, top, right, bottom)).convert("RGB"))
    if crop.size == 0:
        return False
    return bool(np.ptp(crop.reshape(-1, 3), axis=0).max() >= 12)


def source_ink_bounds(record, original, clean):
    """Locate source ink using a glyph mask or bounded strokes that disappeared during cleaning."""
    bounds = box_bounds(record)
    if bounds is None:
        return None
    x1, y1, x2, y2 = bounds
    left, top = max(0, x1), max(0, y1)
    right, bottom = min(clean.width, x2), min(clean.height, y2)
    if left >= right or top >= bottom:
        return None
    visible = (left, top, right, bottom)

    seed = decode_mask_value(record.get("letters_mask"))
    if seed is not None and seed.shape == (y2 - y1, x2 - x1):
        visible_seed = seed[top - y1:bottom - y1, left - x1:right - x1]
        anchor = _pixel_bounds(visible_seed, (left, top))
        if anchor is not None:
            return anchor if original is None or _visible_ink(original, anchor) else None
    if original is None or min(right - left, bottom - top) < 8:
        return None

    raw = np.asarray(original.crop(visible).convert("RGB"))
    delta = cv2.absdiff(raw, np.asarray(clean.crop(visible).convert("RGB"))).max(axis=2)
    changed = delta >= 20
    if np.count_nonzero(changed) > .55 * changed.size:
        return None
    _, labels, pieces, _ = cv2.connectedComponentsWithStats(changed.astype(np.uint8), connectivity=8)
    keep = (pieces[:, cv2.CC_STAT_AREA] >= 12) & (pieces[:, cv2.CC_STAT_HEIGHT] >= 5)
    keep[0] = False
    anchor = _pixel_bounds(keep[labels], (left, top))
    return anchor if anchor is not None and _visible_ink(original, anchor) else None


def detector_anchors(obj, source_boxes, original, clean):
    """Require an independent ink anchor for each source joined into an automatic text object."""
    refs = source_box_ids(obj)
    records = [source_boxes.get(ref) for ref in sorted(refs)] if refs else [
        {**(obj.get("region") or {}), "letters_mask": obj.get("letters_mask")}
    ]
    if any(record is None or record.get("removed") or
           (record.get("ocr_eligible") is False and not obj.get("seam_owner")) for record in records):
        return ()
    anchors = tuple(source_ink_bounds(record, original, clean) for record in records)
    return anchors if all(anchors) else ()


def anchor_union(anchors):
    return tuple((min if i < 2 else max)(anchor[i] for anchor in anchors) for i in range(4))


def covers_source_ink(region, anchor):
    left, top, right, bottom = region
    x1, y1, x2, y2 = anchor
    width = max(0, min(x2, right) - max(x1, left))
    height = max(0, min(y2, bottom) - max(y1, top))
    return width * height >= .72 * (x2 - x1) * (y2 - y1)


def ink_window(anchor, region):
    """Allow typographic breathing room inside the actual detection around measured ink."""
    x1, y1, x2, y2 = anchor
    gap = max(8, round(.15 * min(x2 - x1, y2 - y1)))
    return (max(region[0], x1 - gap), max(region[1], y1 - gap),
            min(region[2], x2 + gap), min(region[3], y2 + gap))
