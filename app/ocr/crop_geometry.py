from __future__ import annotations


import numpy as np

from app.mask_store import decode_mask_value
from app.ocr.identity import (
    ocr_crop_signature,
)
from app.parameters import (
    OCR_BOX_CROP_PADDING,
    OCR_MASK_EDGE_CONTEXT_TRIGGER,
    OCR_MASK_CROP_PADDING,
    OCR_MASK_PAGE_CONTEXT_PADDING,
)
from app.region_policy import geometry_center_in_regions, page_preserve_regions


def _region_overlaps_box(region: dict, box: dict) -> bool:
    if not box:
        return False
    bx = (box.get("x1", 0) + box.get("x2", 0)) / 2
    by = (box.get("y1", 0) + box.get("y2", 0)) / 2
    rx1, ry1 = region.get("x1", 0), region.get("y1", 0)
    rx2, ry2 = region.get("x2", 0), region.get("y2", 0)
    if rx1 <= bx <= rx2 and ry1 <= by <= ry2:
        return True
    ix1 = max(rx1, box.get("x1", 0))
    iy1 = max(ry1, box.get("y1", 0))
    ix2 = min(rx2, box.get("x2", 0))
    iy2 = min(ry2, box.get("y2", 0))
    return ix2 > ix1 and iy2 > iy1



def _clamped_detector_bounds(
    image_shape: tuple[int, ...],
    box: dict,
    *,
    padding: int,
) -> tuple[int, int, int, int]:
    h, w = image_shape[:2]
    try:
        bx1, by1, bx2, by2 = map(
            int, (box["x1"], box["y1"], box["x2"], box["y2"])
        )
    except (KeyError, TypeError, ValueError):
        return 0, 0, 0, 0
    bx1, by1 = max(0, min(w, bx1)), max(0, min(h, by1))
    bx2, by2 = max(bx1, min(w, bx2)), max(by1, min(h, by2))
    if bx2 <= bx1 or by2 <= by1:
        return 0, 0, 0, 0
    pad = max(0, int(padding))
    return (
        max(0, bx1 - pad),
        max(0, by1 - pad),
        min(w, bx2 + pad),
        min(h, by2 + pad),
    )



def box_region(box: dict) -> dict | None:
    try:
        x1, y1, x2, y2 = (
            int(box["x1"]),
            int(box["y1"]),
            int(box["x2"]),
            int(box["y2"]),
        )
    except (KeyError, TypeError, ValueError):
        return None
    if x1 >= x2 or y1 >= y2:
        return None
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}



def ocr_text_region(
    image_shape: tuple[int, ...],
    crop_bounds: tuple[int, int, int, int],
    result: object | None,
    box: dict,
) -> dict | None:
    h, w = image_shape[:2]
    text_bounds = getattr(result, "text_bounds", None) if result is not None else None
    input_shape = getattr(result, "input_shape", None) if result is not None else None
    if text_bounds and input_shape:
        try:
            prepared_h = max(1.0, float(input_shape[0]))
            prepared_w = max(1.0, float(input_shape[1]))
            tx1, ty1, tx2, ty2 = (float(value) for value in text_bounds)
            cx1, cy1, cx2, cy2 = crop_bounds
            crop_w = max(1.0, float(cx2 - cx1))
            crop_h = max(1.0, float(cy2 - cy1))
            x1 = int(round(cx1 + tx1 * crop_w / prepared_w))
            y1 = int(round(cy1 + ty1 * crop_h / prepared_h))
            x2 = int(round(cx1 + tx2 * crop_w / prepared_w))
            y2 = int(round(cy1 + ty2 * crop_h / prepared_h))
            x1, y1 = max(0, min(w, x1)), max(0, min(h, y1))
            x2, y2 = max(x1, min(w, x2)), max(y1, min(h, y2))
            if x1 < x2 and y1 < y2:
                return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        except (TypeError, ValueError, IndexError):
            pass
    region = box_region(box)
    if region is None:
        return None
    region["x1"] = max(0, min(w, region["x1"]))
    region["y1"] = max(0, min(h, region["y1"]))
    region["x2"] = max(region["x1"], min(w, region["x2"]))
    region["y2"] = max(region["y1"], min(h, region["y2"]))
    return region if region["x1"] < region["x2"] and region["y1"] < region["y2"] else None



def ocr_crop_bounds(
    image_shape: tuple[int, ...], box: dict
) -> tuple[int, int, int, int]:
    base = _clamped_detector_bounds(
        image_shape,
        box,
        padding=OCR_BOX_CROP_PADDING,
    )
    if base == (0, 0, 0, 0):
        return base

    h, w = image_shape[:2]
    try:
        bx1, by1, bx2, by2 = map(
            int, (box["x1"], box["y1"], box["x2"], box["y2"])
        )
    except (KeyError, TypeError, ValueError):
        return base
    bx1, by1 = max(0, min(w, bx1)), max(0, min(h, by1))
    bx2, by2 = max(bx1, min(w, bx2)), max(by1, min(h, by2))

    raw_mask = box.get("mask")
    mask = raw_mask if isinstance(raw_mask, np.ndarray) else decode_mask_value(raw_mask)
    expected_shape = (by2 - by1, bx2 - bx1)
    if mask is None or mask.shape != expected_shape:
        return base

    ys, xs = np.nonzero(mask > 127)
    if not xs.size or not ys.size:
        return base

    pad = int(OCR_MASK_CROP_PADDING)
    trigger = int(OCR_MASK_EDGE_CONTEXT_TRIGGER)
    context = int(OCR_MASK_PAGE_CONTEXT_PADDING)
    left = context if int(xs.min()) <= trigger else 0
    top = context if int(ys.min()) <= trigger else 0
    right = context if int(xs.max()) >= mask.shape[1] - 1 - trigger else 0
    bottom = context if int(ys.max()) >= mask.shape[0] - 1 - trigger else 0

    mask_bounds = (
        max(0, bx1 + int(xs.min()) - pad - left),
        max(0, by1 + int(ys.min()) - pad - top),
        min(w, bx1 + int(xs.max()) + 1 + pad + right),
        min(h, by1 + int(ys.max()) + 1 + pad + bottom),
    )
    if mask_bounds[2] <= mask_bounds[0] or mask_bounds[3] <= mask_bounds[1]:
        return base

    return (
        min(base[0], mask_bounds[0]),
        min(base[1], mask_bounds[1]),
        max(base[2], mask_bounds[2]),
        max(base[3], mask_bounds[3]),
    )



_OCR_CONTEXT_RETRY_PADDING = max(
    32,
    int(OCR_BOX_CROP_PADDING) + int(OCR_MASK_PAGE_CONTEXT_PADDING),
)



def expanded_context_crop_bounds(
    image_shape: tuple[int, ...],
    box: dict,
) -> tuple[int, int, int, int]:
    return _clamped_detector_bounds(
        image_shape,
        box,
        padding=_OCR_CONTEXT_RETRY_PADDING,
    )



OCR_EDGE_RECROP_EXTRA_PADDINGS = (24, 48, 96)



def expand_ocr_crop_bounds(
    image_shape: tuple[int, ...],
    crop_bounds: tuple[int, int, int, int],
    extra_padding: int = OCR_EDGE_RECROP_EXTRA_PADDINGS[0],
) -> tuple[int, int, int, int]:
    h, w = image_shape[:2]
    x1, y1, x2, y2 = crop_bounds
    pad = max(0, int(extra_padding))
    return (
        max(0, int(x1) - pad),
        max(0, int(y1) - pad),
        min(w, int(x2) + pad),
        min(h, int(y2) + pad),
    )



def edge_recrop_bounds_sequence(
    image_shape: tuple[int, ...],
    crop_bounds: tuple[int, int, int, int],
) -> tuple[tuple[int, int, int, int], ...]:
    seen = {tuple(int(v) for v in crop_bounds)}
    bounds = []
    for extra_padding in OCR_EDGE_RECROP_EXTRA_PADDINGS:
        expanded = expand_ocr_crop_bounds(
            image_shape, crop_bounds, extra_padding=extra_padding
        )
        if expanded in seen:
            continue
        seen.add(expanded)
        bounds.append(expanded)
    return tuple(bounds)



def active_overlap_signatures(
    page: dict, region: dict
) -> list[tuple[str, str]]:
    matches: list[tuple[int, int, str, str]] = []
    preserve_regions = page_preserve_regions(page)
    for box in page.get("boxes", []) or []:
        if not isinstance(box, dict) or box.get("removed"):
            continue
        if box.get("ocr_eligible") is False:
            continue
        if geometry_center_in_regions(box, preserve_regions):
            continue
        box_id = box.get("id")
        if not box_id or not _region_overlaps_box(region, box):
            continue
        matches.append(
            (
                int(box.get("y1", 0)),
                int(box.get("x1", 0)),
                str(box_id),
                ocr_crop_signature(box),
            )
        )
    matches.sort(key=lambda item: (item[0], item[1], item[2]))
    return [(box_id, geometry) for _, _, box_id, geometry in matches]
