from __future__ import annotations


import cv2
import numpy as np

from app.mask_store import decode_mask_value
from app.ocr.crop_geometry import box_region, ocr_text_region


def visual_cache_complete(box: dict) -> bool:
    if not str(box.get("ocr_text") or "").strip():
        return True
    return bool(
        box.get("ocr_text_color")
        and box.get("ocr_font_size")
        and isinstance(box.get("ocr_text_region"), dict)
    )



def _rgb_hex_from_bgr(values: np.ndarray) -> str | None:
    if values.size == 0:
        return None
    median = np.median(values.reshape(-1, 3), axis=0)
    b, g, r = (int(np.clip(round(float(value)), 0, 255)) for value in median)
    return f"#{r:02x}{g:02x}{b:02x}"



def sample_source_text_color(
    image: np.ndarray,
    box: dict,
    text_region: dict | None,
) -> str | None:
    if image is None or image.size == 0:
        return None

    if str(box.get("source_role") or "").strip().lower() == "text_segmenter":
        region = box_region(box)
        raw_mask = box.get("mask")
        mask = raw_mask if isinstance(raw_mask, np.ndarray) else decode_mask_value(raw_mask)
        if region is not None and mask is not None:
            expected = (region["y2"] - region["y1"], region["x2"] - region["x1"])
            if mask.shape == expected:
                patch = image[
                    region["y1"]:region["y2"],
                    region["x1"]:region["x2"],
                ]
                if patch.shape[:2] == mask.shape:
                    pixels = patch[mask > 127]
                    if pixels.ndim == 2 and pixels.shape[0] >= 8:
                        sampled = _rgb_hex_from_bgr(pixels)
                        if sampled:
                            return sampled

    if not isinstance(text_region, dict):
        return None
    x1, y1, x2, y2 = (
        int(text_region["x1"]),
        int(text_region["y1"]),
        int(text_region["x2"]),
        int(text_region["y2"]),
    )
    roi = image[y1:y2, x1:x2]
    if roi.size == 0 or min(roi.shape[:2]) < 2:
        return None

    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    edge = max(1, min(roi.shape[:2]) // 8)
    border = np.concatenate(
        [
            lab[:edge].reshape(-1, 3),
            lab[-edge:].reshape(-1, 3),
            lab[:, :edge].reshape(-1, 3),
            lab[:, -edge:].reshape(-1, 3),
        ],
        axis=0,
    )
    background = np.median(border, axis=0)
    distance = np.linalg.norm(lab - background, axis=2)
    percentile = float(np.percentile(distance, 70.0))
    threshold = max(12.0, percentile)
    foreground = roi[distance >= threshold]
    if foreground.ndim != 2 or foreground.shape[0] < 8:
        flat = distance.reshape(-1)
        count = min(flat.size, max(8, flat.size // 4))
        if count <= 0:
            return None
        indices = np.argpartition(flat, -count)[-count:]
        foreground = roi.reshape(-1, 3)[indices]
    return _rgb_hex_from_bgr(foreground)



def source_font_size(
    crop_bounds: tuple[int, int, int, int],
    result: object | None,
    text_region: dict | None,
    region_count: int,
) -> int | None:
    if result is not None:
        hint = getattr(result, "font_size_hint", None)
        input_shape = getattr(result, "input_shape", None)
        if hint is not None and input_shape:
            try:
                cx1, cy1, cx2, cy2 = crop_bounds
                prepared_h = max(1.0, float(input_shape[0]))
                prepared_w = max(1.0, float(input_shape[1]))
                orientation = str(getattr(result, "orientation", "horizontal") or "horizontal").lower()
                scale = (
                    max(1.0, float(cx2 - cx1)) / prepared_w
                    if orientation == "vertical"
                    else max(1.0, float(cy2 - cy1)) / prepared_h
                )
                size = int(round(float(hint) * scale))
                if size > 0:
                    return max(4, min(512, size))
            except (TypeError, ValueError, IndexError):
                pass

    if isinstance(text_region, dict):
        height = max(1, int(text_region["y2"]) - int(text_region["y1"]))
        lines = max(1, int(region_count or 1))
        return max(4, min(512, int(round(height / lines))))
    return None



def visual_text_metadata(
    image: np.ndarray,
    box: dict,
    crop_bounds: tuple[int, int, int, int],
    result: object | None,
    *,
    text: str,
    region_count: int,
) -> dict:
    if not str(text or "").strip():
        return {}
    text_region = ocr_text_region(image.shape, crop_bounds, result, box)
    color = sample_source_text_color(image, box, text_region)
    font_size = source_font_size(crop_bounds, result, text_region, region_count)
    metadata: dict = {}
    if text_region is not None:
        metadata["text_region"] = text_region
    if color:
        metadata["text_color"] = color
    if font_size is not None:
        metadata["font_size"] = font_size
    return metadata



def sync_group_visual_metadata(
    obj: dict,
    page: dict,
    source_box_ids: list[str],
) -> None:
    source_ids = {str(value) for value in source_box_ids}
    source_boxes = [
        box
        for box in (page.get("boxes") or [])
        if isinstance(box, dict) and str(box.get("id") or "") in source_ids
    ]

    colors = [
        str(box.get("ocr_text_color") or "")
        for box in source_boxes
        if str(box.get("ocr_text_color") or "")
    ]
    if colors:
        counts: dict[str, int] = {}
        for color in colors:
            counts[color] = counts.get(color, 0) + 1
        obj["ocr_text_color"] = max(counts, key=counts.get)
    else:
        obj.pop("ocr_text_color", None)

    sizes: list[int] = []
    for box in source_boxes:
        try:
            size = int(box.get("ocr_font_size") or 0)
        except (TypeError, ValueError):
            continue
        if size > 0:
            sizes.append(size)
    if sizes:
        obj["ocr_font_size"] = int(round(float(np.median(sizes))))
    else:
        obj.pop("ocr_font_size", None)

    regions = [
        box.get("ocr_text_region")
        for box in source_boxes
        if isinstance(box.get("ocr_text_region"), dict)
    ]
    valid_regions = [region for region in regions if box_region(region) is not None]
    if valid_regions:
        obj["ocr_text_region"] = {
            "x1": min(int(region["x1"]) for region in valid_regions),
            "y1": min(int(region["y1"]) for region in valid_regions),
            "x2": max(int(region["x2"]) for region in valid_regions),
            "y2": max(int(region["y2"]) for region in valid_regions),
        }
    else:
        obj.pop("ocr_text_region", None)
