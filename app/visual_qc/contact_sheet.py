from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from app.visual_qc.regions import QCRegion


@dataclass(frozen=True)
class ContactSheetItem:
    region_id: str
    page_index: int
    source_bbox: tuple[int, int, int, int]
    sheet_bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class ContactSheet:
    image: np.ndarray
    items: tuple[ContactSheetItem, ...]
    scale: float


def _normalize_image(image: np.ndarray, region_id: str) -> np.ndarray:
    if not isinstance(image, np.ndarray) or image.ndim not in (2, 3) or image.size == 0:
        raise ValueError(f"Invalid crop image for {region_id}")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def _fit_scale(
    *,
    max_side: int,
    columns: int,
    rows: int,
    source_cell_image_width: int,
    source_cell_image_height: int,
    label_height: int,
    gutter: int,
    fixed_cell_width: int = 0,
) -> float:
    available_w = max_side - (columns + 1) * gutter - columns * fixed_cell_width
    available_h = max_side - (rows + 1) * gutter - rows * label_height
    if available_w <= 0 or available_h <= 0:
        raise ValueError("max_side is too small for contact-sheet labels and gutters")
    scale_w = available_w / float(max(1, columns * source_cell_image_width))
    scale_h = available_h / float(max(1, rows * source_cell_image_height))
    scale = min(1.0, scale_w, scale_h)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Could not fit contact sheet within max_side")
    return scale


def _resize_for_sheet(image: np.ndarray, scale: float) -> np.ndarray:
    if scale >= 1.0:
        return image
    h, w = image.shape[:2]
    out_w = max(1, int(math.floor(w * scale)))
    out_h = max(1, int(math.floor(h * scale)))
    return cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_AREA)


def build_contact_sheet(crops: list[tuple[QCRegion, np.ndarray]], *, max_side: int = 2048, max_columns: int = 2, label_height: int = 30, gutter: int = 12) -> ContactSheet:
    if not crops:
        raise ValueError("At least one crop is required")
    if max_side <= 0 or max_columns <= 0:
        raise ValueError("max_side and max_columns must be positive")
    if label_height < 0 or gutter < 0:
        raise ValueError("label_height and gutter must be non-negative")

    normalized = [(region, _normalize_image(image, region.region_id)) for region, image in crops]
    columns = min(max_columns, len(normalized))
    rows = int(math.ceil(len(normalized) / columns))
    source_cell_w = max(int(image.shape[1]) for _region, image in normalized)
    source_img_h = max(int(image.shape[0]) for _region, image in normalized)
    scale = _fit_scale(
        max_side=max_side,
        columns=columns,
        rows=rows,
        source_cell_image_width=source_cell_w,
        source_cell_image_height=source_img_h,
        label_height=label_height,
        gutter=gutter,
    )
    normalized = [(region, _resize_for_sheet(image, scale)) for region, image in normalized]

    cell_w = max(int(image.shape[1]) for _region, image in normalized)
    image_cell_h = max(int(image.shape[0]) for _region, image in normalized)
    cell_h = image_cell_h + label_height
    sheet_w = columns * cell_w + (columns + 1) * gutter
    sheet_h = rows * cell_h + (rows + 1) * gutter
    if sheet_w > max_side or sheet_h > max_side:
        raise ValueError("Contact-sheet layout exceeded max_side after scaling")

    sheet = np.full((sheet_h, sheet_w, 3), 255, dtype=np.uint8)
    placements: list[ContactSheetItem] = []
    for idx, (region, image) in enumerate(normalized):
        row, col = divmod(idx, columns)
        x = gutter + col * (cell_w + gutter)
        y = gutter + row * (cell_h + gutter)
        cv2.putText(sheet, region.region_id, (x + 4, y + max(18, label_height - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        image_y = y + label_height
        h, w = image.shape[:2]
        sheet[image_y:image_y + h, x:x + w] = image
        placements.append(ContactSheetItem(region.region_id, region.page_index, region.bbox, (x, image_y, x + w, image_y + h)))
    return ContactSheet(image=sheet, items=tuple(placements), scale=scale)


def build_pair_contact_sheet(pairs: list[tuple[QCRegion, np.ndarray, np.ndarray]], *, max_side: int = 2048, max_columns: int = 1, label_height: int = 34, gutter: int = 12, pair_gap: int = 10) -> ContactSheet:
    if not pairs:
        raise ValueError("At least one region pair is required")
    if max_side <= 0 or max_columns <= 0:
        raise ValueError("max_side and max_columns must be positive")
    if label_height < 0 or gutter < 0 or pair_gap < 0:
        raise ValueError("label_height, gutter and pair_gap must be non-negative")

    normalized: list[tuple[QCRegion, np.ndarray, np.ndarray]] = []
    for region, original, cleaned in pairs:
        original = _normalize_image(original, region.region_id)
        cleaned = _normalize_image(cleaned, region.region_id)
        if original.shape[:2] != cleaned.shape[:2]:
            raise ValueError(f"Original/clean crop dimensions differ for {region.region_id}")
        normalized.append((region, original, cleaned))

    columns = min(max_columns, len(normalized))
    rows = int(math.ceil(len(normalized) / columns))
    source_crop_w = max(int(original.shape[1]) for _region, original, _cleaned in normalized)
    source_crop_h = max(int(original.shape[0]) for _region, original, _cleaned in normalized)
    scale = _fit_scale(
        max_side=max_side,
        columns=columns,
        rows=rows,
        source_cell_image_width=source_crop_w * 2,
        source_cell_image_height=source_crop_h,
        label_height=label_height,
        gutter=gutter,
        fixed_cell_width=pair_gap,
    )
    normalized = [
        (region, _resize_for_sheet(original, scale), _resize_for_sheet(cleaned, scale))
        for region, original, cleaned in normalized
    ]

    crop_w = max(int(original.shape[1]) for _region, original, _cleaned in normalized)
    crop_h = max(int(original.shape[0]) for _region, original, _cleaned in normalized)
    cell_w = crop_w * 2 + pair_gap
    cell_h = crop_h + label_height
    sheet_w = columns * cell_w + (columns + 1) * gutter
    sheet_h = rows * cell_h + (rows + 1) * gutter
    if sheet_w > max_side or sheet_h > max_side:
        raise ValueError("Pair contact-sheet layout exceeded max_side after scaling")

    sheet = np.full((sheet_h, sheet_w, 3), 255, dtype=np.uint8)
    placements: list[ContactSheetItem] = []
    for idx, (region, original, cleaned) in enumerate(normalized):
        row, col = divmod(idx, columns)
        x = gutter + col * (cell_w + gutter)
        y = gutter + row * (cell_h + gutter)
        cv2.putText(sheet, f"{region.region_id}  ORIGINAL | CLEAN", (x + 4, y + max(18, label_height - 9)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        image_y = y + label_height
        oh, ow = original.shape[:2]
        ch, cw = cleaned.shape[:2]
        sheet[image_y:image_y + oh, x:x + ow] = original
        clean_x = x + crop_w + pair_gap
        sheet[image_y:image_y + ch, clean_x:clean_x + cw] = cleaned
        placements.append(ContactSheetItem(region.region_id, region.page_index, region.bbox, (clean_x, image_y, clean_x + cw, image_y + ch)))
    return ContactSheet(image=sheet, items=tuple(placements), scale=scale)
