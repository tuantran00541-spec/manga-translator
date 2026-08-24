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


def build_contact_sheet(crops: list[tuple[QCRegion, np.ndarray]], *, max_side: int = 2048, max_columns: int = 2, label_height: int = 30, gutter: int = 12) -> ContactSheet:
    if not crops:
        raise ValueError("At least one crop is required")
    if max_side <= 0 or max_columns <= 0:
        raise ValueError("max_side and max_columns must be positive")

    normalized: list[tuple[QCRegion, np.ndarray]] = []
    for region, image in crops:
        if not isinstance(image, np.ndarray) or image.ndim not in (2, 3) or image.size == 0:
            raise ValueError(f"Invalid crop image for {region.region_id}")
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        normalized.append((region, image))

    columns = min(max_columns, len(normalized))
    rows = int(math.ceil(len(normalized) / columns))
    cell_w = max(int(image.shape[1]) for _region, image in normalized)
    cell_h = max(int(image.shape[0]) for _region, image in normalized) + label_height
    sheet_w = columns * cell_w + (columns + 1) * gutter
    sheet_h = rows * cell_h + (rows + 1) * gutter
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

    scale = min(1.0, float(max_side) / max(sheet_h, sheet_w))
    if scale < 1.0:
        out_w = max(1, int(round(sheet_w * scale)))
        out_h = max(1, int(round(sheet_h * scale)))
        sheet = cv2.resize(sheet, (out_w, out_h), interpolation=cv2.INTER_AREA)
        placements = [ContactSheetItem(item.region_id, item.page_index, item.source_bbox, tuple(int(round(v * scale)) for v in item.sheet_bbox)) for item in placements]

    return ContactSheet(image=sheet, items=tuple(placements), scale=scale)
