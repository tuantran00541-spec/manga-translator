from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.visual_qc.jobs import QCWorkItem
from app.visual_qc.regions import QCRegion, extract_candidate_regions


@dataclass(frozen=True)
class ChapterQCPlan:
    global_regions: tuple[QCRegion, ...]
    candidate_regions: tuple[QCRegion, ...]
    deep_region_ids: tuple[str, ...]
    skipped_pages: tuple[int, ...]


def _full_page_region(page: dict, page_index: int) -> QCRegion:
    width = int(page.get("width") or 0)
    height = int(page.get("height") or 0)
    if width <= 0 or height <= 0:
        raise ValueError(f"Page {page_index} width/height are required for chapter QC")
    return QCRegion(page_index, f"P{page_index + 1:04d}-GLOBAL", (0, 0, width, height), (), ("global",), 1.0, False)


def build_chapter_qc_plan(manifest: dict, *, manual_masks: dict[int, np.ndarray] | None = None, margin: int = 64, merge_gap: int = 32, deep_area_ratio: float = 0.35) -> ChapterQCPlan:
    manual_masks = manual_masks or {}
    global_regions: list[QCRegion] = []
    candidate_regions: list[QCRegion] = []
    skipped_pages: list[int] = []
    for page_index, page in enumerate(manifest.get("pages") or []):
        if not isinstance(page, dict) or page.get("skipped") or not page.get("clean"):
            skipped_pages.append(page_index)
            continue
        global_regions.append(_full_page_region(page, page_index))
        candidate_regions.extend(extract_candidate_regions(page, page_index, manual_mask=manual_masks.get(page_index), margin=margin, merge_gap=merge_gap, deep_area_ratio=deep_area_ratio))
    deep_region_ids = tuple(region.region_id for region in candidate_regions if region.requires_deep_qc)
    return ChapterQCPlan(tuple(global_regions), tuple(candidate_regions), deep_region_ids, tuple(skipped_pages))


def chunk_regions(regions: tuple[QCRegion, ...] | list[QCRegion], *, batch_size: int, work_prefix: str) -> list[QCWorkItem]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    ordered = list(regions)
    items: list[QCWorkItem] = []
    for start in range(0, len(ordered), int(batch_size)):
        batch = ordered[start:start + int(batch_size)]
        page_indices = tuple(dict.fromkeys(region.page_index for region in batch))
        items.append(QCWorkItem(f"{work_prefix}-{len(items) + 1:04d}", tuple(region.region_id for region in batch), page_indices))
    return items
