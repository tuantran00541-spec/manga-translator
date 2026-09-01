from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

from app.parameters import (
    VISUAL_QC_DEEP_AREA_RATIO,
    VISUAL_QC_MANUAL_COMPONENT_AREA_MIN,
    VISUAL_QC_MANUAL_MASK_THRESHOLD,
    VISUAL_QC_MERGE_GAP,
    VISUAL_QC_REGION_MARGIN,
)

QC_PIPELINE_VERSION = 1


@dataclass(frozen=True)
class QCRegion:
    page_index: int
    region_id: str
    bbox: tuple[int, int, int, int]
    source_box_ids: tuple[str, ...]
    source_kinds: tuple[str, ...]
    area_ratio: float
    requires_deep_qc: bool


@dataclass
class _RegionSeed:
    bbox: tuple[int, int, int, int]
    source_box_ids: set[str]
    source_kinds: set[str]


def _page_dimensions(page: dict, manual_mask: np.ndarray | None) -> tuple[int, int]:
    try:
        width = int(page.get("width") or 0)
        height = int(page.get("height") or 0)
    except (TypeError, ValueError):
        width = height = 0
    if width > 0 and height > 0:
        return width, height
    if manual_mask is not None and manual_mask.ndim >= 2:
        height, width = manual_mask.shape[:2]
        if width > 0 and height > 0:
            return int(width), int(height)
    raise ValueError("Page width/height are required for visual QC region extraction")


def _normalize_bbox(raw: dict, width: int, height: int) -> tuple[int, int, int, int] | None:
    try:
        x1 = int(round(float(raw.get("x1"))))
        y1 = int(round(float(raw.get("y1"))))
        x2 = int(round(float(raw.get("x2"))))
        y2 = int(round(float(raw.get("y2"))))
    except (TypeError, ValueError):
        return None
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _expand_bbox(bbox: tuple[int, int, int, int], *, margin: int, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return max(0, x1 - margin), max(0, y1 - margin), min(width, x2 + margin), min(height, y2 + margin)


def _manual_mask_seeds(manual_mask: np.ndarray | None, *, width: int, height: int, min_component_area: int) -> list[_RegionSeed]:
    if manual_mask is None:
        return []
    if manual_mask.ndim == 3:
        manual_mask = cv2.cvtColor(manual_mask, cv2.COLOR_BGR2GRAY)
    if manual_mask.shape[:2] != (height, width):
        manual_mask = cv2.resize(manual_mask, (width, height), interpolation=cv2.INTER_NEAREST)
    binary = (manual_mask > VISUAL_QC_MANUAL_MASK_THRESHOLD).astype(np.uint8)
    if not np.any(binary):
        return []
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out: list[_RegionSeed] = []
    for label in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[label])
        if area < min_component_area or w <= 0 or h <= 0:
            continue
        out.append(_RegionSeed((x, y, x + w, y + h), set(), {"manual_mask"}))
    return out


def _near_or_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 + gap < bx1 or bx2 + gap < ax1 or ay2 + gap < by1 or by2 + gap < ay1)


def _merge_bbox(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def _merge_seeds(seeds: Iterable[_RegionSeed], merge_gap: int) -> list[_RegionSeed]:
    remaining = sorted(seeds, key=lambda seed: (seed.bbox[1], seed.bbox[0], seed.bbox[3], seed.bbox[2]))
    merged: list[_RegionSeed] = []
    while remaining:
        current = remaining.pop(0)
        changed = True
        while changed:
            changed = False
            keep: list[_RegionSeed] = []
            for candidate in remaining:
                if _near_or_overlap(current.bbox, candidate.bbox, merge_gap):
                    current.bbox = _merge_bbox(current.bbox, candidate.bbox)
                    current.source_box_ids.update(candidate.source_box_ids)
                    current.source_kinds.update(candidate.source_kinds)
                    changed = True
                else:
                    keep.append(candidate)
            remaining = keep
        merged.append(current)
    return sorted(merged, key=lambda seed: (seed.bbox[1], seed.bbox[0], seed.bbox[3], seed.bbox[2]))


def extract_candidate_regions(
    page: dict,
    page_index: int,
    *,
    manual_mask: np.ndarray | None = None,
    margin: int = VISUAL_QC_REGION_MARGIN,
    merge_gap: int = VISUAL_QC_MERGE_GAP,
    min_manual_component_area: int = VISUAL_QC_MANUAL_COMPONENT_AREA_MIN,
    deep_area_ratio: float = VISUAL_QC_DEEP_AREA_RATIO,
) -> list[QCRegion]:
    """Build deterministic QC regions from canonical changed/inpaint sources."""
    if page_index < 0:
        raise ValueError("page_index must be non-negative")
    if margin < 0 or merge_gap < 0:
        raise ValueError("margin and merge_gap must be non-negative")
    width, height = _page_dimensions(page, manual_mask)
    seeds: list[_RegionSeed] = []
    for box in page.get("boxes") or []:
        if not isinstance(box, dict) or box.get("removed"):
            continue
        bbox = _normalize_bbox(box, width, height)
        if bbox is None:
            continue
        box_id = str(box.get("id") or "").strip()
        seeds.append(_RegionSeed(bbox, {box_id} if box_id else set(), {"box"}))
    seeds.extend(
        _manual_mask_seeds(
            manual_mask,
            width=width,
            height=height,
            min_component_area=max(1, int(min_manual_component_area)),
        )
    )
    expanded = [
        _RegionSeed(
            _expand_bbox(seed.bbox, margin=int(margin), width=width, height=height),
            set(seed.source_box_ids),
            set(seed.source_kinds),
        )
        for seed in seeds
    ]
    merged = _merge_seeds(expanded, int(merge_gap))
    page_area = float(max(1, width * height))
    regions: list[QCRegion] = []
    for ordinal, seed in enumerate(merged, start=1):
        x1, y1, x2, y2 = seed.bbox
        area_ratio = ((x2 - x1) * (y2 - y1)) / page_area
        regions.append(
            QCRegion(
                page_index,
                f"P{page_index + 1:04d}-R{ordinal:02d}",
                seed.bbox,
                tuple(sorted(seed.source_box_ids)),
                tuple(sorted(seed.source_kinds)),
                area_ratio,
                area_ratio >= float(deep_area_ratio),
            )
        )
    return regions


def qc_cache_identity(page: dict, *, model: str, mode: str, pipeline_version: int = QC_PIPELINE_VERSION) -> dict:
    return {
        "source_revision": int(page.get("source_revision") or 0),
        "clean_revision": int(page.get("clean_revision") or 0),
        "model": str(model),
        "mode": str(mode),
        "pipeline_version": int(pipeline_version),
    }


def qc_cache_matches(cached_identity: dict | None, page: dict, *, model: str, mode: str, pipeline_version: int = QC_PIPELINE_VERSION) -> bool:
    return (
        isinstance(cached_identity, dict)
        and cached_identity
        == qc_cache_identity(
            page,
            model=model,
            mode=mode,
            pipeline_version=pipeline_version,
        )
    )
