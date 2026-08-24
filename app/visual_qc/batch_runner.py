from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.security import MAX_IMAGE_PIXELS
from app.visual_qc.batch_protocol import RegionBatchDecision
from app.visual_qc.contact_sheet import build_contact_sheet, build_pair_contact_sheet
from app.visual_qc.jobs import QCWorkItem
from app.visual_qc.regions import QCRegion


class RegionBatchRunner:
    def __init__(self, client):
        self.client = client

    @staticmethod
    def _read(path_value: str | Path) -> np.ndarray:
        path = Path(path_value)
        if not path.is_file():
            raise FileNotFoundError(path)
        data = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image at {path}")
        h, w = image.shape[:2]
        if w * h > MAX_IMAGE_PIXELS:
            raise ValueError(f"Image too large at {path}: {w}x{h}")
        return image

    @staticmethod
    def _crop(image: np.ndarray, region: QCRegion) -> np.ndarray:
        x1, y1, x2, y2 = region.bbox
        h, w = image.shape[:2]
        x1 = max(0, min(w, int(x1)))
        x2 = max(0, min(w, int(x2)))
        y1 = max(0, min(h, int(y1)))
        y2 = max(0, min(h, int(y2)))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid crop for {region.region_id}")
        return image[y1:y2, x1:x2]

    @staticmethod
    def _page_for_region(pages: list, region: QCRegion) -> dict:
        if region.page_index < 0 or region.page_index >= len(pages):
            raise ValueError(f"Invalid page index for {region.region_id}")
        page = pages[region.page_index]
        if not isinstance(page, dict):
            raise ValueError(f"Invalid page data for {region.region_id}")
        return page

    def _cached_page_image(
        self,
        cache: dict[int, np.ndarray],
        page: dict,
        region: QCRegion,
        field: str,
    ) -> np.ndarray:
        if region.page_index not in cache:
            cache[region.page_index] = self._read(page.get(field) or "")
        return cache[region.page_index]

    def _build_clean_sheet(self, regions: list[QCRegion], pages: list):
        clean_cache: dict[int, np.ndarray] = {}
        crops = []
        for region in regions:
            page = self._page_for_region(pages, region)
            clean = self._cached_page_image(clean_cache, page, region, "clean")
            crops.append((region, self._crop(clean, region)))
        return build_contact_sheet(crops)

    def _build_pair_sheet(self, regions: list[QCRegion], pages: list):
        clean_cache: dict[int, np.ndarray] = {}
        original_cache: dict[int, np.ndarray] = {}
        pairs = []
        for region in regions:
            page = self._page_for_region(pages, region)
            original = self._cached_page_image(original_cache, page, region, "original")
            clean = self._cached_page_image(clean_cache, page, region, "clean")
            pairs.append((region, self._crop(original, region), self._crop(clean, region)))
        return build_pair_contact_sheet(pairs)

    def inspect(self, item: QCWorkItem, manifest: dict, regions_by_id: dict[str, QCRegion], api_key: str) -> list[RegionBatchDecision]:
        regions = [regions_by_id[region_id] for region_id in item.region_ids]
        pages = manifest.get("pages") or []
        if item.mode in {"global-clean", "region-clean"}:
            sheet = self._build_clean_sheet(regions, pages)
        elif item.mode == "region-pair":
            sheet = self._build_pair_sheet(regions, pages)
        else:
            raise ValueError(f"Unsupported QC work mode: {item.mode}")

        return self.client.inspect(
            sheet,
            {region.region_id: region for region in regions},
            api_key,
            mode=item.mode,
        )
