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

    def inspect(self, item: QCWorkItem, manifest: dict, regions_by_id: dict[str, QCRegion], api_key: str) -> list[RegionBatchDecision]:
        regions = [regions_by_id[region_id] for region_id in item.region_ids]
        pages = manifest.get("pages") or []
        clean_cache: dict[int, np.ndarray] = {}
        original_cache: dict[int, np.ndarray] = {}

        if item.mode in {"global-clean", "region-clean"}:
            crops = []
            for region in regions:
                if region.page_index < 0 or region.page_index >= len(pages):
                    raise ValueError(f"Invalid page index for {region.region_id}")
                page = pages[region.page_index]
                if region.page_index not in clean_cache:
                    clean_cache[region.page_index] = self._read(page.get("clean") or "")
                crops.append((region, self._crop(clean_cache[region.page_index], region)))
            sheet = build_contact_sheet(crops)
        elif item.mode == "region-pair":
            pairs = []
            for region in regions:
                if region.page_index < 0 or region.page_index >= len(pages):
                    raise ValueError(f"Invalid page index for {region.region_id}")
                page = pages[region.page_index]
                if region.page_index not in clean_cache:
                    clean_cache[region.page_index] = self._read(page.get("clean") or "")
                if region.page_index not in original_cache:
                    original_cache[region.page_index] = self._read(page.get("original") or "")
                pairs.append((
                    region,
                    self._crop(original_cache[region.page_index], region),
                    self._crop(clean_cache[region.page_index], region),
                ))
            sheet = build_pair_contact_sheet(pairs)
        else:
            raise ValueError(f"Unsupported QC work mode: {item.mode}")

        return self.client.inspect(
            sheet,
            {region.region_id: region for region in regions},
            api_key,
            mode=item.mode,
        )
