from __future__ import annotations

from collections import OrderedDict
import os
from pathlib import Path
import threading

import cv2
import numpy as np

from app.mask_store import decode_mask_value
from app.ocr.identity import file_revision
from app.ocr.service import OCRService
from app.pipeline import read_image


def _cache_budget_bytes() -> int:
    try:
        value = int(os.getenv("MANGA_OCR_IMAGE_CACHE_MB", "128"))
    except ValueError:
        value = 128
    return max(0, min(value, 1024)) * 1024 * 1024


def _ocr_crop_from_box(image: np.ndarray, box: dict) -> np.ndarray:
    """Crop OCR evidence using inline or sidecar detector masks."""
    h, w = image.shape[:2]
    bx1, by1, bx2, by2 = map(
        int, (box["x1"], box["y1"], box["x2"], box["y2"])
    )
    bx1, by1 = max(0, min(w, bx1)), max(0, min(h, by1))
    bx2, by2 = max(bx1, min(w, bx2)), max(by1, min(h, by2))
    if bx2 <= bx1 or by2 <= by1:
        return image[0:0, 0:0]

    mask = decode_mask_value(box.get("mask"))
    expected_shape = (by2 - by1, bx2 - bx1)
    if mask is not None and mask.shape == expected_shape:
        ys, xs = np.nonzero(mask > 127)
        if xs.size and ys.size:
            pad = 12
            x1 = max(bx1, bx1 + int(xs.min()) - pad)
            y1 = max(by1, by1 + int(ys.min()) - pad)
            x2 = min(bx2, bx1 + int(xs.max()) + 1 + pad)
            y2 = min(by2, by1 + int(ys.max()) + 1 + pad)
            if x2 > x1 and y2 > y1:
                return image[y1:y2, x1:x2]

    pad = 20
    return image[
        max(0, by1 - pad) : min(h, by2 + pad),
        max(0, bx1 - pad) : min(w, bx2 + pad),
    ]


class CachedOCRService(OCRService):
    """OCR service with a small revision-keyed decoded-page cache.

    Chapter OCR schedules boxes one by one. Without a page cache every box on
    the same slice re-opens and decodes the same PNG/JPEG before the OCR model can
    run. The cache is keyed by file identity, bounded by bytes rather than item
    count, and stores only immutable source images; stale files therefore miss
    automatically and large pages cannot consume unbounded RAM.
    """

    def __init__(self, ocr_engine, pipeline):
        super().__init__(ocr_engine, pipeline)
        self._image_cache: OrderedDict[
            tuple[str, tuple[int, int, int]], np.ndarray
        ] = OrderedDict()
        self._image_cache_bytes = 0
        self._image_cache_budget = _cache_budget_bytes()
        self._image_cache_lock = threading.RLock()

    def _cached_source_image(self, original_path: Path) -> np.ndarray:
        revision = file_revision(original_path)
        key = (str(original_path), revision)
        with self._image_cache_lock:
            cached = self._image_cache.pop(key, None)
            if cached is not None:
                self._image_cache[key] = cached
                return cached

        image = read_image(original_path)
        image_bytes = int(image.nbytes)
        if self._image_cache_budget <= 0 or image_bytes > self._image_cache_budget:
            return image

        with self._image_cache_lock:
            # Another worker may have decoded the same page while this worker was
            # outside the lock. Reuse that copy instead of double-accounting it.
            cached = self._image_cache.pop(key, None)
            if cached is not None:
                self._image_cache[key] = cached
                return cached

            while (
                self._image_cache
                and self._image_cache_bytes + image_bytes > self._image_cache_budget
            ):
                _old_key, old_image = self._image_cache.popitem(last=False)
                self._image_cache_bytes -= int(old_image.nbytes)

            self._image_cache[key] = image
            self._image_cache_bytes += image_bytes
        return image

    def _read_box_text(self, original_path: Path, box_snapshot: dict, lang: str) -> str:
        image = self._cached_source_image(original_path)
        crop = _ocr_crop_from_box(image, box_snapshot)
        if not crop.size:
            return ""
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return self.ocr.read(rgb, lang)
