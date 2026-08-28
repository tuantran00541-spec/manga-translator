from __future__ import annotations

from collections import OrderedDict
import os
from pathlib import Path
import threading

import cv2
import numpy as np

from app.ocr.identity import file_revision
from app.ocr.service import OCRService, ocr_crop_from_box
from app.pipeline import read_image


def _cache_budget_bytes() -> int:
    try:
        value = int(os.getenv("MANGA_OCR_IMAGE_CACHE_MB", "128"))
    except ValueError:
        value = 128
    return max(0, min(value, 1024)) * 1024 * 1024


class CachedOCRService(OCRService):
    """OCR service with a small revision-keyed decoded-page cache.

    Chapter OCR schedules boxes one by one.  Without a page cache every box on
    the same slice re-opens and decodes the same PNG/JPEG before the OCR model can
    run.  The cache is keyed by file identity, bounded by bytes rather than item
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
            # outside the lock.  Reuse that copy instead of double-accounting it.
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
        crop = ocr_crop_from_box(image, box_snapshot)
        if not crop.size:
            return ""
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return self.ocr.read(rgb, lang)
