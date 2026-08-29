from __future__ import annotations

import os
import threading

import numpy as np
from PIL import Image

from app.ocr.paddle_v6 import OCRReadResult, PaddleV6OCR


def _env_enabled(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _fallback_threshold() -> float:
    try:
        value = float(os.getenv("MANGA_OCR_JA_FALLBACK_CONFIDENCE", "0.55"))
    except ValueError:
        value = 0.55
    return max(0.0, min(1.0, value))


class MultiLangOCR:
    """Production OCR facade backed by PaddleOCR 3.x.

    PP-OCRv6 small handles English, Chinese, and Japanese. Korean uses the
    dedicated Korean PP-OCRv5 mobile recognizer behind the same PaddleOCR
    detector. MangaOCR is retained only as a conservative Japanese safety
    fallback while the Japanese holdout gate is still active.
    """

    def __init__(self):
        self._paddle = PaddleV6OCR()
        self._manga_ocr = None
        self._manga_lock = threading.RLock()
        self._ja_fallback_enabled = _env_enabled("MANGA_OCR_JA_FALLBACK", True)
        self._ja_fallback_threshold = _fallback_threshold()

    def read(self, image: np.ndarray, lang: str) -> str:
        return self.read_detailed(image, lang).text

    def read_detailed(self, image: np.ndarray, lang: str) -> OCRReadResult:
        if image is None or image.size == 0:
            return OCRReadResult("", None, "none", "unknown", 0)

        normalized = (lang or "").strip().lower()
        is_japanese = normalized in {"ja", "japan"}

        try:
            result = self._paddle.read(image, lang)
        except Exception:
            if not (is_japanese and self._ja_fallback_enabled):
                raise
            fallback = self._read_manga_ocr(image)
            if not fallback:
                raise
            return OCRReadResult(
                fallback,
                None,
                "manga-ocr-fallback",
                "unknown",
                1,
            )

        if not (is_japanese and self._ja_fallback_enabled):
            return result

        should_fallback = (
            not result.text
            or result.confidence is None
            or result.confidence < self._ja_fallback_threshold
        )
        if not should_fallback:
            return result

        fallback = self._read_manga_ocr(image)
        if not fallback:
            return result
        return OCRReadResult(
            fallback,
            None,
            "manga-ocr-fallback",
            result.orientation,
            result.region_count,
        )

    def _read_manga_ocr(self, image: np.ndarray) -> str:
        with self._manga_lock:
            if self._manga_ocr is None:
                from manga_ocr import MangaOcr

                self._manga_ocr = MangaOcr()
            pil_image = Image.fromarray(image)
            return str(self._manga_ocr(pil_image) or "").strip()
