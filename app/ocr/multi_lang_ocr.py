from __future__ import annotations

import threading

import numpy as np
from PIL import Image

from app.ocr.paddle_v6 import OCRReadResult, PaddleV6OCR
from app.ocr.quality import classify_ocr_quality


class MultiLangOCR:
    """Production OCR facade for manga/manhua/webtoon crops.

    Japanese stays on MangaOCR because the Japanese ground-truth gate still
    shows materially better exact transcription there. English and Chinese use
    PP-OCRv6; Korean uses the dedicated Korean PP-OCRv5 mobile recognizer behind
    the same PaddleOCR 3.x detector.
    """

    def __init__(self):
        self._paddle = PaddleV6OCR()
        self._manga_ocr = None
        self._manga_lock = threading.RLock()

    def read(self, image: np.ndarray, lang: str) -> str:
        return self.read_detailed(image, lang).text

    def read_detailed(self, image: np.ndarray, lang: str) -> OCRReadResult:
        if image is None or image.size == 0:
            return OCRReadResult("", None, "none", "unknown", 0, "reject", "empty")

        normalized = (lang or "").strip().lower()
        if normalized in {"ja", "japan"}:
            text = self._read_manga_ocr(image)
            quality = classify_ocr_quality(text, "ja", confidence=None)
            return OCRReadResult(
                text=text,
                confidence=None,
                model="manga-ocr",
                orientation="unknown",
                region_count=1 if text else 0,
                quality=quality.status,
                quality_reason=quality.reason,
            )

        return self._paddle.read(image, lang)

    def _read_manga_ocr(self, image: np.ndarray) -> str:
        with self._manga_lock:
            if self._manga_ocr is None:
                from manga_ocr import MangaOcr

                self._manga_ocr = MangaOcr()
            pil_image = Image.fromarray(image)
            return str(self._manga_ocr(pil_image) or "").strip()
