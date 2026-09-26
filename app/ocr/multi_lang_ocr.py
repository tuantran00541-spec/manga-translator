from __future__ import annotations

import threading

import numpy as np
from PIL import Image

from app.env_utils import env_choice
from app.ocr.paddle_v6 import OCRReadResult, PaddleV6OCR
from app.ocr.quality import classify_ocr_quality
from app.parameters import OCR_CENTERED_SINGLE_LINE_ASPECT


class MultiLangOCR:
    def __init__(self):
        self._paddle = PaddleV6OCR()
        self._paddle_target_mode = env_choice(
            "MANGA_OCR_TARGET_SELECTION",
            default="all",
            allowed={"all", "centered"},
        )
        self._manga_ocr = None
        self._manga_lock = threading.RLock()

    def read_probe(self, image: np.ndarray, lang: str) -> OCRReadResult:
        if (lang or "").strip().lower() in {"ja", "japan"}:
            return self.read_detailed(image, "ja")
        return self._paddle.read_single_pass(image, lang)

    def read(self, image: np.ndarray, lang: str) -> str:
        return self.read_detailed(image, lang).text

    def read_detailed(
        self,
        image: np.ndarray,
        lang: str,
        *,
        target_mode: str | None = None,
    ) -> OCRReadResult:
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

        if normalized in {"en", "english"}:
            fast = None
            height, width = image.shape[:2]
            if width >= OCR_CENTERED_SINGLE_LINE_ASPECT * max(1, height):
                fast = self._paddle.read_recognition_only(image, lang)
                if fast.text.strip() and fast.quality == "good":
                    return fast
            effective_target_mode = target_mode or self._paddle_target_mode
            full = self._paddle.read(image, lang, target_mode=effective_target_mode)
            if full.text.strip() or fast is None:
                return full
            return fast

        effective_target_mode = target_mode or self._paddle_target_mode
        return self._paddle.read(
            image,
            lang,
            target_mode=effective_target_mode,
        )

    def _read_manga_ocr(self, image: np.ndarray) -> str:
        with self._manga_lock:
            if self._manga_ocr is None:
                from manga_ocr import MangaOcr

                self._manga_ocr = MangaOcr()
            pil_image = Image.fromarray(image)
            return str(self._manga_ocr(pil_image) or "").strip()
