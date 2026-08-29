from __future__ import annotations

from dataclasses import dataclass
import os
import statistics
import threading
from typing import Any

import cv2
import numpy as np

from app.env_utils import env_enabled
from app.ocr.quality import classify_ocr_quality
from app.ocr.reading_order import reconstruct_reading_order

UNIFIED_LANGS = {"en", "english", "ch", "zh", "ja", "japan"}
KOREAN_LANGS = {"ko", "korean"}


@dataclass(frozen=True)
class OCRReadResult:
    text: str
    confidence: float | None
    model: str
    orientation: str
    region_count: int
    quality: str = "unknown"
    quality_reason: str | None = None


def _payload(result: Any) -> dict[str, Any]:
    value = getattr(result, "json", result)
    if callable(value):
        value = value()
    if isinstance(value, str):
        import json

        value = json.loads(value)
    if not isinstance(value, dict):
        value = dict(value)
    inner = value.get("res", value)
    return inner if isinstance(inner, dict) else value


def _normalize_lang(lang: str) -> str:
    normalized = (lang or "").strip().lower()
    mapping = {
        "english": "en",
        "japan": "ja",
        "zh": "ch",
        "ko": "korean",
    }
    return mapping.get(normalized, normalized)


def _prepare_rgb_for_paddle(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        return image
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    else:
        # MultiLangOCR receives RGB images from OCRService.
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    height, width = bgr.shape[:2]
    shortest = min(height, width)
    if shortest < 32:
        scale = min(4.0, 32.0 / max(1, shortest))
        bgr = cv2.resize(
            bgr,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_CUBIC,
        )
    return bgr


class PaddleV6OCR:
    """Lazy CPU-only PaddleOCR 3.x backend.

    EN/ZH use PP-OCRv6 small detection + recognition in production. Japanese
    remains supported here for research probes, while MultiLangOCR routes JA to
    MangaOCR. Korean reuses the PP-OCRv6 detector with the dedicated Korean
    PP-OCRv5 mobile recognizer.
    """

    def __init__(self) -> None:
        tier = os.getenv("MANGA_PPOCRV6_TIER", "small").strip().lower()
        if tier not in {"small", "medium"}:
            tier = "small"
        self.tier = tier
        self.device = "cpu"
        # Text-line orientation adds another model/cold-start cost. Detector
        # crops are normally upright, so keep it opt-in and benchmark rotated
        # material separately before enabling it globally.
        self.textline_orientation = env_enabled(
            "MANGA_PPOCRV6_TEXTLINE_ORIENTATION", False
        )
        self._pipelines: dict[str, Any] = {}
        self._locks = {
            "unified": threading.RLock(),
            "korean": threading.RLock(),
        }
        self._creation_lock = threading.RLock()

    @property
    def unified_model_name(self) -> str:
        return f"PP-OCRv6_{self.tier}_rec"

    @property
    def detection_model_name(self) -> str:
        return f"PP-OCRv6_{self.tier}_det"

    @staticmethod
    def korean_model_name() -> str:
        return "korean_PP-OCRv5_mobile_rec"

    def read(self, image: np.ndarray, lang: str) -> OCRReadResult:
        if image is None or image.size == 0:
            return OCRReadResult("", None, "none", "unknown", 0, "reject", "empty")

        normalized = _normalize_lang(lang)
        if normalized in {"en", "ch", "ja"}:
            key = "unified"
            model_name = self.unified_model_name
        elif normalized == "korean":
            key = "korean"
            model_name = self.korean_model_name()
        else:
            raise ValueError(f"Unsupported OCR language for PaddleOCR v6 backend: {lang!r}")

        prepared = _prepare_rgb_for_paddle(image)
        pipeline = self._get_pipeline(key)
        with self._locks[key]:
            outputs = pipeline.predict(input=prepared)

        texts: list[Any] = []
        scores: list[Any] = []
        polygons: list[Any] = []
        for output in outputs:
            data = _payload(output)
            item_texts = list(data.get("rec_texts") or [])
            item_scores = list(data.get("rec_scores") or [])
            item_polygons = data.get("rec_polys")
            if item_polygons is None or len(item_polygons) == 0:
                item_polygons = data.get("dt_polys") or []
            item_polygons = list(item_polygons)

            count = min(len(item_texts), len(item_polygons))
            texts.extend(item_texts[:count])
            polygons.extend(item_polygons[:count])
            scores.extend(
                item_scores[index] if index < len(item_scores) else None
                for index in range(count)
            )

        ordered = reconstruct_reading_order(
            texts,
            scores,
            polygons,
            lang=normalized,
        )
        if ordered["regions"]:
            text = str(ordered["text"] or "").strip()
            confidence = ordered["confidence"]
            quality = classify_ocr_quality(text, normalized, confidence=confidence)
            return OCRReadResult(
                text=text,
                confidence=confidence,
                model=model_name,
                orientation=str(ordered["orientation"]),
                region_count=len(ordered["ordered_indices"]),
                quality=quality.status,
                quality_reason=quality.reason,
            )

        # Paddle can occasionally return recognition text without polygons.
        # Preserve useful text rather than dropping the result entirely.
        fallback_texts = [
            str(value or "").strip()
            for value in texts
            if str(value or "").strip()
        ]
        finite_scores: list[float] = []
        for score in scores:
            try:
                if score is not None:
                    finite_scores.append(float(score))
            except (TypeError, ValueError):
                pass
        separator = "" if normalized == "ja" else "\n"
        text = separator.join(fallback_texts).strip()
        confidence = statistics.fmean(finite_scores) if finite_scores else None
        quality = classify_ocr_quality(text, normalized, confidence=confidence)
        return OCRReadResult(
            text=text,
            confidence=confidence,
            model=model_name,
            orientation="unknown",
            region_count=len(fallback_texts),
            quality=quality.status,
            quality_reason=quality.reason,
        )

    def _get_pipeline(self, key: str) -> Any:
        existing = self._pipelines.get(key)
        if existing is not None:
            return existing
        with self._creation_lock:
            existing = self._pipelines.get(key)
            if existing is not None:
                return existing

            from paddleocr import PaddleOCR

            recognition_model = (
                self.unified_model_name
                if key == "unified"
                else self.korean_model_name()
            )
            pipeline = PaddleOCR(
                text_detection_model_name=self.detection_model_name,
                text_recognition_model_name=recognition_model,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=self.textline_orientation,
                device=self.device,
            )
            self._pipelines[key] = pipeline
            return pipeline
