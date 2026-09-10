from __future__ import annotations

from dataclasses import dataclass, replace
import os
import statistics
import threading
from typing import Any

import cv2
import numpy as np

from app.env_utils import env_enabled
from app.ocr.quality import classify_ocr_quality
from app.ocr.reading_order import reconstruct_reading_order, select_centered_target
from app.parameters import (
    OCR_COMPLETENESS_EDGE_MARGIN,
    OCR_PADDLE_MAX_UPSCALE,
    OCR_PADDLE_MIN_SIDE,
    OCR_RETRY_CONFIDENCE,
    OCR_RETRY_MAX_PIXELS,
    OCR_RETRY_UPSCALE,
    OCR_SELECTIVE_RETRY,
)

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
    coverage: float | None = None
    target_mode: str = "all"
    retry_applied: bool = False
    text_bounds: tuple[float, float, float, float] | None = None
    input_shape: tuple[int, int] | None = None


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
    if shortest < OCR_PADDLE_MIN_SIDE:
        scale = min(
            OCR_PADDLE_MAX_UPSCALE,
            OCR_PADDLE_MIN_SIDE / max(1, shortest),
        )
        bgr = cv2.resize(
            bgr,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_CUBIC,
        )
    return bgr


def _enhance_for_selective_retry(bgr: np.ndarray) -> np.ndarray:
    """Make a single inexpensive comic-font retry candidate.

    CLAHE preserves coloured glyph separation better than binary thresholding;
    a modest upscale is only applied to reasonably sized crops.  This function
    is intentionally never used for the normal first OCR pass.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    l_channel = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l_channel)
    enhanced = cv2.cvtColor(
        cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR
    )
    h, w = enhanced.shape[:2]
    if (
        OCR_RETRY_UPSCALE > 1.0
        and h * w <= OCR_RETRY_MAX_PIXELS
    ):
        enhanced = cv2.resize(
            enhanced,
            (
                max(1, int(round(w * OCR_RETRY_UPSCALE))),
                max(1, int(round(h * OCR_RETRY_UPSCALE))),
            ),
            interpolation=cv2.INTER_CUBIC,
        )
    return enhanced


def _result_rank(result: OCRReadResult) -> tuple[int, float, int]:
    quality_rank = {"reject": 0, "unknown": 0, "review": 1, "good": 2}
    confidence = result.confidence if result.confidence is not None else -1.0
    return quality_rank.get(result.quality, 0), float(confidence), len(result.text)


def _should_selective_retry(result: OCRReadResult, image: np.ndarray) -> bool:
    if not OCR_SELECTIVE_RETRY or image.size == 0:
        return False
    if image.shape[0] * image.shape[1] > OCR_RETRY_MAX_PIXELS:
        return False
    if result.quality != "good":
        return True
    return result.confidence is not None and result.confidence < OCR_RETRY_CONFIDENCE


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

    def read(
        self,
        image: np.ndarray,
        lang: str,
        *,
        target_mode: str = "all",
    ) -> OCRReadResult:
        if image is None or image.size == 0:
            return OCRReadResult("", None, "none", "unknown", 0, "reject", "empty")
        if target_mode not in {"all", "centered"}:
            raise ValueError(f"Unsupported OCR target mode: {target_mode!r}")

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
        result = self._read_once(
            prepared,
            normalized=normalized,
            key=key,
            model_name=model_name,
            target_mode=target_mode,
        )
        if _should_selective_retry(result, prepared):
            retry = self._read_once(
                _enhance_for_selective_retry(prepared),
                normalized=normalized,
                key=key,
                model_name=model_name,
                target_mode=target_mode,
            )
            if _result_rank(retry) > _result_rank(result):
                result = retry
            result = replace(result, retry_applied=True)
        return result

    def _read_once(
        self,
        prepared: np.ndarray,
        *,
        normalized: str,
        key: str,
        model_name: str,
        target_mode: str,
    ) -> OCRReadResult:
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
        original_ordered_count = len(ordered["ordered_indices"])
        if target_mode == "centered" and ordered["regions"]:
            ordered = select_centered_target(
                ordered,
                prepared.shape,
                lang=normalized,
            )
        if ordered["regions"]:
            text = str(ordered["text"] or "").strip()
            confidence = ordered["confidence"]
            selected_count = len(ordered["ordered_indices"])
            selection_coverage = (
                selected_count / max(1, original_ordered_count)
            )
            text_bounds = self._text_bounds(ordered)
            edge_truncated = self._text_touches_crop_edge(text_bounds, prepared.shape)
            quality = classify_ocr_quality(
                text,
                normalized,
                confidence=confidence,
                # Centered mode is intentionally a single-line policy.  Its
                # selection ratio is retained as metadata but does not alone
                # mark an explicit single-line target incomplete.
                coverage=selection_coverage if target_mode == "all" else None,
                may_be_truncated=edge_truncated,
            )
            return OCRReadResult(
                text=text,
                confidence=confidence,
                model=model_name,
                orientation=str(ordered["orientation"]),
                region_count=selected_count,
                quality=quality.status,
                quality_reason=quality.reason,
                coverage=selection_coverage,
                target_mode=target_mode,
                text_bounds=text_bounds,
                input_shape=tuple(int(value) for value in prepared.shape[:2]),
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
            coverage=None,
            target_mode=target_mode,
            input_shape=tuple(int(value) for value in prepared.shape[:2]),
        )

    @staticmethod
    def _text_bounds(
        ordered: dict[str, Any],
    ) -> tuple[float, float, float, float] | None:
        by_index = {int(region["index"]): region for region in ordered["regions"]}
        selected = [
            by_index[index]
            for index in ordered["ordered_indices"]
            if index in by_index
        ]
        if not selected:
            return None
        return (
            min(float(region["box"]["x1"]) for region in selected),
            min(float(region["box"]["y1"]) for region in selected),
            max(float(region["box"]["x2"]) for region in selected),
            max(float(region["box"]["y2"]) for region in selected),
        )

    @staticmethod
    def _text_touches_crop_edge(
        bounds: tuple[float, float, float, float] | None,
        image_shape: tuple[int, ...],
    ) -> bool:
        if bounds is None:
            return False
        height, width = image_shape[:2]
        x1, y1, x2, y2 = bounds
        margin = float(OCR_COMPLETENESS_EDGE_MARGIN)
        return (
            x1 <= margin
            or y1 <= margin
            or x2 >= float(width) - margin
            or y2 >= float(height) - margin
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
