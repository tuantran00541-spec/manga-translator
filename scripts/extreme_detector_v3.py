from __future__ import annotations

"""Branch-only extreme detector experiment.

This experiment removes the heavy YOLO text segmenter from the normal runtime.
PP-OCRv6 tiny provides text-line polygons and a cheap local-contrast operator
builds stroke masks. Bubble YOLO remains as semantic/context evidence. Regions
with suspicious mask density remain review-only instead of gaining erase
authority.
"""

import atexit
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.detector import combined_detector as combined_mod


_EXPERIMENT = os.getenv("MANGA_EXTREME_DETECTOR", "").strip()
_PP_MODEL = os.getenv(
    "MANGA_EXTREME_PPOCR_MODEL", "PP-OCRv6_tiny_det"
).strip() or "PP-OCRv6_tiny_det"
_PP_LIMIT = max(320, int(os.getenv("MANGA_EXTREME_PPOCR_LIMIT", "1024")))
_PP_THREADS = max(1, int(os.getenv("MANGA_EXTREME_PPOCR_THREADS", "2")))
_PP_SCORE = max(
    0.0, min(1.0, float(os.getenv("MANGA_EXTREME_PPOCR_SCORE", "0.25")))
)
_MIN_MASK_RATIO = max(
    0.001, min(0.20, float(os.getenv("MANGA_EXTREME_MASK_RATIO_MIN", "0.006")))
)
_MAX_MASK_RATIO = max(
    _MIN_MASK_RATIO,
    min(0.75, float(os.getenv("MANGA_EXTREME_MASK_RATIO_MAX", "0.42"))),
)
_REPORT_PATH = Path("benchmark-results/extreme-detector-v3.json")

_model = None
_model_create_lock = threading.RLock()
_model_predict_lock = threading.RLock()
_stats_lock = threading.Lock()
_patched = False
_original_init = None
_original_detect = None
_stats = {
    "experiment": _EXPERIMENT or None,
    "ppocr_model": _PP_MODEL,
    "ppocr_limit_side_len": _PP_LIMIT,
    "ppocr_cpu_threads": _PP_THREADS,
    "ppocr_score_min": _PP_SCORE,
    "mask_ratio_min": _MIN_MASK_RATIO,
    "mask_ratio_max": _MAX_MASK_RATIO,
    "combined_invocations": 0,
    "ppocr_runs": 0,
    "ppocr_ms": 0.0,
    "ppocr_proposals": 0,
    "ppocr_failures": 0,
    "last_ppocr_error": None,
    "stroke_mask_ms": 0.0,
    "safe_stroke_masks": 0,
    "review_stroke_regions": 0,
    "flat_bubble_fallbacks": 0,
    "bubble_model_ms": 0.0,
    "mser_ms": 0.0,
    "result_boxes": 0,
}


def _payload(result: Any) -> dict[str, Any]:
    value = getattr(result, "json", result)
    if callable(value):
        value = value()
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        value = dict(value)
    inner = value.get("res", value)
    return inner if isinstance(inner, dict) else value


def _get_model():
    global _model
    if _model is not None:
        return _model
    with _model_create_lock:
        if _model is not None:
            return _model
        from paddleocr import TextDetection

        _model = TextDetection(
            model_name=_PP_MODEL,
            device="cpu",
            enable_mkldnn=True,
            cpu_threads=_PP_THREADS,
        )
        return _model


def _predict_regions(image: np.ndarray):
    h, w = image.shape[:2]
    started = time.perf_counter()
    try:
        model = _get_model()
        with _model_predict_lock:
            outputs = model.predict(
                input=image,
                batch_size=1,
                limit_side_len=_PP_LIMIT,
                limit_type="max",
            )
        regions = []
        for output in outputs:
            data = _payload(output)
            raw_polys = data.get("dt_polys")
            raw_scores = data.get("dt_scores")
            polys = list(raw_polys) if raw_polys is not None else []
            scores = list(raw_scores) if raw_scores is not None else []
            for index, poly in enumerate(polys):
                try:
                    arr = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
                except Exception:
                    continue
                if arr.shape[0] < 4 or not np.all(np.isfinite(arr)):
                    continue
                score = 1.0
                if index < len(scores):
                    try:
                        score = float(scores[index])
                    except (TypeError, ValueError):
                        score = 1.0
                if score < _PP_SCORE:
                    continue
                x1 = max(0, min(w, int(np.floor(arr[:, 0].min()))))
                y1 = max(0, min(h, int(np.floor(arr[:, 1].min()))))
                x2 = max(0, min(w, int(np.ceil(arr[:, 0].max()))))
                y2 = max(0, min(h, int(np.ceil(arr[:, 1].max()))))
                if x2 - x1 < 4 or y2 - y1 < 4:
                    continue
                regions.append(
                    {
                        "poly": arr,
                        "score": max(0.0, min(1.0, score)),
                        "bbox": (x1, y1, x2, y2),
                    }
                )
        return regions, (time.perf_counter() - started) * 1000.0, None
    except Exception as exc:
        return [], (time.perf_counter() - started) * 1000.0, repr(exc)


def _odd_kernel(value: float) -> int:
    size = max(5, min(17, int(round(value))))
    if size % 2 == 0:
        size += 1
    return min(17, size)


def _remove_tiny_components(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return mask
    min_area = max(2, int(round(mask.size * 0.0005)))
    cleaned = np.zeros_like(mask)
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            cleaned[labels == label] = 255
    return cleaned


def _build_stroke_box(
    image: np.ndarray,
    region: dict,
    *,
    semantic_type: str,
):
    page_h, page_w = image.shape[:2]
    x1, y1, x2, y2 = region["bbox"]
    pad = 3
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(page_w, x2 + pad)
    y2 = min(page_h, y2 + pad)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None, False

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None, False

    polygon = np.zeros(crop.shape[:2], dtype=np.uint8)
    shifted = np.rint(region["poly"] - np.array([x1, y1], dtype=np.float32)).astype(np.int32)
    shifted[:, 0] = np.clip(shifted[:, 0], 0, crop.shape[1] - 1)
    shifted[:, 1] = np.clip(shifted[:, 1], 0, crop.shape[0] - 1)
    cv2.fillConvexPoly(polygon, shifted, 255)
    poly_bool = polygon > 0
    poly_count = int(np.count_nonzero(poly_bool))
    if poly_count < 16:
        return None, False

    if crop.ndim == 3:
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        light = lab[:, :, 0]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
    else:
        light = crop
        sat = np.zeros_like(light)

    kernel_size = _odd_kernel(crop.shape[0] * 0.55)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dark_signal = cv2.morphologyEx(light, cv2.MORPH_BLACKHAT, kernel)
    light_signal = cv2.morphologyEx(light, cv2.MORPH_TOPHAT, kernel)
    lum_signal = np.maximum(dark_signal, light_signal)

    blur_size = max(3, min(15, kernel_size))
    if blur_size % 2 == 0:
        blur_size += 1
    sat_bg = cv2.medianBlur(sat, blur_size)
    sat_signal = cv2.absdiff(sat, sat_bg)
    signal = np.maximum(lum_signal, sat_signal)

    values = signal[poly_bool]
    if values.size < 16:
        return None, False
    otsu_input = values.astype(np.uint8).reshape(-1, 1)
    threshold, _ = cv2.threshold(
        otsu_input, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    threshold = max(10.0, min(64.0, float(threshold)))
    strokes = ((signal >= threshold) & poly_bool).astype(np.uint8) * 255
    strokes = cv2.morphologyEx(
        strokes,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), np.uint8),
        iterations=1,
    )
    strokes = _remove_tiny_components(strokes)
    if np.any(strokes > 0):
        strokes = cv2.dilate(
            strokes,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
        strokes[~poly_bool] = 0

    stroke_count = int(np.count_nonzero(strokes > 0))
    ratio = stroke_count / float(max(1, poly_count))
    safe = bool(stroke_count > 0 and _MIN_MASK_RATIO <= ratio <= _MAX_MASK_RATIO)

    if safe:
        ys, xs = np.nonzero(strokes > 0)
        tx1 = max(0, int(xs.min()) - 2)
        ty1 = max(0, int(ys.min()) - 2)
        tx2 = min(crop.shape[1], int(xs.max()) + 3)
        ty2 = min(crop.shape[0], int(ys.max()) + 3)
        local_mask = strokes[ty1:ty2, tx1:tx2].copy()
        box = combined_mod.BubbleBox(
            x1=x1 + tx1,
            y1=y1 + ty1,
            x2=x1 + tx2,
            y2=y1 + ty2,
            confidence=float(region["score"]),
            mask=local_mask,
            source_model="ppocrv6_tiny_stroke",
            class_id=0,
            class_name="text_line",
            semantic_type=semantic_type,
            mask_source="ppocr_local_contrast",
            safe_to_inpaint=True,
            ocr_eligible=True,
            needs_review=False,
        )
        return box, True

    box = combined_mod.BubbleBox(
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        confidence=float(region["score"]),
        mask=None,
        source_model="ppocrv6_tiny_stroke",
        class_id=0,
        class_name="text_line_review",
        semantic_type=semantic_type,
        mask_source="none",
        safe_to_inpaint=False,
        ocr_eligible=True,
        needs_review=True,
    )
    return box, False


def _region_probe(region: dict):
    x1, y1, x2, y2 = region["bbox"]
    return combined_mod.BubbleBox(
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        confidence=float(region["score"]),
        semantic_type="text",
        source_model="ppocrv6_tiny_probe",
    )


def _semantic_for_region(self, region: dict, bubble_boxes) -> str:
    probe = _region_probe(region)
    candidates = [bubble for bubble in bubble_boxes if self._is_inside(probe, bubble)]
    if not candidates:
        return "free_text"
    best = max(candidates, key=lambda box: float(box.confidence))
    semantic = str(getattr(best, "semantic_type", ""))
    return semantic if semantic in {"speech_bubble", "free_text"} else "speech_bubble"


def _extreme_init(self):
    self.bubble_detector = combined_mod.YoloDetector(
        combined_mod.BUBBLE_DETECTOR_MODEL,
        combined_mod.BUBBLE_PROPOSAL_CONF_THRESHOLD,
    )
    # Deliberately do not instantiate the heavy text segmenter session.
    self.text_detector = None
    self.recovery = combined_mod.SecondaryTextRecovery()
    self._metrics_local = threading.local()


def _extreme_detect(self, image: np.ndarray, *, parallel: bool = False):
    started_at = time.perf_counter()
    h, w = image.shape[:2]

    bubble_started = time.perf_counter()
    bubble_boxes = self.bubble_detector.detect(image)
    bubble_ms = (time.perf_counter() - bubble_started) * 1000.0

    regions, ppocr_ms, ppocr_error = _predict_regions(image)
    stroke_started = time.perf_counter()
    result_boxes = []
    safe_count = 0
    review_count = 0
    matched_bubbles: set[int] = set()

    for region in regions:
        semantic = _semantic_for_region(self, region, bubble_boxes)
        box, safe = _build_stroke_box(image, region, semantic_type=semantic)
        if box is None:
            continue
        result_boxes.append(box)
        safe_count += int(safe)
        review_count += int(not safe)
        probe = _region_probe(region)
        for index, bubble in enumerate(bubble_boxes):
            if self._is_inside(probe, bubble):
                matched_bubbles.add(index)

    flat_fallbacks = 0
    for index, bubble in enumerate(bubble_boxes):
        if index in matched_bubbles:
            continue
        fallback = self._flat_bubble_text_fallback(image, bubble, w, h)
        if fallback is not None:
            result_boxes.append(fallback)
            flat_fallbacks += 1
        else:
            result_boxes.append(
                replace(
                    bubble,
                    safe_to_inpaint=False,
                    ocr_eligible=(bubble.semantic_type == "free_text"),
                    needs_review=True,
                )
            )

    stroke_ms = (time.perf_counter() - stroke_started) * 1000.0

    recovery_started = time.perf_counter()
    recovered = self.recovery.detect(image, existing=result_boxes)
    mser_ms = (time.perf_counter() - recovery_started) * 1000.0
    result_boxes.extend(recovered)
    result = self._apply_final_nms(
        result_boxes,
        iou_threshold=combined_mod.DETECTOR_FINAL_NMS_IOU,
    )

    metrics = {
        "bubble_model_ms": round(bubble_ms, 3),
        "text_model_ms": round(ppocr_ms, 3),
        "ppocr_ms": round(ppocr_ms, 3),
        "stroke_mask_ms": round(stroke_ms, 3),
        "mser_ms": round(mser_ms, 3),
        "bubble_proposals": len(bubble_boxes),
        "text_proposals": len(regions),
        "ppocr_proposals": len(regions),
        "ppocr_safe_masks": safe_count,
        "ppocr_review_regions": review_count,
        "flat_bubble_fallbacks": flat_fallbacks,
        "text_grayscale_fallback_runs": 0,
        "text_grayscale_fallback_ms": 0.0,
        "text_grayscale_fallback_proposals": 0,
        "result_boxes": len(result),
        "review_boxes": sum(bool(box.needs_review) for box in result),
        "total_ms": round((time.perf_counter() - started_at) * 1000.0, 3),
    }
    self._metrics_local.value = metrics

    with _stats_lock:
        _stats["combined_invocations"] += 1
        _stats["ppocr_runs"] += 1
        _stats["ppocr_ms"] += ppocr_ms
        _stats["ppocr_proposals"] += len(regions)
        _stats["stroke_mask_ms"] += stroke_ms
        _stats["safe_stroke_masks"] += safe_count
        _stats["review_stroke_regions"] += review_count
        _stats["flat_bubble_fallbacks"] += flat_fallbacks
        _stats["bubble_model_ms"] += bubble_ms
        _stats["mser_ms"] += mser_ms
        _stats["result_boxes"] += len(result)
        if ppocr_error:
            _stats["ppocr_failures"] += 1
            _stats["last_ppocr_error"] = ppocr_error
    return result


def _write_report() -> None:
    try:
        with _stats_lock:
            payload = json.loads(json.dumps(_stats))
        for key in (
            "ppocr_ms",
            "stroke_mask_ms",
            "bubble_model_ms",
            "mser_ms",
        ):
            payload[key] = round(float(payload.get(key, 0.0)), 3)
        _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        _REPORT_PATH.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except Exception:
        pass


def install_extreme_v3() -> bool:
    global _patched, _original_init, _original_detect
    if _patched:
        return True
    if _EXPERIMENT != "ppocr_tiny_stroke_v3":
        return False
    _original_init = combined_mod.CombinedTextDetector.__init__
    _original_detect = combined_mod.CombinedTextDetector.detect
    combined_mod.CombinedTextDetector.__init__ = _extreme_init
    combined_mod.CombinedTextDetector.detect = _extreme_detect
    atexit.register(_write_report)
    _patched = True
    return True


def _self_test() -> None:
    image = np.full((100, 240, 3), 245, dtype=np.uint8)
    cv2.putText(
        image,
        "TEST",
        (30, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    region = {
        "poly": np.array([[20, 25], [180, 25], [180, 75], [20, 75]], dtype=np.float32),
        "score": 0.9,
        "bbox": (20, 25, 180, 75),
    }
    box, safe = _build_stroke_box(image, region, semantic_type="speech_bubble")
    assert box is not None
    assert safe
    assert box.safe_to_inpaint
    assert box.verified_mask

    inverse = np.full((100, 240, 3), 15, dtype=np.uint8)
    cv2.putText(
        inverse,
        "TEST",
        (30, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )
    box2, safe2 = _build_stroke_box(inverse, region, semantic_type="speech_bubble")
    assert box2 is not None
    assert safe2
    assert box2.verified_mask


if __name__ == "__main__":
    _self_test()
    print(
        {
            "status": "pass",
            "experiment": "ppocr_tiny_stroke_v3",
            "model": _PP_MODEL,
            "limit_side_len": _PP_LIMIT,
        }
    )
