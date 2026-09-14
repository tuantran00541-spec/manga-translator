from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np

from app.detector.adaptive_focus_detector import (
    FOCUS_MAX_CHIPS,
    FOCUS_SOURCE_PIXEL_BUDGET,
    FOCUS_TENSOR_PIXEL_BUDGET,
    _adaptive_detect,
    _focus_text_detect,
    plan_focus_chips,
)
from app.detector.bubble_detector import BubbleBox
from app.detector.combined_detector import CombinedTextDetector
from app.detector.sequential_fast_residue_detector import (
    SequentialFastResidueAdaptiveFocusCombinedTextDetector,
)
from app.image_io import read_image
from app.parameters import (
    DETECTOR_FOCUS_HARD_MAX_CHIPS,
    DETECTOR_FOCUS_PROPOSALS_PER_EXTRA_CHIP,
    DETECTOR_INPUT_SIZE,
    DETECTOR_TALL_IMAGE_FACTOR,
)
from app.pipeline import ChapterPipeline
import scripts.benchmark_bubble_uncertainty as audit_base


CHAPTER_URL = "https://asurascans.com/comics/killer-pietro-08677664/chapter/120"
OUT = Path("benchmark-results/text-fullpass-router/report.json")
SAMPLE_COUNT = 16


def _focus_without_fullpass(detector, image: np.ndarray, proposals: list[BubbleBox]):
    h, w = image.shape[:2]
    threshold = DETECTOR_INPUT_SIZE * DETECTOR_TALL_IMAGE_FACTOR
    if h <= threshold:
        result, metrics, deferred = _focus_text_detect(detector, image, proposals)
        metrics = dict(metrics)
        metrics.update({
            "text_router_skip_fullpass": 0,
            "text_router_keep_fullpass": 1,
            "text_fullpass_calls": 1,
        })
        return result, metrics, deferred

    uncovered = list(proposals)
    fallback_image = image if not proposals else None
    extra = max(0, len(uncovered) - 1) // DETECTOR_FOCUS_PROPOSALS_PER_EXTRA_CHIP
    adaptive_max_chips = min(
        DETECTOR_FOCUS_HARD_MAX_CHIPS,
        FOCUS_MAX_CHIPS + extra,
    )
    scale = adaptive_max_chips / float(max(1, FOCUS_MAX_CHIPS))
    chips, deferred = plan_focus_chips(
        h,
        w,
        uncovered,
        max_chips=adaptive_max_chips,
        source_pixel_budget=int(round(FOCUS_SOURCE_PIXEL_BUDGET * scale)),
        tensor_pixel_budget=int(round(FOCUS_TENSOR_PIXEL_BUDGET * scale)),
        fallback_image=fallback_image,
    )

    all_boxes: list[BubbleBox] = []
    for x1, y1, x2, y2 in chips:
        crop = image[y1:y2, x1:x2]
        if crop.size:
            all_boxes.extend(detector._detect_single_plain(crop, x1, y1))

    boxes = detector._nms_boxes(all_boxes)
    result = [
        detector._with_semantics(box)
        for box in detector._filter_invalid(boxes, w, h)
    ]
    deferred_boxes = [
        BubbleBox(
            x1, y1, x2, y2, 0.0, None,
            source_model="adaptive_scheduler",
            class_name="focus_deferred",
            semantic_type="review_region",
            mask_source="none",
            safe_to_inpaint=False,
            ocr_eligible=False,
            needs_review=True,
            source_role="scheduler",
            deferred_reason="focus_budget_exhausted",
        )
        for x1, y1, x2, y2 in deferred
    ]
    return result, {
        "focus_proposals": len(proposals),
        "focus_uncovered_proposals": len(uncovered),
        "focus_chip_calls": len(chips),
        "focus_source_pixels": sum((x2-x1)*(y2-y1) for x1,y1,x2,y2 in chips),
        "focus_tensor_pixels": len(chips) * DETECTOR_INPUT_SIZE * DETECTOR_INPUT_SIZE,
        "focus_deferred_regions": len(deferred_boxes),
        "focus_fallback_calls": int(bool(fallback_image is not None and chips)),
        "text_router_skip_fullpass": 1,
        "text_router_keep_fullpass": 0,
        "text_fullpass_calls": 0,
    }, deferred_boxes


def _pre_features(image: np.ndarray, bubbles: list[BubbleBox], recovery: list[BubbleBox]) -> dict[str, float]:
    proposals = list(bubbles) + list(recovery)
    features = audit_base._pre_features(image, proposals)
    h, w = image.shape[:2]
    features.update({
        "bubble_count": float(len(bubbles)),
        "mser_count": float(len(recovery)),
        "proposal_count": float(len(proposals)),
        "bubble_area_ratio": float(sum(
            max(0, b.x2-b.x1) * max(0, b.y2-b.y1) for b in bubbles
        )) / float(max(1, h*w)),
        "mser_area_ratio": float(sum(
            max(0, b.x2-b.x1) * max(0, b.y2-b.y1) for b in recovery
        )) / float(max(1, h*w)),
    })
    return features


def _finish(detector, image, bubbles, recovery, text_boxes, focus_metrics, deferred, started, bubble_ms, mser_ms, audit_ms):
    with detector.bubble_detector.prefetched(image, bubbles):
        with detector.text_detector.prefetched(image, text_boxes):
            result = CombinedTextDetector.detect(detector, image, parallel=False)
    if deferred:
        result.extend(deferred)
    metrics = dict(getattr(detector._metrics_local, "value", {}) or {})
    metrics.update(focus_metrics)
    metrics.update({
        "bubble_model_ms": round(bubble_ms, 3),
        "focus_prefetch_mser_ms": round(mser_ms, 3),
        "text_router_audit_ms": round(audit_ms, 3),
        "total_ms": round((time.perf_counter() - started) * 1000.0, 3),
    })
    detector._metrics_local.value = metrics
    return result


class ProbeDetector(SequentialFastResidueAdaptiveFocusCombinedTextDetector):
    def detect(self, image: np.ndarray, *, parallel: bool = False):
        if image.shape[0] <= DETECTOR_INPUT_SIZE * DETECTOR_TALL_IMAGE_FACTOR:
            return super().detect(image, parallel=False)
        started = time.perf_counter()
        t = time.perf_counter(); bubbles = _adaptive_detect(self._bubble_model, image); bubble_ms = (time.perf_counter()-t)*1000.0
        t = time.perf_counter(); recovery = self.recovery.detect(image, existing=bubbles); mser_ms = (time.perf_counter()-t)*1000.0
        t = time.perf_counter(); features = _pre_features(image, bubbles, recovery); audit_ms = (time.perf_counter()-t)*1000.0
        t = time.perf_counter(); text_boxes, focus_metrics, deferred = _focus_without_fullpass(self._text_model, image, list(bubbles)+list(recovery)); text_ms = (time.perf_counter()-t)*1000.0
        focus_metrics = dict(focus_metrics); focus_metrics["text_model_ms"] = round(text_ms, 3)
        result = _finish(self, image, bubbles, recovery, text_boxes, focus_metrics, deferred, started, bubble_ms, mser_ms, audit_ms)
        metrics = dict(getattr(self._metrics_local, "value", {}) or {})
        metrics.update({f"router_feature_{k}": float(v) for k, v in features.items()})
        self._metrics_local.value = metrics
        return result


class GatedDetector(SequentialFastResidueAdaptiveFocusCombinedTextDetector):
    def __init__(self, rule: dict):
        super().__init__()
        self.rule = rule

    def detect(self, image: np.ndarray, *, parallel: bool = False):
        if image.shape[0] <= DETECTOR_INPUT_SIZE * DETECTOR_TALL_IMAGE_FACTOR:
            return super().detect(image, parallel=False)
        started = time.perf_counter()
        t = time.perf_counter(); bubbles = _adaptive_detect(self._bubble_model, image); bubble_ms = (time.perf_counter()-t)*1000.0
        t = time.perf_counter(); recovery = self.recovery.detect(image, existing=bubbles); mser_ms = (time.perf_counter()-t)*1000.0
        t = time.perf_counter(); features = _pre_features(image, bubbles, recovery); audit_ms = (time.perf_counter()-t)*1000.0
        skip = audit_base._rule_fast(features, self.rule)
        t = time.perf_counter()
        if skip:
            text_boxes, focus_metrics, deferred = _focus_without_fullpass(self._text_model, image, list(bubbles)+list(recovery))
        else:
            text_boxes, focus_metrics, deferred = _focus_text_detect(self._text_model, image, list(bubbles)+list(recovery))
            focus_metrics = dict(focus_metrics)
            focus_metrics.update({
                "text_router_skip_fullpass": 0,
                "text_router_keep_fullpass": 1,
                "text_fullpass_calls": 1,
            })
        text_ms = (time.perf_counter()-t)*1000.0
        focus_metrics = dict(focus_metrics); focus_metrics["text_model_ms"] = round(text_ms, 3)
        result = _finish(self, image, bubbles, recovery, text_boxes, focus_metrics, deferred, started, bubble_ms, mser_ms, audit_ms)
        metrics = dict(getattr(self._metrics_local, "value", {}) or {})
        metrics.update({f"router_feature_{k}": float(v) for k, v in features.items()})
        self._metrics_local.value = metrics
        return result


def _run(detector, paths: dict[int, Path], indices: list[int], warmup_path: Path, label: str) -> dict:
    warm = read_image(warmup_path); detector.detect(warm, parallel=False); del warm
    elapsed = []
    rows = {}
    authority = {}
    review = {}
    counts = {}
    for index in indices:
        image = read_image(paths[index]); h, w = image.shape[:2]
        t = time.perf_counter(); boxes = detector.detect(image, parallel=False); elapsed.append((time.perf_counter()-t)*1000.0)
        raw_metrics = dict(getattr(detector._metrics_local, "value", {}) or {})
        rows[str(index)] = raw_metrics
        authority[str(index)] = hashlib.sha256(audit_base._authority_mask(boxes, h, w).tobytes()).hexdigest()
        review[str(index)] = audit_base._review_signature(boxes)
        counts[str(index)] = {
            "boxes": len(boxes),
            "safe": sum(bool(b.safe_to_inpaint) for b in boxes),
            "review": sum(bool(b.needs_review) for b in boxes),
        }
        del image, boxes

    def mean_metric(name: str) -> float:
        values = [float(row.get(name) or 0.0) for row in rows.values()]
        return round(statistics.mean(values), 3) if values else 0.0

    return {
        "label": label,
        "wall_ms": round(sum(elapsed), 3),
        "mean_ms": round(statistics.mean(elapsed), 3),
        "median_ms": round(statistics.median(elapsed), 3),
        "text_model_mean_ms": mean_metric("text_model_ms"),
        "focus_chip_calls": int(sum(int(row.get("focus_chip_calls") or 0) for row in rows.values())),
        "text_fullpass_calls": int(sum(int(row.get("text_fullpass_calls") or 0) for row in rows.values())),
        "skip_fullpass_pages": int(sum(int(row.get("text_router_skip_fullpass") or 0) for row in rows.values())),
        "keep_fullpass_pages": int(sum(int(row.get("text_router_keep_fullpass") or 0) for row in rows.values())),
        "rows": rows,
        "authority_hashes": authority,
        "review_signatures": review,
        "counts": counts,
    }


def _mismatch(control: dict, candidate: dict) -> dict:
    return {
        "authority": [k for k,v in control["authority_hashes"].items() if candidate["authority_hashes"].get(k) != v],
        "review": [k for k,v in control["review_signatures"].items() if candidate["review_signatures"].get(k) != v],
        "counts": [k for k,v in control["counts"].items() if candidate["counts"].get(k) != v],
    }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pipeline = ChapterPipeline()
    chapter_id = hashlib.sha256(f"text-router-{time.time_ns()}".encode()).hexdigest()[:8]
    manifest = pipeline.download_chapter(CHAPTER_URL, chapter_id, workers=2)
    pages = manifest.get("pages", [])
    total = len(pages)
    indices = sorted({round(i*(total-1)/(SAMPLE_COUNT-1)) for i in range(SAMPLE_COUNT)})
    paths = {i: Path(pages[i]["original"]) for i in indices}
    warmup_index = next(i for i in range(total) if i not in set(indices))
    warmup_path = Path(pages[warmup_index]["original"])

    control = _run(SequentialFastResidueAdaptiveFocusCombinedTextDetector(), paths, indices, warmup_path, "control")
    gc.collect()
    probe = _run(ProbeDetector(), paths, indices, warmup_path, "probe_skip_all_tall_fullpasses")
    gc.collect()

    probe_quality = _mismatch(control, probe)
    unsafe = set().union(*(set(v) for v in probe_quality.values()))
    features = {
        page: {
            name[len("router_feature_"):]: float(value)
            for name, value in row.items()
            if name.startswith("router_feature_")
        }
        for page, row in probe["rows"].items()
        if any(name.startswith("router_feature_") for name in row)
    }
    rule = audit_base._select_safe_rule(features, unsafe)

    candidate = _run(GatedDetector(rule), paths, indices, warmup_path, "candidate_gated_fullpass_router")
    quality = _mismatch(control, candidate)
    report = {
        "chapter_url": CHAPTER_URL,
        "chapter_id": chapter_id,
        "total_slices": total,
        "sample_indices": indices,
        "control": control,
        "probe": probe,
        "probe_quality": probe_quality,
        "audit": {"rule": rule, "features": features},
        "candidate": candidate,
        "candidate_quality": quality,
        "speedup": {
            "probe_wall_reduction_pct": audit_base._pct(control["wall_ms"], probe["wall_ms"]),
            "candidate_wall_reduction_pct": audit_base._pct(control["wall_ms"], candidate["wall_ms"]),
            "candidate_text_model_reduction_pct": audit_base._pct(control["text_model_mean_ms"], candidate["text_model_mean_ms"]),
            "candidate_skip_fullpass_pages": candidate["skip_fullpass_pages"],
            "candidate_focus_chip_calls": candidate["focus_chip_calls"],
        },
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("TEXT_FULLPASS_ROUTER=" + json.dumps(report, ensure_ascii=False), flush=True)

    if quality["authority"]:
        raise RuntimeError("text full-pass router changed destructive authority masks: " + json.dumps(quality["authority"]))


if __name__ == "__main__":
    main()
