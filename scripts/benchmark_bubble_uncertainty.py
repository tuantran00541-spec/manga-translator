from __future__ import annotations

import gc
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import time

import cv2
import numpy as np

from app.detector.adaptive_focus_detector import _adaptive_detect, _focus_text_detect
from app.detector.bubble_detector import BubbleBox
from app.detector.combined_detector import CombinedTextDetector
from app.detector.sequential_fast_residue_detector import (
    SequentialFastResidueAdaptiveFocusCombinedTextDetector,
)
from app.image_io import read_image
from app.parameters import DETECTOR_INPUT_SIZE, DETECTOR_TALL_IMAGE_FACTOR
from app.pipeline import ChapterPipeline


CHAPTER_URL = "https://asurascans.com/comics/killer-pietro-08677664/chapter/120"
OUT = Path("benchmark-results/bubble-uncertainty/report.json")
SAMPLE_COUNT = 16
AUDIT_BANDS = 8
AUDIT_WIDTH = 256
GABOR_KERNEL = 9
GABOR_SIGMA = 2.5
GABOR_LAMBDA = 5.0
GABOR_GAMMA = 0.5


def _single_pass_bubble_detect(detector, image: np.ndarray) -> list[BubbleBox]:
    h, w = image.shape[:2]
    boxes = detector._detect_single_plain(image, 0, 0)
    return [
        detector._with_semantics(box)
        for box in detector._filter_invalid(boxes, w, h)
    ]


def _overlap_y(box: BubbleBox, y1: int, y2: int) -> int:
    return max(0, min(int(box.y2), int(y2)) - max(int(box.y1), int(y1)))


def _gabor_band_scores(image: np.ndarray, bands: int = AUDIT_BANDS) -> list[float]:
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        return [0.0] * bands
    scale = min(1.0, AUDIT_WIDTH / float(max(1, w)))
    rw = max(1, int(round(w * scale)))
    rh = max(1, int(round(h * scale)))
    small = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_AREA) if (rw, rh) != (w, h) else image
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
    gray = gray.astype(np.float32) / 255.0
    response = np.zeros_like(gray, dtype=np.float32)
    for theta in (0.0, math.pi / 4.0, math.pi / 2.0, 3.0 * math.pi / 4.0):
        kernel = cv2.getGaborKernel(
            (GABOR_KERNEL, GABOR_KERNEL),
            GABOR_SIGMA,
            theta,
            GABOR_LAMBDA,
            GABOR_GAMMA,
            0,
            ktype=cv2.CV_32F,
        )
        filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
        response = np.maximum(response, np.abs(filtered))
    edges = cv2.Canny((gray * 255.0).astype(np.uint8), 60, 160) > 0
    scores: list[float] = []
    for index in range(bands):
        y1 = int(round(index * rh / bands))
        y2 = int(round((index + 1) * rh / bands))
        if y2 <= y1:
            scores.append(0.0)
            continue
        band_response = response[y1:y2]
        band_edges = edges[y1:y2]
        score = float(np.mean(band_response)) + 0.20 * float(np.mean(band_edges))
        scores.append(score)
    return scores


def _audit_features(
    image: np.ndarray,
    bubble_boxes: list[BubbleBox],
    recovery_boxes: list[BubbleBox],
) -> dict[str, float | int | list[float]]:
    h, _w = image.shape[:2]
    proposals = list(bubble_boxes) + list(recovery_boxes)
    scores = _gabor_band_scores(image)
    score_array = np.asarray(scores, dtype=np.float32)
    median = float(np.median(score_array)) if score_array.size else 0.0
    mad = float(np.median(np.abs(score_array - median))) if score_array.size else 0.0
    active_floor = median + max(0.0025, 1.25 * mad)

    uncovered_active = 0
    uncovered_peak_ratio = 0.0
    uncovered_scores: list[float] = []
    for index, score in enumerate(scores):
        y1 = int(round(index * h / AUDIT_BANDS))
        y2 = int(round((index + 1) * h / AUDIT_BANDS))
        covered = any(_overlap_y(box, y1, y2) > 0 for box in proposals)
        if not covered:
            uncovered_scores.append(float(score))
            if score >= active_floor:
                uncovered_active += 1
    if uncovered_scores:
        uncovered_peak_ratio = max(uncovered_scores) / max(1e-6, median)

    centers = sorted(
        max(0.0, min(1.0, ((float(box.y1) + float(box.y2)) * 0.5) / max(1.0, float(h))))
        for box in proposals
    )
    boundaries = [0.0] + centers + [1.0]
    max_gap = max(
        (boundaries[i + 1] - boundaries[i] for i in range(len(boundaries) - 1)),
        default=1.0,
    )
    confidence = [float(box.confidence) for box in bubble_boxes]
    border_bubbles = sum(
        int(box.y1) <= max(8, int(round(h * 0.03)))
        or int(box.y2) >= h - max(8, int(round(h * 0.03)))
        for box in bubble_boxes
    )
    return {
        "bubble_count": int(len(bubble_boxes)),
        "recovery_count": int(len(recovery_boxes)),
        "proposal_count": int(len(proposals)),
        "bubble_conf_mean": round(float(statistics.mean(confidence)), 6) if confidence else 0.0,
        "bubble_conf_min": round(float(min(confidence)), 6) if confidence else 0.0,
        "max_proposal_gap_ratio": round(float(max_gap), 6),
        "uncovered_gabor_bands": int(uncovered_active),
        "uncovered_gabor_peak_ratio": round(float(uncovered_peak_ratio), 6),
        "gabor_median": round(float(median), 6),
        "gabor_mad": round(float(mad), 6),
        "border_bubbles": int(border_bubbles),
        "gabor_band_scores": [round(float(v), 6) for v in scores],
    }


def _authority_mask(boxes: list[BubbleBox], h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    for box in boxes:
        if not bool(box.safe_to_inpaint) or not box.verified_mask:
            continue
        x1 = max(0, min(w, int(box.x1)))
        y1 = max(0, min(h, int(box.y1)))
        x2 = max(x1, min(w, int(box.x2)))
        y2 = max(y1, min(h, int(box.y2)))
        if x2 <= x1 or y2 <= y1:
            continue
        local = box.mask
        expected = (int(box.y2 - box.y1), int(box.x2 - box.x1))
        if local.shape != expected:
            local = cv2.resize(local, (expected[1], expected[0]), interpolation=cv2.INTER_NEAREST)
        sx1, sy1 = x1 - int(box.x1), y1 - int(box.y1)
        sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)
        mask[y1:y2, x1:x2] = np.maximum(
            mask[y1:y2, x1:x2],
            local[sy1:sy2, sx1:sx2],
        )
    return mask


def _review_signature(boxes: list[BubbleBox]) -> list[tuple]:
    return sorted(
        [
            (
                int(box.x1), int(box.y1), int(box.x2), int(box.y2),
                str(box.semantic_type), str(box.source_model), box.deferred_reason,
            )
            for box in boxes if bool(box.needs_review)
        ],
        key=repr,
    )


class ProbeSinglePassBubbleDetector(SequentialFastResidueAdaptiveFocusCombinedTextDetector):
    def detect(self, image: np.ndarray, *, parallel: bool = False):
        started_at = time.perf_counter()
        bubble_started = time.perf_counter()
        bubble_boxes = _single_pass_bubble_detect(self._bubble_model, image)
        bubble_ms = (time.perf_counter() - bubble_started) * 1000.0

        proposal_started = time.perf_counter()
        recovery_boxes = self.recovery.detect(image, existing=bubble_boxes)
        proposal_ms = (time.perf_counter() - proposal_started) * 1000.0
        audit_started = time.perf_counter()
        audit = _audit_features(image, bubble_boxes, recovery_boxes)
        audit_ms = (time.perf_counter() - audit_started) * 1000.0

        text_started = time.perf_counter()
        text_boxes, focus_metrics, deferred_boxes = _focus_text_detect(
            self._text_model,
            image,
            list(bubble_boxes) + list(recovery_boxes),
        )
        text_ms = (time.perf_counter() - text_started) * 1000.0
        with self.bubble_detector.prefetched(image, bubble_boxes):
            with self.text_detector.prefetched(image, text_boxes):
                result = CombinedTextDetector.detect(self, image, parallel=False)
        if deferred_boxes:
            result.extend(deferred_boxes)

        metrics = dict(getattr(self._metrics_local, "value", {}) or {})
        metrics.update(focus_metrics)
        metrics.update({f"audit_{k}": v for k, v in audit.items() if not isinstance(v, list)})
        metrics["bubble_model_ms"] = round(bubble_ms, 3)
        metrics["text_model_ms"] = round(text_ms, 3)
        metrics["focus_prefetch_mser_ms"] = round(proposal_ms, 3)
        metrics["audit_ms"] = round(audit_ms, 3)
        metrics["bubble_fastpath_enabled"] = int(
            image.shape[0] > DETECTOR_INPUT_SIZE * DETECTOR_TALL_IMAGE_FACTOR
        )
        metrics["total_ms"] = round((time.perf_counter() - started_at) * 1000.0, 3)
        metrics["audit_features"] = audit
        self._metrics_local.value = metrics
        return result


def _predicate_candidates(rows: dict[str, dict]) -> list[dict]:
    specs = [
        ("recovery_count", ">="),
        ("uncovered_gabor_bands", ">="),
        ("uncovered_gabor_peak_ratio", ">="),
        ("max_proposal_gap_ratio", ">="),
        ("border_bubbles", ">="),
        ("bubble_count", "<="),
        ("proposal_count", "<="),
        ("bubble_conf_mean", "<="),
    ]
    candidates: list[dict] = []
    for feature, op in specs:
        values = sorted({float(row[feature]) for row in rows.values()})
        if not values:
            continue
        thresholds = values
        if len(values) > 1:
            thresholds = sorted(set(values + [(a + b) * 0.5 for a, b in zip(values, values[1:])]))
        for threshold in thresholds:
            candidates.append({"feature": feature, "op": op, "threshold": float(threshold)})
    return candidates


def _matches(row: dict, predicate: dict) -> bool:
    value = float(row[predicate["feature"]])
    threshold = float(predicate["threshold"])
    return value >= threshold if predicate["op"] == ">=" else value <= threshold


def _select_rule(features: dict[str, dict], unsafe_pages: set[str]) -> dict:
    if not unsafe_pages:
        return {"predicates": [], "fallback_pages": [], "unsafe_pages": []}
    predicates = _predicate_candidates(features)
    best = None
    page_keys = sorted(features, key=lambda x: int(x))
    options = [[p] for p in predicates]
    options.extend([list(pair) for pair in itertools.combinations(predicates, 2)])
    for rule in options:
        fallback = {
            key for key in page_keys
            if any(_matches(features[key], predicate) for predicate in rule)
        }
        if not unsafe_pages.issubset(fallback):
            continue
        false_positives = len(fallback - unsafe_pages)
        score = (len(fallback), false_positives, len(rule))
        if best is None or score < best[0]:
            best = (score, rule, fallback)
    if best is None:
        return {
            "predicates": [{"feature": "proposal_count", "op": ">=", "threshold": 0.0}],
            "fallback_pages": page_keys,
            "unsafe_pages": sorted(unsafe_pages, key=int),
        }
    return {
        "predicates": best[1],
        "fallback_pages": sorted(best[2], key=int),
        "unsafe_pages": sorted(unsafe_pages, key=int),
    }


class GatedBubbleDetector(SequentialFastResidueAdaptiveFocusCombinedTextDetector):
    def __init__(self, rule: dict):
        super().__init__()
        self.rule = dict(rule)

    def _fallback(self, audit: dict) -> bool:
        return any(_matches(audit, predicate) for predicate in self.rule.get("predicates", []))

    def detect(self, image: np.ndarray, *, parallel: bool = False):
        started_at = time.perf_counter()
        h = int(image.shape[0])
        tall = h > DETECTOR_INPUT_SIZE * DETECTOR_TALL_IMAGE_FACTOR

        bubble_started = time.perf_counter()
        fast_bubbles = _single_pass_bubble_detect(self._bubble_model, image)
        bubble_ms_fast = (time.perf_counter() - bubble_started) * 1000.0

        proposal_started = time.perf_counter()
        fast_recovery = self.recovery.detect(image, existing=fast_bubbles)
        proposal_ms = (time.perf_counter() - proposal_started) * 1000.0
        audit_started = time.perf_counter()
        audit = _audit_features(image, fast_bubbles, fast_recovery)
        audit_ms = (time.perf_counter() - audit_started) * 1000.0

        fallback = bool(tall and self._fallback(audit))
        fallback_ms = 0.0
        if fallback:
            fallback_started = time.perf_counter()
            bubble_boxes = _adaptive_detect(self._bubble_model, image)
            recovery_boxes = self.recovery.detect(image, existing=bubble_boxes)
            fallback_ms = (time.perf_counter() - fallback_started) * 1000.0
        else:
            bubble_boxes = fast_bubbles
            recovery_boxes = fast_recovery

        text_started = time.perf_counter()
        text_boxes, focus_metrics, deferred_boxes = _focus_text_detect(
            self._text_model,
            image,
            list(bubble_boxes) + list(recovery_boxes),
        )
        text_ms = (time.perf_counter() - text_started) * 1000.0
        with self.bubble_detector.prefetched(image, bubble_boxes):
            with self.text_detector.prefetched(image, text_boxes):
                result = CombinedTextDetector.detect(self, image, parallel=False)
        if deferred_boxes:
            result.extend(deferred_boxes)

        metrics = dict(getattr(self._metrics_local, "value", {}) or {})
        metrics.update(focus_metrics)
        metrics["bubble_model_ms"] = round(bubble_ms_fast + fallback_ms, 3)
        metrics["bubble_fast_probe_ms"] = round(bubble_ms_fast, 3)
        metrics["bubble_fallback_ms"] = round(fallback_ms, 3)
        metrics["text_model_ms"] = round(text_ms, 3)
        metrics["focus_prefetch_mser_ms"] = round(proposal_ms, 3)
        metrics["audit_ms"] = round(audit_ms, 3)
        metrics["bubble_gate_fallback"] = int(fallback)
        metrics["bubble_gate_fastpath"] = int(tall and not fallback)
        metrics["audit_features"] = audit
        metrics["total_ms"] = round((time.perf_counter() - started_at) * 1000.0, 3)
        self._metrics_local.value = metrics
        return result


def _run(detector, paths: dict[int, Path], indices: list[int], warmup_path: Path, label: str) -> dict:
    warmup = read_image(warmup_path)
    detector.detect(warmup, parallel=False)
    del warmup
    elapsed: list[float] = []
    rows: dict[str, dict] = {}
    authority: dict[str, str] = {}
    review: dict[str, list] = {}
    counts: dict[str, dict] = {}
    for index in indices:
        image = read_image(paths[index])
        h, w = image.shape[:2]
        started = time.perf_counter()
        boxes = detector.detect(image, parallel=False)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        elapsed.append(elapsed_ms)
        metrics = detector.last_metrics()
        rows[str(index)] = dict(metrics)
        authority[str(index)] = hashlib.sha256(_authority_mask(boxes, h, w).tobytes()).hexdigest()
        review[str(index)] = _review_signature(boxes)
        counts[str(index)] = {
            "boxes": len(boxes),
            "safe": sum(bool(box.safe_to_inpaint) for box in boxes),
            "review": sum(bool(box.needs_review) for box in boxes),
        }
        del image, boxes
    return {
        "label": label,
        "wall_ms": round(sum(elapsed), 3),
        "mean_ms": round(statistics.mean(elapsed), 3),
        "median_ms": round(statistics.median(elapsed), 3),
        "rows": rows,
        "authority_hashes": authority,
        "review_signatures": review,
        "counts": counts,
    }


def _mismatch(control: dict, candidate: dict) -> dict[str, list[str]]:
    return {
        "authority": [
            key for key, value in control["authority_hashes"].items()
            if candidate["authority_hashes"].get(key) != value
        ],
        "review": [
            key for key, value in control["review_signatures"].items()
            if candidate["review_signatures"].get(key) != value
        ],
        "counts": [
            key for key, value in control["counts"].items()
            if candidate["counts"].get(key) != value
        ],
    }


def _pct(before: float, after: float) -> float:
    return round((1.0 - after / max(1.0, before)) * 100.0, 2)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pipeline = ChapterPipeline()
    chapter_id = hashlib.sha256(f"bubble-uncertainty-{time.time_ns()}".encode()).hexdigest()[:8]
    manifest = pipeline.download_chapter(CHAPTER_URL, chapter_id, workers=2)
    pages = manifest.get("pages", [])
    total = len(pages)
    indices = sorted({round(i * (total - 1) / (SAMPLE_COUNT - 1)) for i in range(SAMPLE_COUNT)})
    paths = {index: Path(pages[index]["original"]) for index in indices}
    warmup_index = next(index for index in range(total) if index not in set(indices))
    warmup_path = Path(pages[warmup_index]["original"])

    control_detector = SequentialFastResidueAdaptiveFocusCombinedTextDetector()
    control = _run(control_detector, paths, indices, warmup_path, "control_adaptive")
    del control_detector
    gc.collect()

    probe_detector = ProbeSinglePassBubbleDetector()
    probe = _run(probe_detector, paths, indices, warmup_path, "probe_single_pass_with_audit")
    del probe_detector
    gc.collect()

    probe_quality = _mismatch(control, probe)
    strict_unsafe = set(probe_quality["authority"]) | set(probe_quality["review"]) | set(probe_quality["counts"])
    features = {
        key: dict(row.get("audit_features") or {})
        for key, row in probe["rows"].items()
    }
    rule = _select_rule(features, strict_unsafe)

    gated_detector = GatedBubbleDetector(rule)
    gated = _run(gated_detector, paths, indices, warmup_path, "candidate_uncertainty_gated")
    del gated_detector
    gc.collect()
    gated_quality = _mismatch(control, gated)

    report = {
        "chapter_url": CHAPTER_URL,
        "chapter_id": chapter_id,
        "total_slices": total,
        "sample_indices": indices,
        "warmup_index": warmup_index,
        "audit": {
            "bands": AUDIT_BANDS,
            "width": AUDIT_WIDTH,
            "gabor_kernel": GABOR_KERNEL,
            "rule": rule,
            "features": features,
        },
        "control": control,
        "probe": probe,
        "probe_quality": probe_quality,
        "candidate": gated,
        "candidate_quality": gated_quality,
        "speedup": {
            "probe_wall_reduction_pct": _pct(control["wall_ms"], probe["wall_ms"]),
            "gated_wall_reduction_pct": _pct(control["wall_ms"], gated["wall_ms"]),
            "gated_fastpath_pages": sum(
                int(row.get("bubble_gate_fastpath") or 0)
                for row in gated["rows"].values()
            ),
            "gated_fallback_pages": sum(
                int(row.get("bubble_gate_fallback") or 0)
                for row in gated["rows"].values()
            ),
            "audit_mean_ms": round(statistics.mean(
                float(row.get("audit_ms") or 0.0) for row in gated["rows"].values()
            ), 3),
        },
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("BUBBLE_UNCERTAINTY_AB=" + json.dumps(report, ensure_ascii=False), flush=True)

    strict_mismatch = set(gated_quality["authority"]) | set(gated_quality["review"]) | set(gated_quality["counts"])
    if strict_mismatch:
        raise RuntimeError(
            "uncertainty gate failed strict output equality: "
            + json.dumps(sorted(strict_mismatch, key=int))
        )


if __name__ == "__main__":
    main()
