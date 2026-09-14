from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import statistics
import time

import cv2
import numpy as np

from app.detector.adaptive_focus_detector import (
    _focus_text_detect,
)
from app.detector.bubble_detector import BubbleBox
from app.detector.combined_detector import CombinedTextDetector
from app.detector.sequential_fast_residue_detector import (
    SequentialFastResidueAdaptiveFocusCombinedTextDetector,
)
from app.image_io import read_image
from app.parameters import DETECTOR_INPUT_SIZE, DETECTOR_TALL_IMAGE_FACTOR
from app.pipeline import ChapterPipeline


CHAPTER_URL = "https://asurascans.com/comics/killer-pietro-08677664/chapter/120"
OUT = Path("benchmark-results/bubble-fastpath/report.json")
SAMPLE_COUNT = 16


def _single_pass_bubble_detect(detector, image: np.ndarray) -> list[BubbleBox]:
    h, w = image.shape[:2]
    boxes = detector._detect_single_plain(image, 0, 0)
    return [
        detector._with_semantics(box)
        for box in detector._filter_invalid(boxes, w, h)
    ]


class SinglePassBubbleDetector(SequentialFastResidueAdaptiveFocusCombinedTextDetector):
    """Benchmark-only proposal fastpath.

    Bubble YOLO has no destructive mask authority. On tall pages, replace the
    adaptive multi-window bubble proposal scan with one full-page proposal pass.
    The text segmenter, MSER recovery, focus scheduling, classification and final
    destructive mask authority remain unchanged.
    """

    def detect(self, image: np.ndarray, *, parallel: bool = False):
        started_at = time.perf_counter()
        h = int(image.shape[0])
        tall = h > DETECTOR_INPUT_SIZE * DETECTOR_TALL_IMAGE_FACTOR

        bubble_started = time.perf_counter()
        bubble_boxes = _single_pass_bubble_detect(self._bubble_model, image)
        bubble_ms = (time.perf_counter() - bubble_started) * 1000.0

        proposal_started = time.perf_counter()
        recovery_boxes = self.recovery.detect(image, existing=bubble_boxes)
        proposal_ms = (time.perf_counter() - proposal_started) * 1000.0

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
        metrics["bubble_model_ms"] = round(bubble_ms, 3)
        metrics["text_model_ms"] = round(text_ms, 3)
        metrics["focus_prefetch_mser_ms"] = round(proposal_ms, 3)
        metrics["focus_prefetch_proposals"] = len(recovery_boxes)
        metrics["mser_ms"] = round(float(metrics.get("mser_ms", 0.0)) + proposal_ms, 3)
        metrics.update(focus_metrics)
        metrics["bubble_fastpath_enabled"] = int(tall)
        metrics["bubble_fastpath_proposals"] = int(len(bubble_boxes))
        metrics["total_ms"] = round((time.perf_counter() - started_at) * 1000.0, 3)
        self._metrics_local.value = metrics
        return result


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
        sx1 = x1 - int(box.x1)
        sy1 = y1 - int(box.y1)
        sx2 = sx1 + (x2 - x1)
        sy2 = sy1 + (y2 - y1)
        target = mask[y1:y2, x1:x2]
        mask[y1:y2, x1:x2] = np.maximum(target, local[sy1:sy2, sx1:sx2])
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


def _run(detector, paths: dict[int, Path], indices: list[int], warmup_path: Path, label: str) -> dict:
    warmup = read_image(warmup_path)
    detector.detect(warmup, parallel=False)
    del warmup

    elapsed = []
    rows = []
    authority = {}
    review = {}
    counts = {}
    heights = {}
    for index in indices:
        image = read_image(paths[index])
        h, w = image.shape[:2]
        started = time.perf_counter()
        boxes = detector.detect(image, parallel=False)
        elapsed.append((time.perf_counter() - started) * 1000.0)
        metrics = detector.last_metrics()
        rows.append(metrics)
        authority[str(index)] = hashlib.sha256(_authority_mask(boxes, h, w).tobytes()).hexdigest()
        review[str(index)] = _review_signature(boxes)
        counts[str(index)] = {
            "boxes": len(boxes),
            "safe": sum(bool(box.safe_to_inpaint) for box in boxes),
            "review": sum(bool(box.needs_review) for box in boxes),
        }
        heights[str(index)] = int(h)
        del image, boxes

    def mean_metric(name: str) -> float:
        values = [float(row.get(name) or 0.0) for row in rows]
        return round(statistics.mean(values), 3) if values else 0.0

    return {
        "label": label,
        "wall_ms": round(sum(elapsed), 3),
        "mean_ms": round(statistics.mean(elapsed), 3),
        "median_ms": round(statistics.median(elapsed), 3),
        "bubble_model_mean_ms": mean_metric("bubble_model_ms"),
        "text_model_mean_ms": mean_metric("text_model_ms"),
        "total_metric_mean_ms": mean_metric("total_ms"),
        "focus_chip_calls": int(sum(int(row.get("focus_chip_calls") or 0) for row in rows)),
        "bubble_fastpath_pages": int(sum(bool(row.get("bubble_fastpath_enabled")) for row in rows)),
        "authority_hashes": authority,
        "review_signatures": review,
        "counts": counts,
        "heights": heights,
    }


def _pct(before: float, after: float) -> float:
    if before <= 0:
        return 0.0
    return round((1.0 - after / before) * 100.0, 2)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pipeline = ChapterPipeline()
    chapter_id = hashlib.sha256(f"bubble-fastpath-{time.time_ns()}".encode()).hexdigest()[:8]
    manifest = pipeline.download_chapter(CHAPTER_URL, chapter_id, workers=2)
    pages = manifest.get("pages", [])
    total = len(pages)
    if total < SAMPLE_COUNT + 1:
        raise RuntimeError(f"expected >= {SAMPLE_COUNT + 1} slices, got {total}")
    indices = sorted({round(i * (total - 1) / (SAMPLE_COUNT - 1)) for i in range(SAMPLE_COUNT)})
    paths = {index: Path(pages[index]["original"]) for index in indices}
    warmup_index = next(index for index in range(total) if index not in set(indices))
    warmup_path = Path(pages[warmup_index]["original"])

    control_detector = SequentialFastResidueAdaptiveFocusCombinedTextDetector()
    control = _run(control_detector, paths, indices, warmup_path, "control_adaptive_bubble")
    del control_detector
    gc.collect()

    candidate_detector = SinglePassBubbleDetector()
    candidate = _run(candidate_detector, paths, indices, warmup_path, "candidate_single_pass_bubble")
    del candidate_detector
    gc.collect()

    authority_mismatch = [
        key for key, value in control["authority_hashes"].items()
        if candidate["authority_hashes"].get(key) != value
    ]
    review_mismatch = [
        key for key, value in control["review_signatures"].items()
        if candidate["review_signatures"].get(key) != value
    ]
    count_mismatch = [
        key for key, value in control["counts"].items()
        if candidate["counts"].get(key) != value
    ]

    report = {
        "chapter_url": CHAPTER_URL,
        "chapter_id": chapter_id,
        "total_slices": total,
        "sample_indices": indices,
        "warmup_index": warmup_index,
        "control": control,
        "candidate": candidate,
        "speedup": {
            "wall_reduction_pct": _pct(control["wall_ms"], candidate["wall_ms"]),
            "bubble_model_reduction_pct": _pct(control["bubble_model_mean_ms"], candidate["bubble_model_mean_ms"]),
            "text_model_reduction_pct": _pct(control["text_model_mean_ms"], candidate["text_model_mean_ms"]),
        },
        "quality": {
            "authority_mask_mismatch_pages": authority_mismatch,
            "review_signature_mismatch_pages": review_mismatch,
            "box_count_mismatch_pages": count_mismatch,
        },
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("BUBBLE_FASTPATH_AB=" + json.dumps(report, ensure_ascii=False), flush=True)

    if authority_mismatch:
        raise RuntimeError(
            "bubble fastpath changed destructive authority masks: "
            + json.dumps(authority_mismatch, ensure_ascii=False)
        )


if __name__ == "__main__":
    main()
