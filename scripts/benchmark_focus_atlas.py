from __future__ import annotations

from dataclasses import replace
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import cv2
import numpy as np

from app.detector.adaptive_focus_detector import (
    FOCUS_MAX_CHIPS,
    FOCUS_SOURCE_PIXEL_BUDGET,
    FOCUS_TENSOR_PIXEL_BUDGET,
    _adaptive_detect,
    _proposal_has_full_text,
    plan_focus_chips,
)
from app.detector.bubble_detector import BubbleBox, LetterboxTransform
from app.detector.combined_detector import CombinedTextDetector
from app.detector.sequential_fast_residue_detector import (
    SequentialFastResidueAdaptiveFocusCombinedTextDetector,
)
from app.image_io import read_image
from app.parameters import (
    DETECTOR_FOCUS_HARD_MAX_CHIPS,
    DETECTOR_FOCUS_PROPOSALS_PER_EXTRA_CHIP,
    DETECTOR_INPUT_SIZE,
    DETECTOR_LETTERBOX_VALUE,
    DETECTOR_TALL_IMAGE_FACTOR,
)
from app.pipeline import ChapterPipeline


CHAPTER_URL = "https://asurascans.com/comics/killer-pietro-08677664/chapter/120"
OUT = Path("benchmark-results/focus-atlas/report.json")
SAMPLE_COUNT = 16
ATLAS_GUARD = 48


def _slot_for_chip(chip: tuple[int, int, int, int]) -> dict:
    x1, y1, x2, y2 = (int(v) for v in chip)
    w, h = max(1, x2 - x1), max(1, y2 - y1)
    transform = LetterboxTransform.create(w, h, DETECTOR_INPUT_SIZE, DETECTOR_INPUT_SIZE)
    return {
        "chip": (x1, y1, x2, y2),
        "resized_w": int(transform.resized_w),
        "resized_h": int(transform.resized_h),
        "scale_x": float(transform.scale_x),
        "scale_y": float(transform.scale_y),
    }


def _pair_layout(a: dict, b: dict, guard: int = ATLAS_GUARD):
    aw, ah = int(a["resized_w"]), int(a["resized_h"])
    bw, bh = int(b["resized_w"]), int(b["resized_h"])
    outer_a = (aw + 2 * guard, ah + 2 * guard)
    outer_b = (bw + 2 * guard, bh + 2 * guard)

    if max(outer_a[0], outer_b[0]) <= DETECTOR_INPUT_SIZE and outer_a[1] + outer_b[1] <= DETECTOR_INPUT_SIZE:
        return [
            (guard, guard),
            (guard, outer_a[1] + guard),
        ]
    if outer_a[0] + outer_b[0] <= DETECTOR_INPUT_SIZE and max(outer_a[1], outer_b[1]) <= DETECTOR_INPUT_SIZE:
        return [
            (guard, guard),
            (outer_a[0] + guard, guard),
        ]
    return None


def _build_pair_atlas(
    image: np.ndarray,
    chip_a: tuple[int, int, int, int],
    chip_b: tuple[int, int, int, int],
):
    a, b = _slot_for_chip(chip_a), _slot_for_chip(chip_b)
    layout = _pair_layout(a, b)
    if layout is None:
        return None, None
    atlas = np.full(
        (DETECTOR_INPUT_SIZE, DETECTOR_INPUT_SIZE, 3),
        DETECTOR_LETTERBOX_VALUE,
        dtype=np.uint8,
    )
    slots = []
    for slot, origin in zip((a, b), layout):
        sx1, sy1, sx2, sy2 = slot["chip"]
        crop = image[sy1:sy2, sx1:sx2]
        if crop.size == 0:
            return None, None
        rw, rh = int(slot["resized_w"]), int(slot["resized_h"])
        resized = cv2.resize(crop, (rw, rh))
        ax, ay = (int(origin[0]), int(origin[1]))
        if ax < 0 or ay < 0 or ax + rw > DETECTOR_INPUT_SIZE or ay + rh > DETECTOR_INPUT_SIZE:
            return None, None
        atlas[ay:ay + rh, ax:ax + rw] = resized
        slots.append({
            **slot,
            "atlas_rect": (ax, ay, ax + rw, ay + rh),
        })
    return atlas, slots


def _intersection(box: BubbleBox, rect: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x1 = max(int(box.x1), int(rect[0]))
    y1 = max(int(box.y1), int(rect[1]))
    x2 = min(int(box.x2), int(rect[2]))
    y2 = min(int(box.y2), int(rect[3]))
    return x1, y1, x2, y2


def _map_atlas_boxes(
    boxes: list[BubbleBox],
    slots: list[dict],
) -> tuple[list[BubbleBox], bool]:
    mapped: list[BubbleBox] = []
    for box in boxes:
        cx = (float(box.x1) + float(box.x2)) * 0.5
        cy = (float(box.y1) + float(box.y2)) * 0.5
        owners = []
        for slot in slots:
            ax1, ay1, ax2, ay2 = slot["atlas_rect"]
            if ax1 <= cx <= ax2 and ay1 <= cy <= ay2:
                owners.append(slot)
        if len(owners) != 1:
            return [], False

        slot = owners[0]
        ix1, iy1, ix2, iy2 = _intersection(box, slot["atlas_rect"])
        if ix2 <= ix1 or iy2 <= iy1:
            return [], False
        ax1, ay1, _ax2, _ay2 = slot["atlas_rect"]
        sx1, sy1, sx2, sy2 = slot["chip"]
        scale_x = max(1e-9, float(slot["scale_x"]))
        scale_y = max(1e-9, float(slot["scale_y"]))

        local_x1 = int(math.floor((ix1 - ax1) / scale_x))
        local_y1 = int(math.floor((iy1 - ay1) / scale_y))
        local_x2 = int(math.ceil((ix2 - ax1) / scale_x))
        local_y2 = int(math.ceil((iy2 - ay1) / scale_y))
        crop_w, crop_h = sx2 - sx1, sy2 - sy1
        local_x1 = max(0, min(crop_w, local_x1))
        local_y1 = max(0, min(crop_h, local_y1))
        local_x2 = max(local_x1, min(crop_w, local_x2))
        local_y2 = max(local_y1, min(crop_h, local_y2))
        if local_x2 <= local_x1 or local_y2 <= local_y1:
            return [], False

        mapped_mask = None
        if box.mask is not None:
            mx1 = ix1 - int(box.x1)
            my1 = iy1 - int(box.y1)
            mx2 = mx1 + (ix2 - ix1)
            my2 = my1 + (iy2 - iy1)
            mask_crop = box.mask[my1:my2, mx1:mx2]
            target_w = local_x2 - local_x1
            target_h = local_y2 - local_y1
            if mask_crop.size == 0 or target_w <= 0 or target_h <= 0:
                return [], False
            mapped_mask = cv2.resize(
                mask_crop,
                (target_w, target_h),
                interpolation=cv2.INTER_NEAREST,
            )

        mapped.append(
            replace(
                box,
                x1=sx1 + local_x1,
                y1=sy1 + local_y1,
                x2=sx1 + local_x2,
                y2=sy1 + local_y2,
                mask=mapped_mask,
            )
        )
    return mapped, True


def _atlas_focus_calls(detector, image: np.ndarray, chips: list[tuple[int, int, int, int]]):
    remaining = list(chips)
    out: list[BubbleBox] = []
    atlas_calls = 0
    atlas_pairs = 0
    fallback_pair_calls = 0
    separate_calls = 0

    while len(remaining) >= 2:
        pair = None
        pair_indices = None
        for i in range(len(remaining)):
            for j in range(i + 1, len(remaining)):
                atlas, slots = _build_pair_atlas(image, remaining[i], remaining[j])
                if atlas is not None:
                    pair = (atlas, slots, remaining[i], remaining[j])
                    pair_indices = (i, j)
                    break
            if pair is not None:
                break
        if pair is None:
            break

        atlas, slots, chip_a, chip_b = pair
        raw = detector._detect_single_plain(atlas, 0, 0)
        atlas_calls += 1
        mapped, safe = _map_atlas_boxes(raw, slots)
        if safe:
            out.extend(mapped)
            atlas_pairs += 1
        else:
            for chip in (chip_a, chip_b):
                x1, y1, x2, y2 = chip
                crop = image[y1:y2, x1:x2]
                if crop.size:
                    out.extend(detector._detect_single_plain(crop, x1, y1))
                    separate_calls += 1
                    fallback_pair_calls += 1

        i, j = pair_indices
        for index in sorted((i, j), reverse=True):
            remaining.pop(index)

    for x1, y1, x2, y2 in remaining:
        crop = image[y1:y2, x1:x2]
        if crop.size:
            out.extend(detector._detect_single_plain(crop, x1, y1))
            separate_calls += 1

    return out, {
        "atlas_calls": int(atlas_calls),
        "atlas_pairs": int(atlas_pairs),
        "atlas_pair_fallback_model_calls": int(fallback_pair_calls),
        "atlas_separate_calls": int(separate_calls),
        "atlas_model_calls": int(atlas_calls + separate_calls),
        "atlas_original_focus_calls": int(len(chips)),
        "atlas_saved_calls": int(len(chips) - (atlas_calls + separate_calls)),
    }


def _focus_text_detect_atlas(
    detector,
    image: np.ndarray,
    proposals: list[BubbleBox],
):
    h, w = image.shape[:2]
    threshold = DETECTOR_INPUT_SIZE * DETECTOR_TALL_IMAGE_FACTOR
    if h <= threshold:
        boxes = detector._detect_single(image, 0, 0)
        result = [
            detector._with_semantics(box)
            for box in detector._filter_invalid(boxes, w, h)
        ]
        return result, {
            "focus_proposals": len(proposals),
            "focus_uncovered_proposals": 0,
            "focus_chip_calls": 0,
            "focus_source_pixels": 0,
            "focus_tensor_pixels": 0,
            "focus_deferred_regions": 0,
            "focus_fallback_calls": 0,
            "atlas_calls": 0,
            "atlas_pairs": 0,
            "atlas_pair_fallback_model_calls": 0,
            "atlas_separate_calls": 0,
            "atlas_model_calls": 0,
            "atlas_original_focus_calls": 0,
            "atlas_saved_calls": 0,
        }, []

    full_boxes = detector._detect_single_plain(image, 0, 0)
    uncovered = [
        proposal
        for proposal in proposals
        if not _proposal_has_full_text(proposal, full_boxes)
    ]
    fallback_image = image if not proposals and not full_boxes else None
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

    focus_boxes, atlas_metrics = _atlas_focus_calls(detector, image, chips)
    all_boxes = list(full_boxes) + list(focus_boxes)
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
    metrics = {
        "focus_proposals": len(proposals),
        "focus_uncovered_proposals": len(uncovered),
        "focus_chip_calls": len(chips),
        "focus_source_pixels": sum((x2-x1)*(y2-y1) for x1,y1,x2,y2 in chips),
        "focus_tensor_pixels": len(chips) * DETECTOR_INPUT_SIZE * DETECTOR_INPUT_SIZE,
        "focus_deferred_regions": len(deferred_boxes),
        "focus_fallback_calls": int(bool(fallback_image is not None and chips)),
    }
    metrics.update(atlas_metrics)
    return result, metrics, deferred_boxes


class FocusAtlasDetector(SequentialFastResidueAdaptiveFocusCombinedTextDetector):
    def detect(self, image: np.ndarray, *, parallel: bool = False):
        started_at = time.perf_counter()

        bubble_started = time.perf_counter()
        bubble_boxes = _adaptive_detect(self._bubble_model, image)
        bubble_ms = (time.perf_counter() - bubble_started) * 1000.0

        proposal_started = time.perf_counter()
        recovery_boxes = self.recovery.detect(image, existing=bubble_boxes)
        proposal_ms = (time.perf_counter() - proposal_started) * 1000.0

        text_started = time.perf_counter()
        text_boxes, focus_metrics, deferred_boxes = _focus_text_detect_atlas(
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
        metrics.update(focus_metrics)
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
        expected = (int(box.y2-box.y1), int(box.x2-box.x1))
        if local.shape != expected:
            local = cv2.resize(local, (expected[1], expected[0]), interpolation=cv2.INTER_NEAREST)
        sx1, sy1 = x1-int(box.x1), y1-int(box.y1)
        sx2, sy2 = sx1+(x2-x1), sy1+(y2-y1)
        mask[y1:y2, x1:x2] = np.maximum(
            mask[y1:y2, x1:x2],
            local[sy1:sy2, sx1:sx2],
        )
    return mask


def _review_signature(boxes: list[BubbleBox]) -> list[tuple]:
    return sorted(
        [
            (
                int(b.x1), int(b.y1), int(b.x2), int(b.y2),
                str(b.semantic_type), str(b.source_model), b.deferred_reason,
            )
            for b in boxes if bool(b.needs_review)
        ],
        key=repr,
    )


def _run(detector, paths: dict[int, Path], indices: list[int], warmup_path: Path, label: str) -> dict:
    warm = read_image(warmup_path)
    detector.detect(warm, parallel=False)
    del warm
    elapsed = []
    rows = []
    authority = {}
    review = {}
    counts = {}
    for index in indices:
        image = read_image(paths[index])
        h, w = image.shape[:2]
        started = time.perf_counter()
        boxes = detector.detect(image, parallel=False)
        elapsed.append((time.perf_counter()-started)*1000.0)
        metrics = detector.last_metrics()
        rows.append(metrics)
        authority[str(index)] = hashlib.sha256(_authority_mask(boxes, h, w).tobytes()).hexdigest()
        review[str(index)] = _review_signature(boxes)
        counts[str(index)] = {
            "boxes": len(boxes),
            "safe": sum(bool(b.safe_to_inpaint) for b in boxes),
            "review": sum(bool(b.needs_review) for b in boxes),
        }
        del image, boxes

    def mean_metric(name: str) -> float:
        vals = [float(row.get(name) or 0.0) for row in rows]
        return round(statistics.mean(vals), 3) if vals else 0.0

    return {
        "label": label,
        "wall_ms": round(sum(elapsed), 3),
        "mean_ms": round(statistics.mean(elapsed), 3),
        "median_ms": round(statistics.median(elapsed), 3),
        "bubble_model_mean_ms": mean_metric("bubble_model_ms"),
        "text_model_mean_ms": mean_metric("text_model_ms"),
        "focus_chip_calls": int(sum(int(row.get("focus_chip_calls") or 0) for row in rows)),
        "atlas_calls": int(sum(int(row.get("atlas_calls") or 0) for row in rows)),
        "atlas_pairs": int(sum(int(row.get("atlas_pairs") or 0) for row in rows)),
        "atlas_model_calls": int(sum(int(row.get("atlas_model_calls") or 0) for row in rows)),
        "atlas_saved_calls": int(sum(int(row.get("atlas_saved_calls") or 0) for row in rows)),
        "authority_hashes": authority,
        "review_signatures": review,
        "counts": counts,
    }


def _pct(before: float, after: float) -> float:
    return round((1.0-after/max(1.0,before))*100.0, 2)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pipeline = ChapterPipeline()
    chapter_id = hashlib.sha256(f"focus-atlas-{time.time_ns()}".encode()).hexdigest()[:8]
    manifest = pipeline.download_chapter(CHAPTER_URL, chapter_id, workers=2)
    pages = manifest.get("pages", [])
    total = len(pages)
    indices = sorted({round(i*(total-1)/(SAMPLE_COUNT-1)) for i in range(SAMPLE_COUNT)})
    paths = {i: Path(pages[i]["original"]) for i in indices}
    warmup_index = next(i for i in range(total) if i not in set(indices))
    warmup_path = Path(pages[warmup_index]["original"])

    control_detector = SequentialFastResidueAdaptiveFocusCombinedTextDetector()
    control = _run(control_detector, paths, indices, warmup_path, "control_independent_focus")
    del control_detector
    gc.collect()

    candidate_detector = FocusAtlasDetector()
    candidate = _run(candidate_detector, paths, indices, warmup_path, "candidate_focus_atlas")
    del candidate_detector
    gc.collect()

    authority_mismatch = [
        k for k, v in control["authority_hashes"].items()
        if candidate["authority_hashes"].get(k) != v
    ]
    review_mismatch = [
        k for k, v in control["review_signatures"].items()
        if candidate["review_signatures"].get(k) != v
    ]
    count_mismatch = [
        k for k, v in control["counts"].items()
        if candidate["counts"].get(k) != v
    ]
    report = {
        "chapter_url": CHAPTER_URL,
        "chapter_id": chapter_id,
        "total_slices": total,
        "sample_indices": indices,
        "atlas_guard": ATLAS_GUARD,
        "control": control,
        "candidate": candidate,
        "speedup": {
            "wall_reduction_pct": _pct(control["wall_ms"], candidate["wall_ms"]),
            "text_model_reduction_pct": _pct(
                control["text_model_mean_ms"],
                candidate["text_model_mean_ms"],
            ),
            "focus_call_reduction_pct": _pct(
                float(control["focus_chip_calls"]),
                float(candidate["atlas_model_calls"]),
            ) if control["focus_chip_calls"] else 0.0,
        },
        "quality": {
            "authority_mask_mismatch_pages": authority_mismatch,
            "review_signature_mismatch_pages": review_mismatch,
            "box_count_mismatch_pages": count_mismatch,
        },
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("FOCUS_ATLAS_AB=" + json.dumps(report, ensure_ascii=False), flush=True)

    if authority_mismatch:
        raise RuntimeError(
            "focus atlas changed destructive authority masks: "
            + json.dumps(authority_mismatch)
        )


if __name__ == "__main__":
    main()
