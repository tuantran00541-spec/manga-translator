from __future__ import annotations

from contextlib import contextmanager
import math
import threading
import time
from typing import Callable, Iterator

import numpy as np

from app.detector.bubble_detector import BubbleBox, YoloDetector
from app.detector.combined_detector import CombinedTextDetector
from app.parameters import (
    DETECTOR_INPUT_SIZE,
    DETECTOR_TALL_IMAGE_FACTOR,
    DETECTOR_WINDOW_OVERLAP,
)


# Values validated by the real full-chapter V2/V4 benchmark lane.
ADAPTIVE_TILE_MAX = max(DETECTOR_INPUT_SIZE, 1344)
ADAPTIVE_OVERLAP = DETECTOR_WINDOW_OVERLAP
FOCUS_PAD_Y = 96
FOCUS_FREE_PAD_Y = 160
FOCUS_MERGE_GAP = 64
FOCUS_MAX_CHIPS = 2


def plan_adaptive_windows(
    height: int,
    *,
    tile_max: int = ADAPTIVE_TILE_MAX,
    overlap: int = ADAPTIVE_OVERLAP,
) -> list[tuple[int, int]]:
    """Plan gap-free tall-image windows while minimizing model calls."""
    height = int(height)
    if height <= 0:
        return []

    tile_max = max(DETECTOR_INPUT_SIZE, int(tile_max))
    overlap = max(0, min(int(overlap), tile_max - 1))
    if height <= tile_max:
        return [(0, height)]

    stride_budget = max(1, tile_max - overlap)
    count = max(2, int(math.ceil((height - overlap) / stride_budget)))
    tile_h = int(math.ceil((height + (count - 1) * overlap) / count))
    tile_h = max(DETECTOR_INPUT_SIZE, min(tile_max, tile_h))
    if tile_h >= height:
        return [(0, height)]

    span = height - tile_h
    starts = [
        int(round(index * span / (count - 1)))
        for index in range(count)
    ]
    starts[0] = 0
    starts[-1] = span

    windows: list[tuple[int, int]] = []
    for start in starts:
        start = max(0, min(int(start), height - 1))
        end = min(height, start + tile_h)
        if windows and start > windows[-1][1]:
            start = windows[-1][1]
            end = min(height, start + tile_h)
        if not windows or (start, end) != windows[-1]:
            windows.append((start, end))

    if windows[-1][1] < height:
        windows.append((max(0, height - tile_h), height))
    return windows


def _proposal_is_free_text(box: BubbleBox) -> bool:
    return (
        str(getattr(box, "semantic_type", "")) == "free_text"
        or str(getattr(box, "source_model", "")) == "opencv_mser"
    )


def _proposal_has_full_text(
    proposal: BubbleBox,
    text_boxes: list[BubbleBox],
) -> bool:
    if _proposal_is_free_text(proposal):
        return False

    px1 = int(getattr(proposal, "x1", 0))
    py1 = int(getattr(proposal, "y1", 0))
    px2 = int(getattr(proposal, "x2", 0))
    py2 = int(getattr(proposal, "y2", 0))
    if px2 <= px1 or py2 <= py1:
        return True

    for text in text_boxes:
        tx1 = int(getattr(text, "x1", 0))
        ty1 = int(getattr(text, "y1", 0))
        tx2 = int(getattr(text, "x2", 0))
        ty2 = int(getattr(text, "y2", 0))
        if tx2 <= tx1 or ty2 <= ty1:
            continue

        cx = (tx1 + tx2) * 0.5
        cy = (ty1 + ty2) * 0.5
        if px1 <= cx <= px2 and py1 <= cy <= py2:
            return True

        ix1 = max(px1, tx1)
        iy1 = max(py1, ty1)
        ix2 = min(px2, tx2)
        iy2 = min(py2, ty2)
        if ix2 <= ix1 or iy2 <= iy1:
            continue

        text_area = max(1, (tx2 - tx1) * (ty2 - ty1))
        overlap = ((ix2 - ix1) * (iy2 - iy1)) / float(text_area)
        if overlap >= 0.6:
            return True

    return False


def plan_focus_bands(
    height: int,
    proposals: list[BubbleBox],
    *,
    max_chips: int = FOCUS_MAX_CHIPS,
    merge_gap: int = FOCUS_MERGE_GAP,
) -> list[tuple[int, int]]:
    """Collapse proposal Y ranges into a bounded set of full-width focus chips."""
    height = int(height)
    if height <= 0:
        return []

    intervals: list[tuple[int, int]] = []
    for proposal in proposals:
        y1 = int(getattr(proposal, "y1", 0))
        y2 = int(getattr(proposal, "y2", 0))
        if y2 <= y1:
            continue
        pad = FOCUS_FREE_PAD_Y if _proposal_is_free_text(proposal) else FOCUS_PAD_Y
        start = max(0, y1 - pad)
        end = min(height, y2 + pad)
        if end > start:
            intervals.append((start, end))

    if not intervals:
        return []

    intervals.sort()
    merged: list[list[int]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1] + int(merge_gap):
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    max_chips = max(1, int(max_chips))
    while len(merged) > max_chips:
        gaps = [
            max(0, merged[index + 1][0] - merged[index][1])
            for index in range(len(merged) - 1)
        ]
        merge_at = min(range(len(gaps)), key=lambda index: gaps[index])
        merged[merge_at][1] = merged[merge_at + 1][1]
        del merged[merge_at + 1]

    return [
        (int(start), int(end))
        for start, end in merged
        if end > start
    ]


def _adaptive_detect(
    detector: YoloDetector,
    image: np.ndarray,
) -> list[BubbleBox]:
    h, w = image.shape[:2]
    threshold = DETECTOR_INPUT_SIZE * DETECTOR_TALL_IMAGE_FACTOR

    if h <= threshold:
        boxes = detector._detect_single(image, 0, 0)
    else:
        all_boxes: list[BubbleBox] = []
        for start, end in plan_adaptive_windows(h):
            all_boxes.extend(
                detector._detect_single(image[start:end, :], 0, start)
            )

        # Keep the full-image text pass as recall insurance. Focus mode uses its
        # own full pass and therefore does not call this path for the main text
        # detector, but grayscale fallback still benefits from it.
        if "text_segmenter" in str(detector.source_model).lower():
            all_boxes.extend(detector._detect_single_plain(image, 0, 0))

        boxes = detector._nms_boxes(all_boxes)

    return [
        detector._with_semantics(box)
        for box in detector._filter_invalid(boxes, w, h)
    ]


def _focus_text_detect(
    detector: YoloDetector,
    image: np.ndarray,
    proposals: list[BubbleBox],
) -> tuple[list[BubbleBox], dict[str, int]]:
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
        }

    full_boxes = detector._detect_single_plain(image, 0, 0)
    uncovered = [
        proposal
        for proposal in proposals
        if not _proposal_has_full_text(proposal, full_boxes)
    ]
    bands = plan_focus_bands(h, uncovered)

    all_boxes = list(full_boxes)
    for start, end in bands:
        crop = image[start:end, :]
        if crop.size == 0:
            continue
        all_boxes.extend(detector._detect_single_plain(crop, 0, start))

    boxes = detector._nms_boxes(all_boxes)
    result = [
        detector._with_semantics(box)
        for box in detector._filter_invalid(boxes, w, h)
    ]
    return result, {
        "focus_proposals": len(proposals),
        "focus_uncovered_proposals": len(uncovered),
        "focus_chip_calls": len(bands),
    }


class _AdaptiveDetectorProxy:
    """Thread-local one-shot cache in front of an adaptive detector fallback."""

    def __init__(
        self,
        detector: YoloDetector,
        fallback: Callable[[YoloDetector, np.ndarray], list[BubbleBox]],
    ):
        self._detector = detector
        self._fallback = fallback
        self._local = threading.local()

    @contextmanager
    def prefetched(
        self,
        image: np.ndarray,
        boxes: list[BubbleBox],
    ) -> Iterator[None]:
        previous = getattr(self._local, "value", None)
        self._local.value = {
            "image_id": id(image),
            "shape": tuple(image.shape[:2]),
            "boxes": list(boxes),
            "used": False,
        }
        try:
            yield
        finally:
            self._local.value = previous

    def detect(self, image: np.ndarray) -> list[BubbleBox]:
        state = getattr(self._local, "value", None)
        if (
            state
            and not state["used"]
            and state["image_id"] == id(image)
            and state["shape"] == tuple(image.shape[:2])
        ):
            state["used"] = True
            return list(state["boxes"])
        return self._fallback(self._detector, image)

    def __getattr__(self, name):
        return getattr(self._detector, name)


class AdaptiveFocusCombinedTextDetector(CombinedTextDetector):
    """Production detector validated by the V2 logic + V4 OpenVINO benchmark.

    Bubble proposals use a minimal gap-free tall-image window plan. The text
    segmenter runs one full-image pass and only refines proposal bands that lack
    plausible full-pass text, capped at two focus chips. Existing production
    grouping, safety classification, grayscale fallback, MSER recovery and final
    NMS stay in CombinedTextDetector.
    """

    def __init__(self):
        super().__init__()
        self._bubble_model = self.bubble_detector
        self._text_model = self.text_detector
        self.bubble_detector = _AdaptiveDetectorProxy(
            self._bubble_model, _adaptive_detect
        )
        self.text_detector = _AdaptiveDetectorProxy(
            self._text_model, _adaptive_detect
        )

    def detect(
        self,
        image: np.ndarray,
        *,
        parallel: bool = False,
    ) -> list[BubbleBox]:
        # The chapter pipeline already runs two pages concurrently. Keeping the
        # two neural models sequential inside each page avoids CPU oversubscription.
        started_at = time.perf_counter()

        bubble_started = time.perf_counter()
        bubble_boxes = _adaptive_detect(self._bubble_model, image)
        bubble_ms = (time.perf_counter() - bubble_started) * 1000.0

        proposal_started = time.perf_counter()
        recovery_boxes = self.recovery.detect(image, existing=bubble_boxes)
        proposal_ms = (time.perf_counter() - proposal_started) * 1000.0

        text_started = time.perf_counter()
        text_boxes, focus_metrics = _focus_text_detect(
            self._text_model,
            image,
            list(bubble_boxes) + list(recovery_boxes),
        )
        text_ms = (time.perf_counter() - text_started) * 1000.0

        with self.bubble_detector.prefetched(image, bubble_boxes):
            with self.text_detector.prefetched(image, text_boxes):
                result = super().detect(image, parallel=False)

        metrics = dict(getattr(self._metrics_local, "value", {}) or {})
        metrics["bubble_model_ms"] = round(bubble_ms, 3)
        metrics["text_model_ms"] = round(text_ms, 3)
        metrics["focus_prefetch_mser_ms"] = round(proposal_ms, 3)
        metrics["focus_prefetch_proposals"] = len(recovery_boxes)
        metrics["mser_ms"] = round(
            float(metrics.get("mser_ms", 0.0)) + proposal_ms,
            3,
        )
        metrics.update(focus_metrics)
        metrics["total_ms"] = round(
            (time.perf_counter() - started_at) * 1000.0,
            3,
        )
        self._metrics_local.value = metrics
        return result
