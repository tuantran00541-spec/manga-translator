from __future__ import annotations

from contextlib import contextmanager
import math
import threading
import time
from typing import Callable, Iterator

import cv2
import numpy as np

from app.detector.bubble_detector import BubbleBox, YoloDetector
from app.detector.combined_detector import CombinedTextDetector
from app.detector.evidence import DetectionEvidence, EvidenceKind
from app.detector.proposal_planner import ProposalPlanner
from app.parameters import (
    DETECTOR_FOCUS_HARD_MAX_CHIPS,
    DETECTOR_FOCUS_MAX_CHIPS,
    DETECTOR_FOCUS_PROPOSALS_PER_EXTRA_CHIP,
    DETECTOR_FOCUS_ROI_FIRST_ENABLED,
    DETECTOR_INPUT_SIZE,
    DETECTOR_TALL_IMAGE_FACTOR,
    DETECTOR_WINDOW_OVERLAP,
)


# Values validated by the real full-chapter V2/V4 benchmark lane.
ADAPTIVE_TILE_MAX = max(DETECTOR_INPUT_SIZE, 1344)
ADAPTIVE_OVERLAP = DETECTOR_WINDOW_OVERLAP
FOCUS_PAD_X = 96
FOCUS_PAD_Y = 96
FOCUS_FREE_PAD_X = 160
FOCUS_FREE_PAD_Y = 160
FOCUS_MERGE_GAP = 64
FOCUS_MAX_CHIPS = DETECTOR_FOCUS_MAX_CHIPS
FOCUS_MAX_SOURCE_SIDE = ADAPTIVE_TILE_MAX
FOCUS_SOURCE_PIXEL_BUDGET = FOCUS_MAX_CHIPS * FOCUS_MAX_SOURCE_SIDE * FOCUS_MAX_SOURCE_SIDE
FOCUS_TENSOR_PIXEL_BUDGET = FOCUS_MAX_CHIPS * DETECTOR_INPUT_SIZE * DETECTOR_INPUT_SIZE
FOCUS_TILE_OVERLAP = 96
FOCUS_COMPLETENESS_SPAN_MIN = 0.18
FOCUS_BOUNDARY_RATIO = 0.06


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


def _matching_text_boxes(
    proposal: BubbleBox,
    text_boxes: list[BubbleBox],
) -> list[BubbleBox]:
    px1, py1, px2, py2 = (int(proposal.x1), int(proposal.y1), int(proposal.x2), int(proposal.y2))
    if px2 <= px1 or py2 <= py1:
        return []
    matched: list[BubbleBox] = []
    for text in text_boxes:
        tx1, ty1, tx2, ty2 = (int(text.x1), int(text.y1), int(text.x2), int(text.y2))
        if tx2 <= tx1 or ty2 <= ty1:
            continue
        cx = (tx1 + tx2) * 0.5
        cy = (ty1 + ty2) * 0.5
        inside = px1 <= cx <= px2 and py1 <= cy <= py2
        ix1, iy1 = max(px1, tx1), max(py1, ty1)
        ix2, iy2 = min(px2, tx2), min(py2, ty2)
        overlap = 0.0
        if ix2 > ix1 and iy2 > iy1:
            text_area = max(1, (tx2 - tx1) * (ty2 - ty1))
            overlap = ((ix2 - ix1) * (iy2 - iy1)) / float(text_area)
        if inside or overlap >= 0.6:
            matched.append(text)
    return matched


def _proposal_has_full_text(
    proposal: BubbleBox,
    text_boxes: list[BubbleBox],
) -> bool:
    """Conservative completeness: scale, boundary and residual-span aware."""
    if _proposal_is_free_text(proposal):
        return False
    px1, py1, px2, py2 = (int(proposal.x1), int(proposal.y1), int(proposal.x2), int(proposal.y2))
    pw, ph = px2 - px1, py2 - py1
    if pw <= 0 or ph <= 0:
        return True
    matched = _matching_text_boxes(proposal, text_boxes)
    if not matched:
        return False
    if max(pw, ph) > FOCUS_MAX_SOURCE_SIDE:
        return False
    sx1 = min(int(box.x1) for box in matched)
    sy1 = min(int(box.y1) for box in matched)
    sx2 = max(int(box.x2) for box in matched)
    sy2 = max(int(box.y2) for box in matched)
    span_x = (sx2 - sx1) / float(max(1, pw))
    span_y = (sy2 - sy1) / float(max(1, ph))
    if span_x < FOCUS_COMPLETENESS_SPAN_MIN and span_y < FOCUS_COMPLETENESS_SPAN_MIN:
        return False
    boundary = max(4, int(round(min(pw, ph) * FOCUS_BOUNDARY_RATIO)))
    return not any(
        int(text.x1) <= px1 + boundary
        or int(text.y1) <= py1 + boundary
        or int(text.x2) >= px2 - boundary
        or int(text.y2) >= py2 - boundary
        for text in matched
    )


def _axis_tiles(start: int, end: int, bound: int) -> list[tuple[int, int]]:
    start = max(0, min(int(start), int(bound)))
    end = max(start, min(int(end), int(bound)))
    if end <= start:
        return []
    limit = max(1, min(int(FOCUS_MAX_SOURCE_SIDE), int(bound)))
    if end - start <= limit:
        return [(start, end)]
    overlap = max(0, min(int(FOCUS_TILE_OVERLAP), limit - 1))
    stride = max(1, limit - overlap)
    out: list[tuple[int, int]] = []
    cursor = start
    while cursor < end:
        stop = min(end, cursor + limit)
        if stop - cursor < limit and end - limit >= start:
            cursor = end - limit
            stop = end
        item = (cursor, stop)
        if not out or item != out[-1]:
            out.append(item)
        if stop >= end:
            break
        cursor += stride
    return out


def _proposal_tiles(proposal: BubbleBox, width: int, height: int) -> list[tuple[int, int, int, int]]:
    free = _proposal_is_free_text(proposal)
    pad_x = FOCUS_FREE_PAD_X if free else FOCUS_PAD_X
    pad_y = FOCUS_FREE_PAD_Y if free else FOCUS_PAD_Y
    x1 = max(0, int(proposal.x1) - pad_x)
    y1 = max(0, int(proposal.y1) - pad_y)
    x2 = min(int(width), int(proposal.x2) + pad_x)
    y2 = min(int(height), int(proposal.y2) + pad_y)
    if x2 <= x1 or y2 <= y1:
        return []
    xs = _axis_tiles(x1, x2, width)
    ys = _axis_tiles(y1, y2, height)
    return [(ax, ay, bx, by) for ay, by in ys for ax, bx in xs]


def _page_tiles(width: int, height: int) -> list[tuple[int, int, int, int]]:
    xs = _axis_tiles(0, width, width)
    ys = _axis_tiles(0, height, height)
    return [(ax, ay, bx, by) for ay, by in ys for ax, bx in xs]


def _focus_signal_score(image: np.ndarray, chip: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = chip
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return -1.0
    h, w = crop.shape[:2]
    scale = min(1.0, 192.0 / max(1, h, w))
    if scale < 1.0:
        crop = cv2.resize(
            crop,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    if crop.ndim == 2:
        gray = crop
        chroma = 0.0
    else:
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        gray = lab[:, :, 0]
        chroma = float(lab[:, :, 1].std() + lab[:, :, 2].std())
    edges = cv2.Canny(gray, 60, 160)
    return chroma + 80.0 * float(np.mean(edges > 0)) + 0.05 * float(gray.std())


def plan_focus_chips(
    height: int,
    width: int,
    proposals: list[BubbleBox],
    *,
    max_chips: int = FOCUS_MAX_CHIPS,
    source_pixel_budget: int = FOCUS_SOURCE_PIXEL_BUDGET,
    tensor_pixel_budget: int = FOCUS_TENSOR_PIXEL_BUDGET,
    fallback_image: np.ndarray | None = None,
) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
    """Bound source geometry, model calls and total focus pixels."""
    height, width = int(height), int(width)
    if height <= 0 or width <= 0:
        return [], []
    ranked: list[tuple[tuple[int, int, int], tuple[int, int, int, int]]] = []
    serial = 0
    for proposal in proposals:
        area = max(1, (int(proposal.x2) - int(proposal.x1)) * (int(proposal.y2) - int(proposal.y1)))
        free_rank = 0 if _proposal_is_free_text(proposal) else 1
        for chip in _proposal_tiles(proposal, width, height):
            ranked.append(((free_rank, area, serial), chip))
            serial += 1
    if not ranked and fallback_image is not None and fallback_image.size:
        candidates = _page_tiles(width, height)
        if candidates:
            best = max(candidates, key=lambda chip: (_focus_signal_score(fallback_image, chip), -chip[1], -chip[0]))
            ranked.append(((2, 0, serial), best))
    dedup: dict[tuple[int, int, int, int], tuple[int, int, int]] = {}
    for priority, chip in ranked:
        if chip not in dedup or priority < dedup[chip]:
            dedup[chip] = priority
    ordered = sorted(((priority, chip) for chip, priority in dedup.items()), key=lambda item: (item[0], item[1]))
    scheduled: list[tuple[int, int, int, int]] = []
    deferred: list[tuple[int, int, int, int]] = []
    used_source = 0
    used_tensor = 0
    tensor_per_call = DETECTOR_INPUT_SIZE * DETECTOR_INPUT_SIZE
    for _priority, chip in ordered:
        x1, y1, x2, y2 = chip
        pixels = max(0, (x2 - x1) * (y2 - y1))
        fits = (
            len(scheduled) < max(0, int(max_chips))
            and used_source + pixels <= max(0, int(source_pixel_budget))
            and used_tensor + tensor_per_call <= max(0, int(tensor_pixel_budget))
        )
        if fits:
            scheduled.append(chip)
            used_source += pixels
            used_tensor += tensor_per_call
        else:
            deferred.append(chip)
    return scheduled, deferred


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
        if detector.model_role == "text_segmenter":
            all_boxes.extend(detector._detect_single_plain(image, 0, 0))

        boxes = detector._nms_boxes(all_boxes)

    return [
        detector._with_semantics(box)
        for box in detector._filter_invalid(boxes, w, h)
    ]


def _proposal_resolved_by_roi(
    proposal: BubbleBox,
    text_boxes: list[BubbleBox],
) -> bool:
    """Require verified segmenter evidence before an ROI can close a proposal."""
    matched = [
        box
        for box in _matching_text_boxes(proposal, text_boxes)
        if box.verified_mask and box.source_role == "text_segmenter"
    ]
    if not matched:
        return False
    if _proposal_is_free_text(proposal):
        return True
    return _proposal_has_full_text(proposal, matched)


def _chip_has_internal_boundary_text(
    chip: tuple[int, int, int, int],
    boxes: list[BubbleBox],
    *,
    page_width: int,
    page_height: int,
) -> bool:
    """Treat text touching an internal ROI edge as incomplete evidence."""
    x1, y1, x2, y2 = chip
    width, height = x2 - x1, y2 - y1
    if width <= 0 or height <= 0:
        return True
    boundary = max(
        4,
        int(round(min(width, height) * FOCUS_BOUNDARY_RATIO)),
    )
    for box in boxes:
        if (
            (x1 > 0 and int(box.x1) <= x1 + boundary)
            or (y1 > 0 and int(box.y1) <= y1 + boundary)
            or (x2 < page_width and int(box.x2) >= x2 - boundary)
            or (y2 < page_height and int(box.y2) >= y2 - boundary)
        ):
            return True
    return False


def _focus_text_detect_full_first(
    detector: YoloDetector,
    image: np.ndarray,
    proposals: list[BubbleBox],
) -> tuple[list[BubbleBox], dict[str, int], list[BubbleBox]]:
    """Canonical tall-page schedule used by production and ROI fallback."""
    h, w = image.shape[:2]
    full_boxes = detector._detect_single_plain(image, 0, 0)
    uncovered = [
        proposal
        for proposal in proposals
        if not _proposal_has_full_text(proposal, full_boxes)
    ]
    fallback_image = image if not proposals and not full_boxes else None
    extra = (
        max(0, len(uncovered) - 1)
        // DETECTOR_FOCUS_PROPOSALS_PER_EXTRA_CHIP
    )
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

    all_boxes = list(full_boxes)
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
            source_model="adaptive_scheduler", class_name="focus_deferred",
            semantic_type="review_region", mask_source="none",
            safe_to_inpaint=False, ocr_eligible=False, needs_review=True,
            source_role="scheduler", deferred_reason="focus_budget_exhausted",
        )
        for x1, y1, x2, y2 in deferred
    ]
    return result, {
        "focus_roi_first_page": 0,
        "focus_proposals": len(proposals),
        "focus_uncovered_proposals": len(uncovered),
        "focus_chip_calls": len(chips),
        "focus_source_pixels": sum(
            (x2 - x1) * (y2 - y1)
            for x1, y1, x2, y2 in chips
        ),
        "focus_tensor_pixels": (
            len(chips) * DETECTOR_INPUT_SIZE * DETECTOR_INPUT_SIZE
        ),
        "focus_deferred_regions": len(deferred_boxes),
        "focus_fallback_calls": int(bool(fallback_image is not None and chips)),
        "focus_full_page_calls": 1,
        "focus_full_page_skipped": 0,
        "focus_fallback_no_proposals": 0,
        "focus_fallback_deferred": 0,
        "focus_fallback_unresolved": 0,
        "focus_fallback_boundary": 0,
    }, deferred_boxes


def _focus_text_detect_roi_first(
    detector: YoloDetector,
    image: np.ndarray,
    proposals: list[BubbleBox],
) -> tuple[list[BubbleBox], dict[str, int], list[BubbleBox]]:
    """Run bounded authority ROIs first and pay for full-frame only on doubt."""
    h, w = image.shape[:2]
    extra = (
        max(0, len(proposals) - 1)
        // DETECTOR_FOCUS_PROPOSALS_PER_EXTRA_CHIP
    )
    adaptive_max_chips = min(
        DETECTOR_FOCUS_HARD_MAX_CHIPS,
        FOCUS_MAX_CHIPS + extra,
    )
    scale = adaptive_max_chips / float(max(1, FOCUS_MAX_CHIPS))

    # Geometry-only clustered coverage is the active ROI scheduler in the
    # experimental ROI-first path. Authority and fallback decisions remain
    # separate below: this planner only decides which bounded crops to run.
    proposal_evidence = [
        DetectionEvidence(
            bbox=(int(box.x1), int(box.y1), int(box.x2), int(box.y2)),
            confidence=float(box.confidence),
            semantic=str(getattr(box, "semantic_type", "unknown")),
            source=str(getattr(box, "source_model", "unknown")),
            evidence_kind=(
                EvidenceKind.RECOVERY
                if str(getattr(box, "source_model", "")) == "opencv_mser"
                else EvidenceKind.BOX
            ),
            metadata={
                "pad_x": FOCUS_FREE_PAD_X if _proposal_is_free_text(box) else FOCUS_PAD_X,
                "pad_y": FOCUS_FREE_PAD_Y if _proposal_is_free_text(box) else FOCUS_PAD_Y,
                # Structured detector proposals get first scheduling priority;
                # review/recovery evidence remains represented but cannot starve
                # every bounded authority ROI on dense pages.
                "priority": 1 if _proposal_is_free_text(box) else 0,
            },
        )
        for box in proposals
    ]
    planner = ProposalPlanner(
        pad_x=FOCUS_PAD_X,
        pad_y=FOCUS_PAD_Y,
        max_rois=adaptive_max_chips,
        max_source_side=FOCUS_MAX_SOURCE_SIDE,
        merge_overlap=0.10,
        merge_gap=FOCUS_MERGE_GAP,
    )
    plan = planner.plan_clustered_shape(
        image_shape=(h, w),
        proposals=proposal_evidence,
        source_pixel_budget=int(round(FOCUS_SOURCE_PIXEL_BUDGET * scale)),
        tensor_pixel_budget=int(round(FOCUS_TENSOR_PIXEL_BUDGET * scale)),
        tensor_pixels_per_roi=DETECTOR_INPUT_SIZE * DETECTOR_INPUT_SIZE,
        tile_overlap=FOCUS_TILE_OVERLAP,
        span_short_axis=True,
    )

    chips = list(plan.rois)
    deferred = list(plan.deferred)

    roi_raw: list[BubbleBox] = []
    boundary_ambiguous = False
    for chip in chips:
        x1, y1, x2, y2 = chip
        crop = image[y1:y2, x1:x2]
        if not crop.size:
            continue
        chip_boxes = detector._detect_single_plain(crop, x1, y1)
        roi_raw.extend(chip_boxes)
        if _chip_has_internal_boundary_text(
            chip,
            chip_boxes,
            page_width=w,
            page_height=h,
        ):
            boundary_ambiguous = True

    roi_nms = detector._nms_boxes(roi_raw)
    roi_semantic = [
        detector._with_semantics(box)
        for box in detector._filter_invalid(roi_nms, w, h)
    ]
    unresolved = [
        proposal
        for proposal in proposals
        if not _proposal_resolved_by_roi(proposal, roi_semantic)
    ]

    unresolved_free = [box for box in unresolved if _proposal_is_free_text(box)]
    unresolved_structured = [box for box in unresolved if not _proposal_is_free_text(box)]

    fallback_no_proposals = not proposals
    fallback_deferred = bool(deferred)
    fallback_unresolved = bool(unresolved)
    needs_full_page = bool(
        fallback_no_proposals
        or fallback_deferred
        or fallback_unresolved
        or boundary_ambiguous
    )

    roi_verified = [
        box for box in roi_semantic
        if box.verified_mask and box.source_role == "text_segmenter"
    ]
    full_verified: list[BubbleBox] = []
    full_unique_verified: list[BubbleBox] = []
    fallback_metrics: dict[str, int] = {}

    if needs_full_page:
        # A fallback is a correctness escape hatch, not another source to fuse
        # with ROI detections. Re-run the canonical full-first schedule so an
        # ROI box cannot win NMS and silently replace baseline authority.
        result, fallback_metrics, deferred_boxes = _focus_text_detect_full_first(
            detector,
            image,
            proposals,
        )
        full_verified = [
            box for box in result
            if box.verified_mask and box.source_role == "text_segmenter"
        ]
        full_unique_verified = [
            box for box in full_verified
            if not _matching_text_boxes(box, roi_verified)
        ]
    else:
        result = roi_semantic
        deferred_boxes = []
    return result, {
        "focus_roi_first_page": 1,
        "focus_proposals": len(proposals),
        "focus_uncovered_proposals": len(unresolved),
        "focus_chip_calls": len(chips),
        "focus_source_pixels": sum(
            (x2 - x1) * (y2 - y1)
            for x1, y1, x2, y2 in chips
        ),
        "focus_tensor_pixels": (
            len(chips) * DETECTOR_INPUT_SIZE * DETECTOR_INPUT_SIZE
        ),
        "focus_deferred_regions": len(deferred),
        "focus_fallback_calls": int(
            fallback_metrics.get("focus_fallback_calls", 0)
        ),
        "focus_full_page_calls": int(needs_full_page),
        "focus_full_page_skipped": int(not needs_full_page),
        "focus_fallback_no_proposals": int(fallback_no_proposals),
        "focus_fallback_deferred": int(fallback_deferred),
        "focus_fallback_unresolved": int(fallback_unresolved),
        "focus_fallback_boundary": int(boundary_ambiguous),
        "focus_unresolved_free_text": len(unresolved_free),
        "focus_unresolved_structured": len(unresolved_structured),
        "focus_roi_verified_boxes": len(roi_verified),
        "focus_full_verified_boxes": len(full_verified),
        "focus_full_unique_verified_boxes": len(full_unique_verified),
        "focus_fallback_refine_calls": int(
            fallback_metrics.get("focus_chip_calls", 0)
        ),
        "focus_fallback_refine_deferred_regions": int(
            fallback_metrics.get("focus_deferred_regions", 0)
        ),
        "focus_clustered_rois": len(plan.rois),
        "focus_clustered_candidate_rois": plan.candidate_roi_count,
        "focus_clustered_deferred_rois": plan.deferred_roi_count,
        "focus_clustered_covered_proposals": len(plan.covered),
        "focus_clustered_deferred_proposals": len(plan.deferred),
        "focus_clustered_groups": plan.group_count,
    }, deferred_boxes


def _focus_text_detect(
    detector: YoloDetector,
    image: np.ndarray,
    proposals: list[BubbleBox],
) -> tuple[list[BubbleBox], dict[str, int], list[BubbleBox]]:
    h, w = image.shape[:2]
    threshold = DETECTOR_INPUT_SIZE * DETECTOR_TALL_IMAGE_FACTOR
    if h <= threshold:
        boxes = detector._detect_single(image, 0, 0)
        result = [detector._with_semantics(box) for box in detector._filter_invalid(boxes, w, h)]
        return result, {
            "focus_roi_first_page": 0,
            "focus_proposals": len(proposals), "focus_uncovered_proposals": 0,
            "focus_chip_calls": 0, "focus_source_pixels": 0,
            "focus_tensor_pixels": 0, "focus_deferred_regions": 0,
            "focus_fallback_calls": 0, "focus_full_page_calls": 1,
            "focus_full_page_skipped": 0,
            "focus_fallback_no_proposals": 0,
            "focus_fallback_deferred": 0,
            "focus_fallback_unresolved": 0,
            "focus_fallback_boundary": 0,
        }, []

    if DETECTOR_FOCUS_ROI_FIRST_ENABLED:
        return _focus_text_detect_roi_first(detector, image, proposals)

    return _focus_text_detect_full_first(detector, image, proposals)


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
        text_boxes, focus_metrics, deferred_boxes = _focus_text_detect(
            self._text_model,
            image,
            list(bubble_boxes) + list(recovery_boxes),
        )
        text_ms = (time.perf_counter() - text_started) * 1000.0

        with self.bubble_detector.prefetched(image, bubble_boxes):
            with self.text_detector.prefetched(image, text_boxes):
                result = super().detect(image, parallel=False)

        if deferred_boxes:
            result.extend(deferred_boxes)

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
