from pathlib import Path
import re


def replace_once(source: str, pattern: str, replacement: str, label: str) -> str:
    updated, count = re.subn(pattern, replacement, source, count=1, flags=re.S)
    if count != 1:
        raise RuntimeError(f"{label}: expected one replacement, got {count}")
    return updated


adaptive_path = Path("app/detector/adaptive_focus_detector.py")
adaptive = adaptive_path.read_text(encoding="utf-8")
adaptive = adaptive.replace("import numpy as np\n", "import cv2\nimport numpy as np\n", 1)
adaptive = replace_once(
    adaptive,
    r"FOCUS_PAD_Y = 96\nFOCUS_FREE_PAD_Y = 160\nFOCUS_MERGE_GAP = 64\nFOCUS_MAX_CHIPS = 2",
    '''FOCUS_PAD_X = 96
FOCUS_PAD_Y = 96
FOCUS_FREE_PAD_X = 160
FOCUS_FREE_PAD_Y = 160
FOCUS_MERGE_GAP = 64
FOCUS_MAX_CHIPS = 2
FOCUS_MAX_SOURCE_SIDE = ADAPTIVE_TILE_MAX
FOCUS_SOURCE_PIXEL_BUDGET = FOCUS_MAX_CHIPS * FOCUS_MAX_SOURCE_SIDE * FOCUS_MAX_SOURCE_SIDE
FOCUS_TENSOR_PIXEL_BUDGET = FOCUS_MAX_CHIPS * DETECTOR_INPUT_SIZE * DETECTOR_INPUT_SIZE
FOCUS_TILE_OVERLAP = 96
FOCUS_COMPLETENESS_SPAN_MIN = 0.18
FOCUS_BOUNDARY_RATIO = 0.06''',
    "focus constants",
)
adaptive = replace_once(
    adaptive,
    r"def _proposal_has_full_text\(.*?\n\ndef plan_focus_bands",
    '''def _matching_text_boxes(
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


def plan_focus_bands''',
    "completeness",
)
insert = '''\n\ndef _axis_tiles(start: int, end: int, bound: int) -> list[tuple[int, int]]:
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
'''
adaptive = adaptive.replace("\n\ndef _adaptive_detect(\n", insert + "\n\ndef _adaptive_detect(\n", 1)
adaptive = adaptive.replace('if "text_segmenter" in str(detector.source_model).lower():', 'if detector.model_role == "text_segmenter":', 1)
adaptive = replace_once(
    adaptive,
    r"def _focus_text_detect\(.*?\n\n\nclass _AdaptiveDetectorProxy",
    '''def _focus_text_detect(
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
            "focus_proposals": len(proposals), "focus_uncovered_proposals": 0,
            "focus_chip_calls": 0, "focus_source_pixels": 0,
            "focus_tensor_pixels": 0, "focus_deferred_regions": 0,
            "focus_fallback_calls": 0,
        }, []

    full_boxes = detector._detect_single_plain(image, 0, 0)
    uncovered = [proposal for proposal in proposals if not _proposal_has_full_text(proposal, full_boxes)]
    fallback_image = image if not proposals and not full_boxes else None
    chips, deferred = plan_focus_chips(h, w, uncovered, fallback_image=fallback_image)
    all_boxes = list(full_boxes)
    for x1, y1, x2, y2 in chips:
        crop = image[y1:y2, x1:x2]
        if crop.size:
            all_boxes.extend(detector._detect_single_plain(crop, x1, y1))
    boxes = detector._nms_boxes(all_boxes)
    result = [detector._with_semantics(box) for box in detector._filter_invalid(boxes, w, h)]
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
        "focus_proposals": len(proposals),
        "focus_uncovered_proposals": len(uncovered),
        "focus_chip_calls": len(chips),
        "focus_source_pixels": sum((x2-x1)*(y2-y1) for x1,y1,x2,y2 in chips),
        "focus_tensor_pixels": len(chips) * DETECTOR_INPUT_SIZE * DETECTOR_INPUT_SIZE,
        "focus_deferred_regions": len(deferred_boxes),
        "focus_fallback_calls": int(bool(fallback_image is not None and chips)),
    }, deferred_boxes


class _AdaptiveDetectorProxy''',
    "focus detector",
)
adaptive = adaptive.replace(
    "text_boxes, focus_metrics = _focus_text_detect(\n",
    "text_boxes, focus_metrics, deferred_boxes = _focus_text_detect(\n",
    1,
)
adaptive = adaptive.replace(
    "                result = super().detect(image, parallel=False)\n\n        metrics = dict",
    "                result = super().detect(image, parallel=False)\n\n        if deferred_boxes:\n            result.extend(deferred_boxes)\n\n        metrics = dict",
    1,
)
adaptive_path.write_text(adaptive, encoding="utf-8")

recovery_path = Path("app/detector/recovery.py")
recovery = recovery_path.read_text(encoding="utf-8")
recovery = recovery.replace("import threading\n", "import threading\nimport weakref\n", 1)
recovery = recovery.replace(
    "        self._mser_lock = threading.Lock()\n",
    "        self._mser_lock = threading.Lock()\n        self._primitive_local = threading.local()\n",
    1,
)
primitive_helper = '''\n    def _extract_primitives(self, image: np.ndarray, gray: np.ndarray) -> np.ndarray:
        """Extract raw MSER regions once per source-image object and worker thread."""
        state = getattr(self._primitive_local, "value", None)
        if state is not None:
            image_ref = state.get("image_ref")
            if image_ref is not None and image_ref() is image and state.get("shape") == tuple(image.shape[:2]):
                return state["boxes"]
        with self._mser_lock:
            _, boxes = self._mser.detectRegions(gray)
        if boxes is None or len(boxes) == 0:
            raw = np.empty((0, 4), dtype=np.int32)
        else:
            raw = np.asarray(boxes, dtype=np.int32).reshape(-1, 4).copy()
        self._primitive_local.value = {
            "image_ref": weakref.ref(image),
            "shape": tuple(image.shape[:2]),
            "boxes": raw,
        }
        return raw
\n'''
recovery = recovery.replace("\n    def detect(self, image: np.ndarray, existing: list[BubbleBox] | None = None) -> list[BubbleBox]:\n", primitive_helper + "    def detect(self, image: np.ndarray, existing: list[BubbleBox] | None = None) -> list[BubbleBox]:\n", 1)
recovery = replace_once(
    recovery,
    r"        with self\._mser_lock:\n            _, boxes = self\._mser\.detectRegions\(gray\)\n        if boxes is None or len\(boxes\) == 0:\n            return \[\]",
    '''        boxes = self._extract_primitives(image, gray)
        if boxes.size == 0:
            return []''',
    "MSER extraction",
)
recovery_path.write_text(recovery, encoding="utf-8")

sanity = r'''from __future__ import annotations

import sys
import threading
import types
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if "onnxruntime" not in sys.modules:
    sys.modules["onnxruntime"] = types.ModuleType("onnxruntime")

from app.detector.adaptive_focus_detector import (
    FOCUS_MAX_CHIPS,
    FOCUS_MAX_SOURCE_SIDE,
    _proposal_has_full_text,
    plan_focus_chips,
)
from app.detector.bubble_detector import BubbleBox
from app.detector.recovery import SecondaryTextRecovery


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def scheduling_checks():
    giant = BubbleBox(0, 0, 900, 4096, 0.8, semantic_type="free_text")
    chips, deferred = plan_focus_chips(4096, 900, [giant])
    check(1 <= len(chips) <= FOCUS_MAX_CHIPS, "focus call budget")
    check(bool(deferred), "giant proposal must expose deferred budget regions")
    check(all(max(x2-x1, y2-y1) <= FOCUS_MAX_SOURCE_SIDE for x1,y1,x2,y2 in chips), "giant proposal defeated source resolution bound")

    proposal = BubbleBox(100, 100, 600, 600, 0.8, semantic_type="speech_bubble")
    tiny = BubbleBox(320, 320, 330, 330, 0.8, semantic_type="text")
    check(not _proposal_has_full_text(proposal, [tiny]), "tiny presence incorrectly marked complete")
    dialogue = BubbleBox(120, 120, 360, 260, 0.8, semantic_type="speech_bubble")
    text = BubbleBox(180, 155, 300, 225, 0.8, semantic_type="text")
    check(_proposal_has_full_text(dialogue, [text]), "ordinary dialogue completeness regressed")

    image = np.full((4096, 900, 3), 128, np.uint8)
    image[2500:2750, 120:780] = (255, 0, 255)
    cv2.putText(image, "SFX", (180, 2680), cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 255, 0), 16, cv2.LINE_AA)
    fallback, fallback_deferred = plan_focus_chips(4096, 900, [], fallback_image=image)
    check(len(fallback) == 1 and not fallback_deferred, "proposal-independent fallback budget")
    check(any(y1 < 2750 and y2 > 2500 for _x1,y1,_x2,y2 in fallback), "chromatic free-text signal not selected")
    print("phase4 focus budget/completeness/fallback checks PASS")


def extraction_cache_check():
    recovery = SecondaryTextRecovery()
    calls = {"count": 0}
    class FakeMSER:
        def detectRegions(self, gray):
            calls["count"] += 1
            return [], np.array([[10, 10, 12, 8], [25, 10, 12, 8], [40, 10, 12, 8]], dtype=np.int32)
    recovery._mser = FakeMSER()
    recovery._mser_lock = threading.Lock()
    image = np.zeros((120, 160, 3), np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    first = recovery._extract_primitives(image, gray)
    second = recovery._extract_primitives(image, gray)
    check(calls["count"] == 1, "MSER primitives extracted more than once")
    check(np.array_equal(first, second), "cached primitives changed")
    other = image.copy()
    recovery._extract_primitives(other, gray)
    check(calls["count"] == 2, "cache ownership leaked across source images")
    print("phase4 MSER primitive ownership/cache check PASS")


scheduling_checks()
extraction_cache_check()
print("backend adaptive scheduling sanity: phase 4 PASS")
'''
Path("scripts/backend_phase4_sanity.py").write_text(sanity, encoding="utf-8")
print("phase4 patch applied")
