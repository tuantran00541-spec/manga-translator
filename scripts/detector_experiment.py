from __future__ import annotations

"""Branch-only detector experiments for reproducible GitHub Actions A/B runs.

This module deliberately monkey-patches the detector only when the E2E workflow
opts in. Production/default code paths remain unchanged until an experiment
wins the quality + speed gate and is ported cleanly.
"""

import atexit
import json
import math
import os
from pathlib import Path
import sys
import threading
import time

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.detector import bubble_detector as bubble_mod
from app.detector import combined_detector as combined_mod


_EXPERIMENT = os.getenv("MANGA_DETECTOR_EXPERIMENT", "").strip()
_TILE_MAX = max(
    bubble_mod.INPUT_SIZE,
    int(os.getenv("MANGA_DETECTOR_ADAPTIVE_TILE_MAX", "1344")),
)
_TILE_OVERLAP = max(
    0,
    int(os.getenv("MANGA_DETECTOR_ADAPTIVE_OVERLAP", "200")),
)
_FOCUS_PAD_Y = max(0, int(os.getenv("MANGA_DETECTOR_FOCUS_PAD_Y", "96")))
_FOCUS_FREE_PAD_Y = max(
    _FOCUS_PAD_Y,
    int(os.getenv("MANGA_DETECTOR_FOCUS_FREE_PAD_Y", "160")),
)
_FOCUS_MERGE_GAP = max(
    0,
    int(os.getenv("MANGA_DETECTOR_FOCUS_MERGE_GAP", "64")),
)
_FOCUS_MAX_CHIPS = max(
    1,
    int(os.getenv("MANGA_DETECTOR_FOCUS_MAX_CHIPS", "2")),
)
_REPORT_PATH = Path("benchmark-results/detector-experiment.json")

_lock = threading.Lock()
_focus_local = threading.local()
_installed = False
_original_yolo_detect = None
_original_detect_single_plain = None
_original_combined_detect = None
_stats: dict = {
    "experiment": _EXPERIMENT or None,
    "adaptive_tile_max": _TILE_MAX,
    "adaptive_overlap": _TILE_OVERLAP,
    "focus_pad_y": _FOCUS_PAD_Y,
    "focus_free_pad_y": _FOCUS_FREE_PAD_Y,
    "focus_merge_gap": _FOCUS_MERGE_GAP,
    "focus_max_chips": _FOCUS_MAX_CHIPS,
    "models": {},
    "v2": {
        "combined_invocations": 0,
        "bubble_cache_hits": 0,
        "bubble_prefetch_ms": 0.0,
        "proposal_prefetch_ms": 0.0,
        "proposal_prefetch_boxes": 0,
    },
}


def _model_row(source_model: str) -> dict:
    with _lock:
        return _stats["models"].setdefault(
            source_model,
            {
                "single_model_calls": 0,
                "single_model_ms": 0.0,
                "detect_invocations": 0,
                "tall_invocations": 0,
                "adaptive_window_calls": 0,
                "full_image_text_calls": 0,
                "adaptive_windows_total": 0,
                "adaptive_window_height_min": None,
                "adaptive_window_height_max": None,
                "focus_invocations": 0,
                "focus_proposals": 0,
                "focus_uncovered_proposals": 0,
                "focus_chip_calls": 0,
                "focus_chip_height_min": None,
                "focus_chip_height_max": None,
                "focus_no_context_fallbacks": 0,
            },
        )


def plan_adaptive_windows(
    height: int,
    *,
    tile_max: int = _TILE_MAX,
    overlap: int = _TILE_OVERLAP,
) -> list[tuple[int, int]]:
    """Plan gap-free windows while minimizing the number of model calls."""
    height = int(height)
    if height <= 0:
        return []
    tile_max = max(bubble_mod.INPUT_SIZE, int(tile_max))
    overlap = max(0, min(int(overlap), tile_max - 1))
    if height <= tile_max:
        return [(0, height)]

    stride_budget = max(1, tile_max - overlap)
    count = max(2, int(math.ceil((height - overlap) / stride_budget)))
    tile_h = int(math.ceil((height + (count - 1) * overlap) / count))
    tile_h = max(bubble_mod.INPUT_SIZE, min(tile_max, tile_h))
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


def _validate_plan(height: int, windows: list[tuple[int, int]]) -> None:
    if not windows:
        raise AssertionError(f"no windows for height={height}")
    if windows[0][0] != 0 or windows[-1][1] != height:
        raise AssertionError((height, windows))
    previous_end = 0
    for start, end in windows:
        if not (0 <= start < end <= height):
            raise AssertionError((height, windows))
        if start > previous_end:
            raise AssertionError(f"gap at {previous_end}:{start} for {height}")
        previous_end = max(previous_end, end)


def _counted_detect_single_plain(self, image, offset_x: int, offset_y: int):
    started = time.perf_counter()
    try:
        return _original_detect_single_plain(self, image, offset_x, offset_y)
    finally:
        elapsed = (time.perf_counter() - started) * 1000.0
        row = _model_row(str(getattr(self, "source_model", "unknown")))
        with _lock:
            row["single_model_calls"] += 1
            row["single_model_ms"] += elapsed


def _adaptive_detect(self, image):
    h, w = image.shape[:2]
    row = _model_row(str(getattr(self, "source_model", "unknown")))
    with _lock:
        row["detect_invocations"] += 1

    threshold = bubble_mod.INPUT_SIZE * bubble_mod.DETECTOR_TALL_IMAGE_FACTOR
    if h <= threshold:
        boxes = self._detect_single(image, 0, 0)
    else:
        windows = plan_adaptive_windows(h)
        _validate_plan(h, windows)
        all_boxes = []
        with _lock:
            row["tall_invocations"] += 1
            row["adaptive_windows_total"] += len(windows)
            row["adaptive_window_calls"] += len(windows)
            for start, end in windows:
                size = int(end - start)
                current_min = row["adaptive_window_height_min"]
                current_max = row["adaptive_window_height_max"]
                row["adaptive_window_height_min"] = (
                    size if current_min is None else min(int(current_min), size)
                )
                row["adaptive_window_height_max"] = (
                    size if current_max is None else max(int(current_max), size)
                )

        for start, end in windows:
            all_boxes.extend(self._detect_single(image[start:end, :], 0, start))

        if "text_segmenter" in str(self.source_model).lower():
            all_boxes.extend(self._detect_single_plain(image, 0, 0))
            with _lock:
                row["full_image_text_calls"] += 1

        boxes = self._nms_boxes(all_boxes)

    return [
        self._with_semantics(box)
        for box in self._filter_invalid(boxes, w, h)
    ]


def _proposal_is_free_text(box) -> bool:
    return (
        str(getattr(box, "semantic_type", "")) == "free_text"
        or str(getattr(box, "source_model", "")) == "opencv_mser"
    )


def _proposal_has_full_text(proposal, text_boxes) -> bool:
    """Return whether a speech-bubble proposal has plausible full-pass text.

    Free-text and MSER proposals always receive focused refinement because they
    are the cases most likely to disappear when a tall page is squeezed into a
    1024px full-image pass.
    """
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
        if ((ix2 - ix1) * (iy2 - iy1)) / float(text_area) >= 0.6:
            return True
    return False


def plan_focus_bands(
    height: int,
    proposals,
    *,
    max_chips: int = _FOCUS_MAX_CHIPS,
    merge_gap: int = _FOCUS_MERGE_GAP,
) -> list[tuple[int, int]]:
    """Project proposals onto Y and collapse them into a few full-width chips."""
    height = int(height)
    if height <= 0:
        return []
    intervals: list[tuple[int, int]] = []
    for proposal in proposals:
        y1 = int(getattr(proposal, "y1", 0))
        y2 = int(getattr(proposal, "y2", 0))
        if y2 <= y1:
            continue
        pad = _FOCUS_FREE_PAD_Y if _proposal_is_free_text(proposal) else _FOCUS_PAD_Y
        start = max(0, y1 - pad)
        end = min(height, y2 + pad)
        if end > start:
            intervals.append((start, end))
    if not intervals:
        return []

    intervals.sort()
    merged: list[list[int]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1] + merge_gap:
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

    return [(int(start), int(end)) for start, end in merged if end > start]


def _focus_text_detect(self, image):
    h, w = image.shape[:2]
    context = getattr(_focus_local, "value", None)
    row = _model_row(str(getattr(self, "source_model", "unknown")))

    if (
        not context
        or context.get("image_id") != id(image)
        or context.get("shape") != tuple(image.shape[:2])
    ):
        with _lock:
            row["focus_no_context_fallbacks"] += 1
        return _adaptive_detect(self, image)

    with _lock:
        row["detect_invocations"] += 1
        row["focus_invocations"] += 1

    threshold = bubble_mod.INPUT_SIZE * bubble_mod.DETECTOR_TALL_IMAGE_FACTOR
    if h <= threshold:
        boxes = self._detect_single(image, 0, 0)
        return [
            self._with_semantics(box)
            for box in self._filter_invalid(boxes, w, h)
        ]

    full_boxes = self._detect_single_plain(image, 0, 0)
    with _lock:
        row["full_image_text_calls"] += 1

    proposals = list(context.get("proposals") or [])
    uncovered = [
        proposal
        for proposal in proposals
        if not _proposal_has_full_text(proposal, full_boxes)
    ]
    bands = plan_focus_bands(h, uncovered)

    with _lock:
        row["focus_proposals"] += len(proposals)
        row["focus_uncovered_proposals"] += len(uncovered)

    all_boxes = list(full_boxes)
    for start, end in bands:
        crop = image[start:end, :]
        if crop.size == 0:
            continue
        all_boxes.extend(self._detect_single_plain(crop, 0, start))
        size = int(end - start)
        with _lock:
            row["focus_chip_calls"] += 1
            current_min = row["focus_chip_height_min"]
            current_max = row["focus_chip_height_max"]
            row["focus_chip_height_min"] = (
                size if current_min is None else min(int(current_min), size)
            )
            row["focus_chip_height_max"] = (
                size if current_max is None else max(int(current_max), size)
            )

    boxes = self._nms_boxes(all_boxes)
    return [
        self._with_semantics(box)
        for box in self._filter_invalid(boxes, w, h)
    ]


def _dispatch_yolo_detect(self, image):
    source_name = str(getattr(self, "source_model", "")).lower()
    if _EXPERIMENT == "adaptive_focus_v2":
        context = getattr(_focus_local, "value", None)
        if (
            "bubble" in source_name
            and context
            and context.get("image_id") == id(image)
            and context.get("shape") == tuple(image.shape[:2])
            and not context.get("bubble_cache_used")
        ):
            context["bubble_cache_used"] = True
            with _lock:
                _stats["v2"]["bubble_cache_hits"] += 1
            return list(context.get("bubble_boxes") or [])
        if "text_segmenter" in source_name:
            return _focus_text_detect(self, image)
    return _adaptive_detect(self, image)


def _v2_combined_detect(self, image, *, parallel: bool = False):
    """Prefetch proposal context once, then reuse production post-processing."""
    bubble_started = time.perf_counter()
    bubble_boxes = _adaptive_detect(self.bubble_detector, image)
    bubble_ms = (time.perf_counter() - bubble_started) * 1000.0

    proposal_started = time.perf_counter()
    recovery_boxes = self.recovery.detect(image, existing=bubble_boxes)
    proposal_ms = (time.perf_counter() - proposal_started) * 1000.0

    context = {
        "image_id": id(image),
        "shape": tuple(image.shape[:2]),
        "bubble_boxes": list(bubble_boxes),
        "bubble_cache_used": False,
        "proposals": list(bubble_boxes) + list(recovery_boxes),
    }
    _focus_local.value = context
    try:
        result = _original_combined_detect(self, image, parallel=False)
    finally:
        _focus_local.value = None

    metrics = dict(getattr(self._metrics_local, "value", {}) or {})
    metrics["bubble_model_ms"] = round(bubble_ms, 3)
    metrics["focus_prefetch_mser_ms"] = round(proposal_ms, 3)
    metrics["focus_prefetch_proposals"] = len(recovery_boxes)
    metrics["mser_ms"] = round(float(metrics.get("mser_ms", 0.0)) + proposal_ms, 3)
    metrics["total_ms"] = round(
        float(metrics.get("total_ms", 0.0)) + bubble_ms + proposal_ms,
        3,
    )
    self._metrics_local.value = metrics

    with _lock:
        _stats["v2"]["combined_invocations"] += 1
        _stats["v2"]["bubble_prefetch_ms"] += bubble_ms
        _stats["v2"]["proposal_prefetch_ms"] += proposal_ms
        _stats["v2"]["proposal_prefetch_boxes"] += len(recovery_boxes)
    return result


def _write_report() -> None:
    try:
        with _lock:
            payload = json.loads(json.dumps(_stats))
        for row in payload.get("models", {}).values():
            row["single_model_ms"] = round(float(row["single_model_ms"]), 3)
        for key in ("bubble_prefetch_ms", "proposal_prefetch_ms"):
            payload["v2"][key] = round(float(payload["v2"].get(key, 0.0)), 3)
        _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        _REPORT_PATH.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except Exception:
        pass


def install_from_env() -> bool:
    global _installed
    global _original_yolo_detect
    global _original_detect_single_plain
    global _original_combined_detect

    if _installed:
        return True
    if _EXPERIMENT not in {"adaptive_tiles_v1", "adaptive_focus_v2"}:
        return False

    _original_yolo_detect = bubble_mod.YoloDetector.detect
    _original_detect_single_plain = bubble_mod.YoloDetector._detect_single_plain
    _original_combined_detect = combined_mod.CombinedTextDetector.detect

    bubble_mod.YoloDetector._detect_single_plain = _counted_detect_single_plain
    bubble_mod.YoloDetector.detect = _dispatch_yolo_detect
    if _EXPERIMENT == "adaptive_focus_v2":
        combined_mod.CombinedTextDetector.detect = _v2_combined_detect

    atexit.register(_write_report)
    _installed = True
    return True


def _self_test() -> None:
    threshold = int(
        bubble_mod.INPUT_SIZE * bubble_mod.DETECTOR_TALL_IMAGE_FACTOR
    )
    assert threshold == 1536
    cases = {
        1537: 2,
        2400: 2,
        4096: 4,
    }
    for height, expected_calls in cases.items():
        windows = plan_adaptive_windows(height)
        _validate_plan(height, windows)
        if len(windows) != expected_calls:
            raise AssertionError(
                f"height={height}: expected {expected_calls} windows, got {windows}"
            )
        if any((end - start) > _TILE_MAX for start, end in windows):
            raise AssertionError((height, windows))
    assert plan_adaptive_windows(2400) == [(0, 1300), (1100, 2400)]

    if _EXPERIMENT == "adaptive_focus_v2":
        proposals = [
            bubble_mod.BubbleBox(10, 100, 200, 250, 0.9, semantic_type="speech_bubble"),
            bubble_mod.BubbleBox(10, 700, 200, 850, 0.9, semantic_type="free_text"),
            bubble_mod.BubbleBox(
                10, 2000, 200, 2150, 0.9,
                source_model="opencv_mser", semantic_type="text",
            ),
        ]
        bands = plan_focus_bands(2400, proposals, max_chips=2)
        if not (1 <= len(bands) <= 2):
            raise AssertionError(bands)
        if any(not (0 <= start < end <= 2400) for start, end in bands):
            raise AssertionError(bands)


if __name__ == "__main__":
    _self_test()
    payload = {
        "status": "pass",
        "experiment": _EXPERIMENT or None,
        "tile_max": _TILE_MAX,
        "overlap": _TILE_OVERLAP,
        "plans": {
            str(height): plan_adaptive_windows(height)
            for height in (1537, 2400, 4096)
        },
    }
    if _EXPERIMENT == "adaptive_focus_v2":
        payload["focus"] = {
            "pad_y": _FOCUS_PAD_Y,
            "free_pad_y": _FOCUS_FREE_PAD_Y,
            "merge_gap": _FOCUS_MERGE_GAP,
            "max_chips": _FOCUS_MAX_CHIPS,
        }
    print(json.dumps(payload, indent=2))
