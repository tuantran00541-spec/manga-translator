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
import threading
import time

from app.detector import bubble_detector as bubble_mod


_EXPERIMENT = os.getenv("MANGA_DETECTOR_EXPERIMENT", "").strip()
_TILE_MAX = max(
    bubble_mod.INPUT_SIZE,
    int(os.getenv("MANGA_DETECTOR_ADAPTIVE_TILE_MAX", "1344")),
)
_TILE_OVERLAP = max(
    0,
    int(os.getenv("MANGA_DETECTOR_ADAPTIVE_OVERLAP", "200")),
)
_REPORT_PATH = Path("benchmark-results/detector-experiment.json")

_lock = threading.Lock()
_installed = False
_original_detect = None
_original_detect_single_plain = None
_stats: dict = {
    "experiment": _EXPERIMENT or None,
    "adaptive_tile_max": _TILE_MAX,
    "adaptive_overlap": _TILE_OVERLAP,
    "models": {},
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
            },
        )


def plan_adaptive_windows(
    height: int,
    *,
    tile_max: int = _TILE_MAX,
    overlap: int = _TILE_OVERLAP,
) -> list[tuple[int, int]]:
    """Plan gap-free windows while minimizing the number of model calls.

    For a 2400px core with tile_max=1344 and overlap=200 this yields two
    1300px windows rather than the legacy 1024px windows at 0/824/1648.
    """
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

        # V1 intentionally preserves the legacy text-segmenter full-image pass.
        # This isolates the quality/speed effect of adaptive window planning.
        if "text_segmenter" in str(self.source_model).lower():
            all_boxes.extend(self._detect_single_plain(image, 0, 0))
            with _lock:
                row["full_image_text_calls"] += 1

        boxes = self._nms_boxes(all_boxes)

    return [
        self._with_semantics(box)
        for box in self._filter_invalid(boxes, w, h)
    ]


def _write_report() -> None:
    try:
        with _lock:
            payload = json.loads(json.dumps(_stats))
        for row in payload.get("models", {}).values():
            row["single_model_ms"] = round(float(row["single_model_ms"]), 3)
        _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        _REPORT_PATH.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except Exception:
        pass


def install_from_env() -> bool:
    global _installed, _original_detect, _original_detect_single_plain
    if _installed:
        return True
    if _EXPERIMENT != "adaptive_tiles_v1":
        return False

    _original_detect = bubble_mod.YoloDetector.detect
    _original_detect_single_plain = bubble_mod.YoloDetector._detect_single_plain
    bubble_mod.YoloDetector._detect_single_plain = _counted_detect_single_plain
    bubble_mod.YoloDetector.detect = _adaptive_detect
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
    windows_2400 = plan_adaptive_windows(2400)
    assert windows_2400 == [(0, 1300), (1100, 2400)]


if __name__ == "__main__":
    _self_test()
    print(
        json.dumps(
            {
                "status": "pass",
                "experiment": "adaptive_tiles_v1",
                "tile_max": _TILE_MAX,
                "overlap": _TILE_OVERLAP,
                "plans": {
                    str(height): plan_adaptive_windows(height)
                    for height in (1537, 2400, 4096)
                },
            },
            indent=2,
        )
    )
