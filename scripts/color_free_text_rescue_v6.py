from __future__ import annotations

"""Branch-only colorful free-text proposal rescue for detector A/B runs.

The production detector already focuses the neural text segmenter around bubble
and MSER proposals. This experiment adds cheap chroma-sensitive MSER proposals
without granting destructive inpaint authority. The existing V4 focus path then
gets a chance to recover a verified text-segmenter mask from those proposals.
"""

import atexit
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import threading
import time

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import numpy as np

from app.detector import recovery as recovery_mod


_MODE = os.getenv("MANGA_COLOR_FREE_TEXT_RESCUE", "").strip().lower()
_MIN_CHROMA = max(0.0, float(os.getenv("MANGA_COLOR_RESCUE_MIN_CHROMA", "18")))
_REPORT_PATH = Path("benchmark-results/color-free-text-rescue-v6.json")
_LOCK = threading.Lock()
_INSTALLED = False
_ORIGINAL_DETECT = None

_STATS = {
    "mode": _MODE or None,
    "min_chroma": _MIN_CHROMA,
    "invocations": 0,
    "skipped_low_chroma": 0,
    "base_boxes": 0,
    "extra_boxes": 0,
    "total_ms": 0.0,
    "channels": {
        name: {"runs": 0, "raw_regions": 0, "proposals": 0, "ms": 0.0}
        for name in ("lab_a", "lab_b", "hsv_s", "chroma")
    },
}


def _channel_views(image: np.ndarray) -> tuple[dict[str, np.ndarray], float]:
    if image is None or image.size == 0:
        return {}, 0.0
    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        bgr = image

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(bgr)
    high = np.maximum(np.maximum(b, g), r)
    low = np.minimum(np.minimum(b, g), r)
    chroma = cv2.subtract(high, low)

    sat = hsv[:, :, 1]
    color_score = max(
        float(np.percentile(sat, 95)),
        float(np.percentile(chroma, 95)),
    )
    return {
        "lab_a": lab[:, :, 1],
        "lab_b": lab[:, :, 2],
        "hsv_s": sat,
        "chroma": chroma,
    }, color_score


def _is_duplicate(candidate, boxes) -> bool:
    for existing in boxes:
        if recovery_mod.SecondaryTextRecovery._iou(candidate, existing) >= 0.30:
            return True
    return False


def _color_detect(self, image, existing=None):
    started = time.perf_counter()
    base_existing = list(existing or [])
    base = _ORIGINAL_DETECT(self, image, existing=existing)

    views, color_score = _channel_views(image)
    extras = []
    skipped = color_score < _MIN_CHROMA

    with _LOCK:
        _STATS["invocations"] += 1
        _STATS["base_boxes"] += len(base)
        if skipped:
            _STATS["skipped_low_chroma"] += 1

    if not skipped:
        for name, channel in views.items():
            channel_started = time.perf_counter()
            with self._mser_lock:
                _, raw_boxes = self._mser.detectRegions(channel)
            raw_count = 0 if raw_boxes is None else int(len(raw_boxes))

            proposals = self._residual_line_candidates(
                raw_boxes,
                image.shape[:2],
                base_existing + list(base) + list(extras),
            )
            accepted = []
            for proposal in proposals:
                candidate = replace(
                    proposal,
                    source_model=f"color_mser_{name}",
                    class_name="color_text_recovery",
                    semantic_type="free_text",
                    mask=None,
                    mask_source="none",
                    safe_to_inpaint=False,
                    ocr_eligible=True,
                    needs_review=True,
                )
                if _is_duplicate(candidate, base_existing + list(base) + extras):
                    continue
                accepted.append(candidate)
                extras.append(candidate)

            elapsed = (time.perf_counter() - channel_started) * 1000.0
            with _LOCK:
                row = _STATS["channels"][name]
                row["runs"] += 1
                row["raw_regions"] += raw_count
                row["proposals"] += len(accepted)
                row["ms"] += elapsed

    elapsed_total = (time.perf_counter() - started) * 1000.0
    with _LOCK:
        _STATS["extra_boxes"] += len(extras)
        _STATS["total_ms"] += elapsed_total
    return list(base) + extras


def _write_report() -> None:
    try:
        with _LOCK:
            payload = json.loads(json.dumps(_STATS))
        payload["total_ms"] = round(float(payload["total_ms"]), 3)
        for row in payload["channels"].values():
            row["ms"] = round(float(row["ms"]), 3)
        _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        _REPORT_PATH.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except Exception:
        pass


def install_from_env() -> bool:
    global _INSTALLED, _ORIGINAL_DETECT
    if _INSTALLED:
        return True
    if _MODE != "lab_mser_v1":
        return False
    _ORIGINAL_DETECT = recovery_mod.SecondaryTextRecovery.detect
    recovery_mod.SecondaryTextRecovery.detect = _color_detect
    atexit.register(_write_report)
    _INSTALLED = True
    return True


def _self_test() -> None:
    # Magenta and this green have nearly the same grayscale luminance but very
    # different Lab chroma, which models the failure mode this rescue targets.
    image = np.full((180, 640, 3), (0, 180, 0), dtype=np.uint8)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.putText(mask, "COLOR TEXT", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 2.2, 255, 7, cv2.LINE_AA)
    image[mask > 0] = (255, 0, 255)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    views, score = _channel_views(image)
    text = mask > 127
    bg = mask == 0
    gray_gap = abs(float(np.mean(gray[text])) - float(np.mean(gray[bg])))
    lab_gap = abs(float(np.mean(views["lab_a"][text])) - float(np.mean(views["lab_a"][bg])))
    if gray_gap >= 20.0:
        raise AssertionError(f"synthetic grayscale gap unexpectedly high: {gray_gap}")
    if lab_gap <= 40.0:
        raise AssertionError(f"Lab-a did not expose chroma contrast: {lab_gap}")
    if score <= _MIN_CHROMA:
        raise AssertionError((score, _MIN_CHROMA))


if __name__ == "__main__":
    _self_test()
    print(json.dumps({
        "status": "pass",
        "mode": _MODE or None,
        "min_chroma": _MIN_CHROMA,
        "channels": ["lab_a", "lab_b", "hsv_s", "chroma"],
    }, indent=2))
