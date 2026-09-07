from __future__ import annotations

"""Branch-only color-normalized focus TTA for detector A/B runs.

Run #16 showed that full-page multi-channel MSER produces many noisy proposals
without increasing verified text masks. This follow-up keeps the V4 proposal set
unchanged and only gives the existing text segmenter extra transformed views for
proposal-local chips whose content has meaningful chroma variation.

The workflow intentionally keeps the historical environment variable and report
filename so this experiment can replace the previous branch-only hook without
adding another heavy workflow or changing production code.
"""

import atexit
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

from scripts import detector_experiment as detector_exp


_MODE = os.getenv("MANGA_COLOR_FREE_TEXT_RESCUE", "").strip().lower()
_MIN_CHROMA = max(0.0, float(os.getenv("MANGA_COLOR_RESCUE_MIN_CHROMA", "18")))
_REPORT_PATH = Path("benchmark-results/color-free-text-rescue-v6.json")
_LOCK = threading.Lock()
_INSTALLED = False
_ORIGINAL_FOCUS = None
_PAD_X = 96
_PAD_Y = 96
_MAX_CHIPS = 4

_STATS = {
    "mode": _MODE or None,
    "strategy": "proposal_local_color_tta_v1",
    "min_chroma": _MIN_CHROMA,
    "invocations": 0,
    "skipped_low_chroma": 0,
    "base_boxes": 0,
    "extra_boxes": 0,
    "eligible_proposals": 0,
    "chips": 0,
    "total_ms": 0.0,
    "best_channel_counts": {},
    "channels": {
        name: {"runs": 0, "raw_regions": 0, "proposals": 0, "ms": 0.0}
        for name in ("clahe_l", "best_chroma")
    },
}


def _as_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def _channel_bank(image: np.ndarray) -> dict[str, np.ndarray]:
    bgr = _as_bgr(image)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(bgr)
    high = np.maximum(np.maximum(b, g), r)
    low = np.minimum(np.minimum(b, g), r)
    return {
        "lab_a": lab[:, :, 1],
        "lab_b": lab[:, :, 2],
        "hsv_s": hsv[:, :, 1],
        "chroma": cv2.subtract(high, low),
    }


def _robust_spread(channel: np.ndarray) -> float:
    if channel.size == 0:
        return 0.0
    p05, p95 = np.percentile(channel, (5, 95))
    return max(0.0, float(p95) - float(p05))


def _color_score(image: np.ndarray) -> tuple[float, str | None]:
    bank = _channel_bank(image)
    if not bank:
        return 0.0, None
    spreads = {name: _robust_spread(channel) for name, channel in bank.items()}
    name = max(spreads, key=spreads.get)
    return float(spreads[name]), name


def _clahe_l_view(image: np.ndarray) -> np.ndarray:
    bgr = _as_bgr(image)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    enhanced = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((enhanced, a, b)), cv2.COLOR_LAB2BGR)


def _best_chroma_view(image: np.ndarray) -> tuple[np.ndarray, str, float]:
    bank = _channel_bank(image)
    spreads = {name: _robust_spread(channel) for name, channel in bank.items()}
    name = max(spreads, key=spreads.get)
    channel = bank[name]
    p05, p95 = np.percentile(channel, (5, 95))
    spread = max(0.0, float(p95) - float(p05))
    if spread < 1.0:
        normalized = channel.copy()
    else:
        normalized = np.clip(
            (channel.astype(np.float32) - float(p05)) * (255.0 / spread),
            0.0,
            255.0,
        ).astype(np.uint8)
    normalized = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(normalized)
    return cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR), name, spread


def _iou(a, b) -> float:
    x1 = max(int(getattr(a, "x1", 0)), int(getattr(b, "x1", 0)))
    y1 = max(int(getattr(a, "y1", 0)), int(getattr(b, "y1", 0)))
    x2 = min(int(getattr(a, "x2", 0)), int(getattr(b, "x2", 0)))
    y2 = min(int(getattr(a, "y2", 0)), int(getattr(b, "y2", 0)))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    aa = max(1, (int(getattr(a, "x2", 0)) - int(getattr(a, "x1", 0))) * (int(getattr(a, "y2", 0)) - int(getattr(a, "y1", 0))))
    bb = max(1, (int(getattr(b, "x2", 0)) - int(getattr(b, "x1", 0))) * (int(getattr(b, "y2", 0)) - int(getattr(b, "y1", 0))))
    return inter / float(aa + bb - inter)


def _novel_count(candidates, existing) -> int:
    count = 0
    for candidate in candidates:
        if not any(_iou(candidate, box) >= 0.30 for box in existing):
            count += 1
    return count


def _proposal_chip(image: np.ndarray, proposal) -> tuple[np.ndarray, int, int] | None:
    h, w = image.shape[:2]
    x1 = max(0, int(getattr(proposal, "x1", 0)) - _PAD_X)
    y1 = max(0, int(getattr(proposal, "y1", 0)) - _PAD_Y)
    x2 = min(w, int(getattr(proposal, "x2", 0)) + _PAD_X)
    y2 = min(h, int(getattr(proposal, "y2", 0)) + _PAD_Y)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop, x1, y1


def _dedupe_chips(chips: list[tuple[np.ndarray, int, int, object]]) -> list[tuple[np.ndarray, int, int, object]]:
    kept = []
    for item in chips:
        _, x, y, proposal = item
        duplicate = False
        for _, ox, oy, other in kept:
            ax1, ay1 = x, y
            ax2 = x + int(item[0].shape[1])
            ay2 = y + int(item[0].shape[0])
            bx1, by1 = ox, oy
            bx2 = ox + int(_[0].shape[1]) if False else 0
            _ = other
            # Proposal-level overlap is enough here; duplicate focus proposals
            # usually describe the same text line with slightly different pads.
            if _iou(proposal, other) >= 0.45:
                duplicate = True
                break
        if not duplicate:
            kept.append(item)
        if len(kept) >= _MAX_CHIPS:
            break
    return kept


def _focus_tta_detect(self, image):
    base = _ORIGINAL_FOCUS(self, image)
    context = getattr(detector_exp._focus_local, "value", None)
    if (
        not context
        or context.get("image_id") != id(image)
        or context.get("shape") != tuple(image.shape[:2])
    ):
        return base

    started = time.perf_counter()
    proposals = list(context.get("proposals") or [])
    candidate_chips: list[tuple[np.ndarray, int, int, object]] = []

    for proposal in proposals:
        raw = _proposal_chip(image, proposal)
        if raw is None:
            continue
        crop, x, y = raw
        px1 = max(0, int(getattr(proposal, "x1", 0)))
        py1 = max(0, int(getattr(proposal, "y1", 0)))
        px2 = min(image.shape[1], int(getattr(proposal, "x2", 0)))
        py2 = min(image.shape[0], int(getattr(proposal, "y2", 0)))
        score_crop = image[py1:py2, px1:px2]
        if score_crop.size == 0:
            score_crop = crop
        score, _ = _color_score(score_crop)
        if score < _MIN_CHROMA:
            continue
        candidate_chips.append((crop, x, y, proposal))

    candidate_chips = _dedupe_chips(candidate_chips)
    tta_boxes = []

    with _LOCK:
        _STATS["invocations"] += 1
        _STATS["base_boxes"] += len(base)
        _STATS["eligible_proposals"] += len(candidate_chips)
        if not candidate_chips:
            _STATS["skipped_low_chroma"] += 1

    for crop, offset_x, offset_y, _proposal in candidate_chips:
        with _LOCK:
            _STATS["chips"] += 1

        views = [("clahe_l", _clahe_l_view(crop), None)]
        chroma_view, best_name, _spread = _best_chroma_view(crop)
        views.append(("best_chroma", chroma_view, best_name))

        for view_name, view, best_name in views:
            view_started = time.perf_counter()
            boxes = self._detect_single_plain(view, offset_x, offset_y)
            elapsed = (time.perf_counter() - view_started) * 1000.0
            novel = _novel_count(boxes, list(base) + list(tta_boxes))
            tta_boxes.extend(boxes)
            with _LOCK:
                row = _STATS["channels"][view_name]
                row["runs"] += 1
                row["raw_regions"] += len(boxes)
                row["proposals"] += novel
                row["ms"] += elapsed
                if best_name:
                    counts = _STATS["best_channel_counts"]
                    counts[best_name] = int(counts.get(best_name, 0)) + 1

    if not tta_boxes:
        with _LOCK:
            _STATS["total_ms"] += (time.perf_counter() - started) * 1000.0
        return base

    merged = self._nms_boxes(list(base) + tta_boxes)
    h, w = image.shape[:2]
    result = [
        self._with_semantics(box)
        for box in self._filter_invalid(merged, w, h)
    ]
    with _LOCK:
        _STATS["extra_boxes"] += max(0, len(result) - len(base))
        _STATS["total_ms"] += (time.perf_counter() - started) * 1000.0
    return result


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
    global _INSTALLED, _ORIGINAL_FOCUS
    if _INSTALLED:
        return True
    if _MODE != "lab_mser_v1":
        return False
    if detector_exp._EXPERIMENT != "adaptive_focus_v2":
        return False
    _ORIGINAL_FOCUS = detector_exp._focus_text_detect
    detector_exp._focus_text_detect = _focus_tta_detect
    atexit.register(_write_report)
    _INSTALLED = True
    return True


def _self_test() -> None:
    # Synthetic high-chroma / low-luminance-contrast text. The chroma-derived
    # view should expose a much larger robust spread than grayscale luminance.
    image = np.full((180, 640, 3), (0, 180, 0), dtype=np.uint8)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.putText(mask, "COLOR TEXT", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 2.2, 255, 7, cv2.LINE_AA)
    image[mask > 0] = (255, 0, 255)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = mask > 127
    bg = mask == 0
    gray_gap = abs(float(np.mean(gray[text])) - float(np.mean(gray[bg])))
    score, best_name = _color_score(image)
    clahe = _clahe_l_view(image)
    chroma, chosen, spread = _best_chroma_view(image)
    if gray_gap >= 20.0:
        raise AssertionError(f"synthetic grayscale gap unexpectedly high: {gray_gap}")
    if score <= _MIN_CHROMA or spread <= _MIN_CHROMA:
        raise AssertionError((score, spread, _MIN_CHROMA))
    if clahe.shape != image.shape or chroma.shape != image.shape:
        raise AssertionError((clahe.shape, chroma.shape, image.shape))
    if best_name != chosen:
        raise AssertionError((best_name, chosen))


if __name__ == "__main__":
    _self_test()
    print(json.dumps({
        "status": "pass",
        "mode": _MODE or None,
        "strategy": "proposal_local_color_tta_v1",
        "min_chroma": _MIN_CHROMA,
        "channels": ["clahe_l", "best_chroma"],
    }, indent=2))
