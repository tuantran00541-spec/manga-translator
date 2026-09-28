"""Clean a whole chapter as one strip: detect top-down with carried windows, erase per block, cut slices last."""
from __future__ import annotations

import cv2
import numpy as np

import ctd_mask
from app.detector.kiuyha_detector import _text_box

WIN, OVERLAP, BAND_PAD, CUT_TARGET = 2400, 128, 256, 2400


def page_height(shape, width: int) -> int:
    h, w = shape[:2]
    return h if w == width else int(round(h * width / w))


def build_strip(images: list[np.ndarray], width: int) -> np.ndarray:
    """All pages stacked at one width."""
    rows = [im if im.shape[1] == width else cv2.resize(im, (width, page_height(im.shape, width)), interpolation=cv2.INTER_AREA)
            for im in images]
    return np.ascontiguousarray(np.vstack(rows))


def flow_boxes(strip: np.ndarray, det) -> list[tuple[int, int, int, int, float]]:
    """Boxes over the strip; a block touching a window's bottom is left for the next window, which starts at its top."""
    H = strip.shape[0]
    found, y = [], 0
    while y < H:
        end = min(H, y + WIN)
        boxes = det.detect_slice(np.ascontiguousarray(strip[y:end]))
        nxt = end - OVERLAP if end < H else H
        cut = [b for b in boxes if end < H and b[3] >= end - y - 4]
        for b in cut:
            nxt = min(nxt, y + max(0, b[1] - 16))
        if end < H:
            nxt = max(nxt, y + WIN // 4)
        for x1, y1, x2, y2, s in boxes:
            if (x1, y1, x2, y2, s) in cut or (end < H and y + y1 >= nxt):
                continue  # the next window sees it whole
            found.append((x1, y + y1, x2, y + y2, s))
        y = nxt
    return _dedupe(found)


def _dedupe(boxes):
    """Boxes seen by two windows are one block."""
    out = []
    for b in sorted(boxes, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1])):
        if not any(_overlap(b, o) > 0.6 for o in out):
            out.append(b)
    return sorted(out, key=lambda b: b[1])


def _overlap(a, b) -> float:
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return iw * ih / max(1, small)


def _bands(boxes, H):
    """Row ranges that hold nearby blocks together, padded for inpaint context."""
    bands = []
    for b in boxes:
        y1, y2 = max(0, b[1] - BAND_PAD), min(H, b[3] + BAND_PAD)
        if bands and y1 <= bands[-1][1]:
            bands[-1][1] = max(bands[-1][1], y2)
            bands[-1][2].append(b)
        else:
            bands.append([y1, y2, [b]])
    return bands


def _erase(band: np.ndarray, boxes, inpainter):
    masked = []
    for x1, y1, x2, y2, s in boxes:
        (X1, Y1, X2, Y2), m = ctd_mask.letter_mask(band, (x1, y1, x2, y2))
        if m.any():
            masked.append(_text_box(X1, Y1, X2, Y2, s, m.astype(np.uint8) * 255, "ctd-flow"))
    return inpainter.inpaint(band, masked) if masked else band


def clean(strip: np.ndarray, det, inpainter, passes: int = 2):
    """Clean strip and the blocks that were erased."""
    boxes = flow_boxes(strip, det)
    out = strip.copy()
    for y1, y2, group in _bands(boxes, strip.shape[0]):
        local = [(x1, by1 - y1, x2, by2 - y1, s) for x1, by1, x2, by2, s in group]
        band = _erase(np.ascontiguousarray(strip[y1:y2]), local, inpainter)
        for _ in range(passes - 1):  # text still seen inside an erased block gets one more pass
            left = [b for b in det.detect_slice(band)
                    if any(t[0] <= (b[0] + b[2]) / 2 <= t[2] and t[1] <= (b[1] + b[3]) / 2 <= t[3] for t in local)]
            if not left:
                break
            band = _erase(band, left, inpainter)
        out[y1:y2] = band
    return out, boxes


def cut_rows(strip: np.ndarray, boxes) -> list[int]:
    """Slice edges near the target height on the calmest row that crosses no block."""
    H = strip.shape[0]
    busy = np.zeros(H, bool)
    for _, y1, _, y2, _ in boxes:
        busy[max(0, y1 - 24):y2 + 24] = True
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY).astype(np.float32)
    detail = np.abs(np.diff(gray, axis=1)).mean(1)
    cuts, y = [0], 0
    while H - y > CUT_TARGET * 1.5:
        lo, hi = y + CUT_TARGET // 2, min(H - CUT_TARGET // 2, y + CUT_TARGET * 3 // 2)
        rows = [r for r in range(lo, hi) if not busy[r]]
        r = min(rows, key=lambda r: detail[r] + abs(r - y - CUT_TARGET) * 1e-3) if rows else hi
        cuts.append(r)
        y = r
    return cuts + [H]
