"""Erase masks from the fast CTD letters, grown into outline and glow that differ from the background."""
from __future__ import annotations

import os

import cv2
import numpy as np
import onnxruntime as ort

THRESHOLD, REACH, OUTLINE, EDGE = 0.3, 1.0, 0.12, 60.0
_opts = ort.SessionOptions()
_opts.intra_op_num_threads = int(os.getenv("CTD_THREADS", "4"))
SESSION = ort.InferenceSession(os.getenv("CTD_MODEL", "models/ctd_seg.onnx"), _opts, providers=["CPUExecutionProvider"])


def _prob(img: np.ndarray, scale: float) -> np.ndarray:
    h, w = img.shape[:2]
    im = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale != 1 else img
    H, W = im.shape[:2]
    im = cv2.copyMakeBorder(im, 0, (-H) % 64, 0, (-W) % 64, cv2.BORDER_REFLECT)
    x = (cv2.cvtColor(im, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None] / 255).astype(np.float32)
    p = SESSION.run(None, {"images": x})[0][0, 0][:H, :W]
    return cv2.resize(p, (w, h)) if scale != 1 else p


def text_size(p: np.ndarray) -> int:
    _, _, st, _ = cv2.connectedComponentsWithStats(p.astype(np.uint8))
    hs = st[1:, cv2.CC_STAT_HEIGHT]
    return int(np.median(hs[hs >= 6])) if (hs >= 6).any() else 0


def grow(img, seed, box, ring, reach):
    """Grow letters into pixels unlike the background round the box; far growth that runs off or keeps going is art."""
    y1, y2, x1, x2 = box
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    bgpx = lab[ring]
    if len(bgpx) < 20 or reach < 1:
        return seed
    _, _, centers = cv2.kmeans(bgpx, min(4, len(bgpx)), None,
                               (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 2, cv2.KMEANS_PP_CENTERS)

    def dist(a):
        return np.min(np.linalg.norm(a[..., None, :] - centers[None], axis=-1), -1)

    thr = np.percentile(dist(bgpx), 90) + 6
    sub = lab[y1:y2, x1:x2]
    L = np.ascontiguousarray(sub[..., 0])
    smooth = np.hypot(cv2.Sobel(L, cv2.CV_32F, 1, 0), cv2.Sobel(L, cv2.CV_32F, 0, 1)) < EDGE  # glow fades, art has edges
    nonbg = (dist(sub.reshape(-1, 3)).reshape(sub.shape[:2]) > thr) & smooth
    r = max(2, int(reach * OUTLINE))
    outline = cv2.dilate(seed[y1:y2, x1:x2].astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)) > 0
    start = outline  # letters keep an outline even when it matches the background
    k3 = np.ones((3, 3), np.uint8)

    def spread(region, steps):
        for _ in range(steps):
            nxt = (cv2.dilate(region.astype(np.uint8), k3) > 0) & (nonbg | region)
            if (nxt == region).all():
                break
            region = nxt
        return region

    region = spread(start, reach)
    further = spread(region, reach) & ~region
    d = cv2.distanceTransform((~start).astype(np.uint8), cv2.DIST_L2, 3)
    n, parts = cv2.connectedComponents((region & ~start & (d > max(2, reach * 0.4))).astype(np.uint8))
    for i in range(1, n):
        part = parts == i
        edge = part[:2].any() or part[-2:].any() or part[:, :2].any() or part[:, -2:].any()
        touch = cv2.dilate(part.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        if edge or (further & touch).sum() > 0.3 * part.sum():
            region &= ~part
    out = seed.copy()
    out[y1:y2, x1:x2] = region
    return out


def letter_mask(image: np.ndarray, box):
    """Padded box and the erase mask of its letters, outline and glow."""
    x1, y1, x2, y2 = box
    H, W = image.shape[:2]
    pad = max(8, (y2 - y1) // 4)
    X1, Y1, X2, Y2 = max(0, x1 - pad), max(0, y1 - pad), min(W, x2 + pad), min(H, y2 + pad)
    crop = image[Y1:Y2, X1:X2]
    if min(crop.shape[:2]) < 8:
        return (X1, Y1, X2, Y2), np.zeros(crop.shape[:2], bool)
    raw = np.maximum(_prob(crop, 1.0), _prob(crop, 0.5)) > THRESHOLD
    e = 6
    ey1, ex1, ey2, ex2 = max(0, Y1 - e), max(0, X1 - e), min(H, Y2 + e), min(W, X2 + e)
    wide = image[ey1:ey2, ex1:ex2]
    ring = np.ones(wide.shape[:2], bool)
    ring[Y1 - ey1:Y2 - ey1, X1 - ex1:X2 - ex1] = False
    seed = np.zeros(wide.shape[:2], bool)
    seed[Y1 - ey1:Y2 - ey1, X1 - ex1:X2 - ex1] = raw
    if not ring.any():
        return (X1, Y1, X2, Y2), raw
    grown = grow(wide, seed, (Y1 - ey1, Y2 - ey1, X1 - ex1, X2 - ex1), ring, int(text_size(raw) * REACH))
    return (X1, Y1, X2, Y2), grown[Y1 - ey1:Y2 - ey1, X1 - ex1:X2 - ex1]
