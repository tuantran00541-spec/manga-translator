"""Erase masks from comic-text-detector letters, grown into outline and glow that differ from the background."""
from __future__ import annotations

import os
import threading

import cv2
import numpy as np

from app.config import CTD_MODEL

THRESHOLD = 0.3  # letter probability that counts as ink
SCALES = (1.0, 0.5)  # the half-size pass catches very large lettering
STRIDE = 64  # the model's input sides must be multiples of this
PAD_SHARE = 0.25  # room round a box, as a share of its height, for letters its edge cuts
REACH = 1.0  # how far, in letter heights, outline and glow may spread from the letters
OUTLINE = 0.12  # share of the reach every letter keeps as outline, even one matching the background
EDGE = 60.0  # Lab lightness gradient above which a pixel is drawn art, not fading glow
RING = 6  # width of the band past the crop where the background colours are read
LEFT_SHARE = 0.004  # share of an erased block the model may still read before it gets another pass
REACH_ROUNDS = 3  # times a box grows toward letters its edge still cuts
CHAIN = 0.6  # widest gap, in letter heights, between letters of one text

_session = None
_lock = threading.Lock()


def _get_session():
    global _session
    with _lock:
        if _session is None:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = max(1, os.cpu_count() or 1)
            _session = ort.InferenceSession(str(CTD_MODEL), opts, providers=["CPUExecutionProvider"])
    return _session


def _prob(img: np.ndarray, scale: float) -> np.ndarray:
    h, w = img.shape[:2]
    im = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale != 1 else img
    ih, iw = im.shape[:2]
    im = cv2.copyMakeBorder(im, 0, (-ih) % STRIDE, 0, (-iw) % STRIDE, cv2.BORDER_REFLECT)
    x = (cv2.cvtColor(im, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None] / 255).astype(np.float32)
    p = _get_session().run(None, {"images": x})[0][0, 0][:ih, :iw]
    return cv2.resize(p, (w, h)) if scale != 1 else p


def letters(img: np.ndarray) -> np.ndarray:
    """Pixels the model reads as letters, at full and half size."""
    return np.max([_prob(img, s) for s in SCALES], axis=0) > THRESHOLD


def text_size(part: np.ndarray) -> int:
    """Median letter height of a letter mask."""
    _, _, stats, _ = cv2.connectedComponentsWithStats(part.astype(np.uint8))
    heights = stats[1:, cv2.CC_STAT_HEIGHT]
    heights = heights[heights >= 6]
    return int(np.median(heights)) if len(heights) else 0


def grow(img: np.ndarray, seed: np.ndarray, bg: np.ndarray, reach: int) -> np.ndarray:
    """Letters grown by an outline, then into smooth pixels unlike the background colours; growth that runs off is art."""
    if reach < 1 or not seed.any() or len(bg) < 20:
        return seed
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg = bg.astype(np.float32)
    _, _, centers = cv2.kmeans(bg, min(4, len(bg)), None,
                               (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 2, cv2.KMEANS_PP_CENTERS)

    def dist(a):
        # One centre at a time: the same distances without a pixels x centres x channels array.
        nearest = None
        for centre in centers:
            diff = a - centre
            square = np.einsum("...k,...k->...", diff, diff)
            nearest = square if nearest is None else np.minimum(nearest, square)
        return np.sqrt(nearest)

    smooth = np.hypot(cv2.Sobel(lab[..., 0], cv2.CV_32F, 1, 0), cv2.Sobel(lab[..., 0], cv2.CV_32F, 0, 1)) < EDGE
    open_ = ((dist(lab.reshape(-1, 3)).reshape(lab.shape[:2]) > np.percentile(dist(bg), 90) + 6) & smooth) | seed
    r = max(2, int(reach * OUTLINE))
    start = cv2.dilate(seed.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)) > 0
    k3 = np.ones((3, 3), np.uint8)

    def spread(region, steps):
        for _ in range(steps):
            nxt = (cv2.dilate(region.astype(np.uint8), k3) > 0) & (open_ | region)
            if (nxt == region).all():
                break
            region = nxt
        return region

    region = spread(start, reach)
    further = spread(region, reach) & ~region
    far = cv2.distanceTransform((~start).astype(np.uint8), cv2.DIST_L2, 3) > max(2, reach * 0.4)
    n, parts = cv2.connectedComponents((region & ~start & far).astype(np.uint8))
    for i in range(1, n):
        part = parts == i
        edge = part[:2].any() or part[-2:].any() or part[:, :2].any() or part[:, -2:].any()
        near = cv2.dilate(part.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        if edge or (further & near).sum() > 0.3 * part.sum():
            region &= ~part
    return region


SPECK = 0.35  # letters shorter than this share of the text size (dots, accents, noise) never lead growth


def _parts(seed: np.ndarray):
    """Component labels of the letters, and which of them are letter-sized rather than specks."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(seed.astype(np.uint8))
    size = text_size(seed)
    big = np.zeros(n, bool)
    big[1:] = stats[1:, cv2.CC_STAT_HEIGHT] >= max(4, SPECK * size)
    return labels, big, size


def _chained(seed: np.ndarray, inside: np.ndarray) -> np.ndarray:
    """Letters in the first box, letter-sized ones linked to them by gaps under CHAIN letter heights, and specks beside those."""
    labels, big, size = _parts(seed)
    if len(big) <= 1:
        return seed
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * max(4, int(CHAIN * size)) + 1,) * 2)
    keep = np.zeros(len(big), bool)
    keep[np.unique(labels[inside & seed])] = True
    keep[0] = False
    while True:
        near = cv2.dilate(np.isin(labels, np.nonzero(keep & big)[0]).astype(np.uint8), kernel) > 0
        grown = keep.copy()
        grown[np.unique(labels[near & seed])] = True
        grown[0] = False
        # Specks join the text beside them but never lead it further.
        if (grown & big == keep & big).all():
            return np.isin(labels, np.nonzero(grown)[0])
        keep = grown


def _cut(seed: np.ndarray) -> tuple[bool, bool, bool, bool]:
    """Which sides of the crop cut a letter-sized letter: left, top, right, bottom."""
    labels, big, _ = _parts(seed)
    edge = {side: np.unique(part) for side, part in
            (("l", labels[:, :2]), ("t", labels[:2]), ("r", labels[:, -2:]), ("b", labels[-2:]))}
    return tuple(bool(big[edge[side]].any()) for side in "ltrb")


def _reach_cut_letters(image: np.ndarray, first: tuple[int, int, int, int], step: int):
    """Grow the crop while its edge cuts letters, keeping only the text the first box holds; returns crop and seed."""
    h, w = image.shape[:2]
    bx1, by1, bx2, by2 = first
    seed = letters(image[by1:by2, bx1:bx2])
    for _ in range(REACH_ROUNDS):
        left, top, right, bottom = _cut(seed)
        cut = (left and bx1 > 0, top and by1 > 0, right and bx2 < w, bottom and by2 < h)
        if not any(cut):
            break
        bx1, by1 = max(0, bx1 - step * cut[0]), max(0, by1 - step * cut[1])
        bx2, by2 = min(w, bx2 + step * cut[2]), min(h, by2 + step * cut[3])
        seed = letters(image[by1:by2, bx1:bx2])
    if (bx1, by1, bx2, by2) == first:
        return first, seed
    inside = np.zeros(seed.shape, bool)
    inside[first[1] - by1:first[3] - by1, first[0] - bx1:first[2] - bx1] = True
    seed = _chained(seed, inside)
    # The crop shrinks back to the first box plus the letters it had cut.
    ys, xs = np.nonzero(seed)
    if not len(xs):
        return first, seed[first[1] - by1:first[3] - by1, first[0] - bx1:first[2] - bx1]
    margin = max(8, text_size(seed) // 2)
    nx1, ny1 = min(first[0], max(bx1, bx1 + int(xs.min()) - margin)), min(first[1], max(by1, by1 + int(ys.min()) - margin))
    nx2 = max(first[2], min(bx2, bx1 + int(xs.max()) + 1 + margin))
    ny2 = max(first[3], min(by2, by1 + int(ys.max()) + 1 + margin))
    return (nx1, ny1, nx2, ny2), seed[ny1 - by1:ny2 - by1, nx1 - bx1:nx2 - bx1]


def letter_mask(image: np.ndarray, box: tuple[int, int, int, int], *, with_letters: bool = False):
    """Padded box, grown over letters its edge cuts, and the erase mask of its letters, outline and glow.

    ``with_letters`` also returns the letters alone, which show where the lines of text are.
    """
    x1, y1, x2, y2 = box
    h, w = image.shape[:2]
    pad = max(8, int((y2 - y1) * PAD_SHARE))
    first = (max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad))
    if min(first[2] - first[0], first[3] - first[1]) < 8:
        empty = np.zeros((max(0, first[3] - first[1]), max(0, first[2] - first[0])), bool)
        return (first, empty, empty) if with_letters else (first, empty)
    (bx1, by1, bx2, by2), seed = _reach_cut_letters(image, first, pad)
    crop = image[by1:by2, bx1:bx2]
    ex1, ey1, ex2, ey2 = max(0, bx1 - RING), max(0, by1 - RING), min(w, bx2 + RING), min(h, by2 + RING)
    ring = np.ones((ey2 - ey1, ex2 - ex1), bool)
    ring[by1 - ey1:by2 - ey1, bx1 - ex1:bx2 - ex1] = False
    bg = cv2.cvtColor(image[ey1:ey2, ex1:ex2], cv2.COLOR_BGR2LAB)[ring]
    grown = grow(crop, seed, bg, int(text_size(seed) * REACH))
    return ((bx1, by1, bx2, by2), grown, seed) if with_letters else ((bx1, by1, bx2, by2), grown)


def still_reads(clean: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    """Letters the model still reads inside an erased block."""
    x1, y1, x2, y2 = box
    crop = clean[max(0, y1):y2, max(0, x1):x2]
    if min(crop.shape[:2]) < 8:
        return False
    seen = _prob(crop, 1.0) > THRESHOLD
    return int(seen.sum()) > max(40, LEFT_SHARE * seen.size)
