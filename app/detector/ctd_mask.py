from __future__ import annotations

import threading

import cv2
import numpy as np

from app.config import CTD_MODEL
from app.knobs import knob
from app.ort_utils import _configured_intra_op_threads

THRESHOLD = knob("mask.ink_threshold")
EDGE_FRAGMENT = 40.0
FAINT = 0.12
LINE_GAP = 0.15
CORE = 0.8
SCALES = (1.0, 0.5)
STRIDE = 64
PAD_SHARE = knob("mask.pad_share")
REACH = knob("mask.reach")
OUTLINE = knob("mask.outline")
EDGE = knob("mask.edge")
RING = knob("mask.ring")
LEFT_SHARE = knob("mask.left_share")
REACH_ROUNDS = knob("mask.reach_rounds")
CHAIN = knob("mask.chain")
KMEANS_SEED = 1234
FRINGE = knob("mask.fringe")
FRINGE_NEAR = 0.25
FRINGE_FADE = 0.2
FRINGE_TOLERANCE = knob("mask.fringe_tolerance")
FRINGE_FLAT_SHARE = knob("mask.fringe_flat_share")
FRINGE_INK_SHARE = knob("mask.fringe_ink_share")

_session = None
_lock = threading.Lock()


def _get_session():
    global _session
    with _lock:
        if _session is None:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = _configured_intra_op_threads()
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


def probability(img: np.ndarray) -> np.ndarray:
    return np.max([_prob(img, s) for s in SCALES], axis=0)


def text_size(part: np.ndarray, prob: np.ndarray | None = None) -> int:
    if prob is not None and prob.shape == part.shape:
        ink = (prob > CORE).sum(axis=1)
        rows = ink > LINE_GAP * ink.max()
        edges = np.flatnonzero(np.diff(np.concatenate(([0], rows.astype(np.int8), [0]))))
        lines = [end - start for start, end in zip(edges[::2], edges[1::2]) if end - start >= 6]
        if lines:
            return int(np.median(lines))
    _, _, stats, _ = cv2.connectedComponentsWithStats(part.astype(np.uint8))
    heights = stats[1:, cv2.CC_STAT_HEIGHT]
    heights = heights[heights >= 6]
    return int(np.median(heights)) if len(heights) else 0


def _centres(bg: np.ndarray, count: int = 4) -> np.ndarray:
    cv2.setRNGSeed(KMEANS_SEED)
    _, _, centers = cv2.kmeans(bg.astype(np.float32), min(count, len(bg)), None,
                               (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 2, cv2.KMEANS_PP_CENTERS)
    return centers


def _distance(a: np.ndarray, centers: np.ndarray) -> np.ndarray:
    nearest = None
    for centre in centers:
        diff = a - centre
        square = np.einsum("...k,...k->...", diff, diff)
        nearest = square if nearest is None else np.minimum(nearest, square)
    return np.sqrt(nearest)


def _plate(img: np.ndarray, seed: np.ndarray, grown: np.ndarray, bg: np.ndarray, size: int,
           inner: tuple[bool, bool, bool, bool]):
    if len(bg) < 20 or not seed.any() or size < 4:
        return None
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    centers = _centres(bg)
    unlike = _distance(lab, centers) > np.percentile(_distance(bg.astype(np.float32), centers), 95) + PLATE_MARGIN
    unlike = cv2.morphologyEx(unlike.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8),
                              borderType=cv2.BORDER_CONSTANT, borderValue=0)
    thin = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, size // 2) | 1,) * 2)
    raw = (unlike | grown.astype(np.uint8)) > 0
    body = cv2.morphologyEx(raw.astype(np.uint8), cv2.MORPH_OPEN, thin) > 0
    _, labels = cv2.connectedComponents(body.astype(np.uint8))
    ids = np.unique(labels[seed & body])
    part = np.isin(labels, ids[ids > 0])
    if not part.any():
        return None
    left, top, right, bottom = inner
    if (top and part[0].any()) or (bottom and part[-1].any()) or (left and part[:, 0].any()) or (right and part[:, -1].any()):
        return None
    under = part & ~grown
    if under.any() and np.median(lab[..., 0][under]) > PLATE_LIGHT:
        return None
    ys, xs = np.nonzero(part)
    sx = np.nonzero(seed)[1]
    height, width = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
    if height > PLATE_TALL * size or width > PLATE_WIDE * (sx.max() - sx.min() + 1):
        return None
    if part.sum() < PLATE_SOLID * height * width:
        return None
    part |= (cv2.dilate(part.astype(np.uint8), thin) > 0) & raw
    outline, _ = cv2.findContours(part.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cv2.drawContours(np.zeros(part.shape, np.uint8), outline, -1, 1, -1) > 0


def grow(img: np.ndarray, seed: np.ndarray, bg: np.ndarray, reach: int) -> np.ndarray:
    if reach < 1 or not seed.any() or len(bg) < 20:
        return seed
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg = bg.astype(np.float32)
    r = max(2, int(reach * OUTLINE))
    start = cv2.dilate(seed.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)) > 0
    far = cv2.distanceTransform((~seed).astype(np.uint8), cv2.DIST_L2, 3) > reach
    extra = lab[far][::7]
    if len(extra) >= 20:
        bg = np.concatenate([bg, extra])
    centers = _centres(bg, 6 if len(extra) >= 20 else 4)

    def dist(a):
        return _distance(a, centers)

    smooth = np.hypot(cv2.Sobel(lab[..., 0], cv2.CV_32F, 1, 0), cv2.Sobel(lab[..., 0], cv2.CV_32F, 0, 1)) < EDGE
    open_ = ((dist(lab.reshape(-1, 3)).reshape(lab.shape[:2]) > np.percentile(dist(bg), 90) + 6) & smooth) | seed
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
    return _fringe(lab, seed, region, reach)


def _fringe(lab: np.ndarray, seed: np.ndarray, region: np.ndarray, reach: int) -> np.ndarray:
    width = max(3, int(reach * FRINGE_NEAR))
    far = max(width, int(reach * FRINGE))

    def around(px):
        return cv2.dilate(region.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1,) * 2)) > 0

    band = around(far + 2 * width) & ~around(far)
    if band.sum() < 50 or not seed.any():
        return region
    bins, counts = np.unique(np.round(lab[band] / 4).astype(np.int32), axis=0, return_counts=True)
    back = bins[counts.argmax()].astype(np.float32) * 4
    to_back = np.linalg.norm(lab - back, axis=-1)
    tolerance = FRINGE_TOLERANCE
    if (to_back[band] <= FRINGE_TOLERANCE).mean() < FRINGE_FLAT_SHARE:
        back = np.median(lab[band], axis=0)
        to_back = np.linalg.norm(lab - back, axis=-1)
        tolerance = max(FRINGE_TOLERANCE, float(np.percentile(to_back[band], 75)))
        width = 0
    ink = float(np.median(to_back[seed]))
    shade = (to_back > tolerance) & (to_back < FRINGE_INK_SHARE * ink)
    gap = cv2.distanceTransform((~region).astype(np.uint8), cv2.DIST_L2, 3)
    smooth = cv2.GaussianBlur(to_back, (0, 0), 1.0 if width else 2.0)
    grown, k3 = region, np.ones((3, 3), np.uint8)
    for _ in range(far):
        inner = cv2.dilate(np.where(grown, smooth, 0).astype(np.float32), k3)
        fading = (gap <= width) | (smooth <= inner - FRINGE_FADE)
        nxt = grown | ((cv2.dilate(grown.astype(np.uint8), k3) > 0) & shade & (gap <= far) & fading)
        if (nxt == grown).all():
            break
        grown = nxt
    return grown


PLATE_MARGIN = 8
PLATE_LIGHT = 200
PLATE_TALL = 4.0
PLATE_WIDE = 1.6
PLATE_SOLID = 0.6

SPECK = knob("mask.speck")
SPECK_HOPS = knob("mask.speck_hops")
DOT = knob("mask.dot")


def _parts(seed: np.ndarray):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(seed.astype(np.uint8))
    size = text_size(seed)
    big = np.zeros(n, bool)
    big[1:] = stats[1:, cv2.CC_STAT_HEIGHT] >= max(4, SPECK * size)
    return labels, big, size


def _chained(seed: np.ndarray, inside: np.ndarray) -> np.ndarray:
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
        if (grown & big == keep & big).all():
            break
        keep = grown
    stats = cv2.connectedComponentsWithStats(seed.astype(np.uint8))[2]
    width, height, area = (stats[:, column] for column in (cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT, cv2.CC_STAT_AREA))
    short = np.minimum(width, height)
    dot = (short >= max(3, DOT * size)) & (np.maximum(width, height) <= 2 * short) & (area >= 0.5 * width * height)
    for _ in range(SPECK_HOPS):
        near = cv2.dilate(np.isin(labels, np.nonzero(grown & ~big)[0]).astype(np.uint8), kernel) > 0
        found = np.unique(labels[near & seed])
        found = found[~big[found] & ~grown[found] & (found > 0) & dot[found]]
        if not len(found):
            break
        grown[found] = True
    return np.isin(labels, np.nonzero(grown)[0])


def _cut(seed: np.ndarray) -> tuple[bool, bool, bool, bool]:
    labels, big, _ = _parts(seed)
    edge = {side: np.unique(part) for side, part in
            (("l", labels[:, :2]), ("t", labels[:2]), ("r", labels[:, -2:]), ("b", labels[-2:]))}
    return tuple(bool(big[edge[side]].any()) for side in "ltrb")


def _reach_cut_letters(image: np.ndarray, first: tuple[int, int, int, int], step: int):
    h, w = image.shape[:2]
    bx1, by1, bx2, by2 = first
    prob = probability(image[by1:by2, bx1:bx2])
    seed = prob > THRESHOLD
    for _ in range(REACH_ROUNDS):
        left, top, right, bottom = _cut(seed)
        cut = (left and bx1 > 0, top and by1 > 0, right and bx2 < w, bottom and by2 < h)
        if not any(cut):
            break
        bx1, by1 = max(0, bx1 - step * cut[0]), max(0, by1 - step * cut[1])
        bx2, by2 = min(w, bx2 + step * cut[2]), min(h, by2 + step * cut[3])
        prob = probability(image[by1:by2, bx1:bx2])
        seed = prob > THRESHOLD
    if (bx1, by1, bx2, by2) == first:
        return first, seed, prob
    inside = np.zeros(seed.shape, bool)
    inside[first[1] - by1:first[3] - by1, first[0] - bx1:first[2] - bx1] = True
    seed = _chained(seed, inside)
    ys, xs = np.nonzero(seed)
    inner = (slice(first[1] - by1, first[3] - by1), slice(first[0] - bx1, first[2] - bx1))
    if not len(xs):
        return first, seed[inner], prob[inner]
    margin = max(8, text_size(seed, prob) // 2)
    nx1, ny1 = min(first[0], max(bx1, bx1 + int(xs.min()) - margin)), min(first[1], max(by1, by1 + int(ys.min()) - margin))
    nx2 = max(first[2], min(bx2, bx1 + int(xs.max()) + 1 + margin))
    ny2 = max(first[3], min(by2, by1 + int(ys.max()) + 1 + margin))
    keep = (slice(ny1 - by1, ny2 - by1), slice(nx1 - bx1, nx2 - bx1))
    return (nx1, ny1, nx2, ny2), seed[keep], prob[keep]


def _faint_letters(seed: np.ndarray, prob: np.ndarray) -> np.ndarray:
    if not seed.any():
        return seed
    size = text_size(seed, prob)
    faint = (prob > FAINT) & ~seed
    if not faint.any() or size < 4:
        return seed
    rows = np.zeros(seed.shape, bool)
    rows[seed.any(axis=1)] = True
    near = cv2.dilate(seed.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * max(3, int(CHAIN * size)) + 1,) * 2)) > 0
    _, labels, stats, _ = cv2.connectedComponentsWithStats(faint.astype(np.uint8))
    keep = np.unique(labels[faint & near & rows])
    keep = keep[(keep > 0) & (stats[keep, cv2.CC_STAT_HEIGHT] <= 2 * size) & (stats[keep, cv2.CC_STAT_WIDTH] <= 2 * size)]
    return seed | np.isin(labels, keep)


LINE_INK = 110
LINE_WIDE = 0.9
LINE_TALL = 0.6
LINE_ON = 0.6
LINE_PAPER = 0.6
LINE_DARK = 0.4


def _line_art(around: np.ndarray, seed: np.ndarray, at: tuple[int, int]) -> np.ndarray:
    if not seed.any():
        return seed
    gray = cv2.cvtColor(around, cv2.COLOR_BGR2GRAY) if around.ndim == 3 else around
    _, ink, stats, _ = cv2.connectedComponentsWithStats((gray < LINE_INK).astype(np.uint8), connectivity=8)
    lines = np.flatnonzero((stats[:, cv2.CC_STAT_WIDTH] >= LINE_WIDE * gray.shape[1])
                           & (stats[:, cv2.CC_STAT_HEIGHT] >= LINE_TALL * gray.shape[0]))
    lines = lines[lines > 0]
    if not len(lines):
        return seed
    oy, ox = at
    view = (slice(oy, oy + seed.shape[0]), slice(ox, ox + seed.shape[1]))
    on_line = np.isin(ink[view], lines)
    dark = gray[view] < LINE_INK
    count, parts = cv2.connectedComponents(seed.astype(np.uint8), connectivity=8)
    dark_px = np.bincount(parts[seed & dark], minlength=count)
    hits = np.bincount(parts[seed & on_line], minlength=count)
    sizes = np.maximum(np.bincount(parts[seed], minlength=count), 1)
    drop = (dark_px >= LINE_DARK * sizes) & (hits >= LINE_ON * np.maximum(dark_px, 1))
    drop[0] = False
    for k in np.flatnonzero(drop):
        piece = (parts == k).astype(np.uint8)
        rim = (cv2.dilate(piece, np.ones((7, 7), np.uint8)) > 0) & (piece == 0) & ~seed
        if rim.any() and (~dark[rim]).mean() < LINE_PAPER:
            drop[k] = False
    return seed & ~drop[parts]


def _edge_fragments(img: np.ndarray, seed: np.ndarray, grown: np.ndarray, size: int,
                    at_edge: tuple[bool, bool]) -> np.ndarray:
    if size < 4 or not seed.any() or not any(at_edge):
        return grown
    depth = max(3, size // 2)
    cols = np.flatnonzero(seed.any(axis=0))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    around = ~grown & (cv2.dilate(grown.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0)
    if not around.any():
        return grown
    bins, counts = np.unique(np.round(lab[around] / 4).astype(np.int32), axis=0, return_counts=True)
    paper = bins[counts.argmax()].astype(np.float32) * 4
    unlike = np.linalg.norm(lab - paper, axis=-1) > EDGE_FRAGMENT
    area = np.zeros(seed.shape, bool)
    left, right = max(0, cols[0] - size), cols[-1] + 1 + size
    rows = np.flatnonzero(seed.any(axis=1))
    if at_edge[0] and rows[0] < depth:
        area[:depth, left:right] = True
    if at_edge[1] and rows[-1] >= seed.shape[0] - depth:
        area[-depth:, left:right] = True
    _, labels = cv2.connectedComponents((unlike & area).astype(np.uint8))
    touching = np.unique(np.concatenate([labels[0][area[0]], labels[-1][area[-1]]]))
    touching = touching[touching > 0]
    if not len(touching):
        return grown
    pad = cv2.dilate(np.isin(labels, touching).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    return grown | (pad & area)


def letter_mask(image: np.ndarray, box: tuple[int, int, int, int], *, with_letters: bool = False):
    x1, y1, x2, y2 = box
    h, w = image.shape[:2]
    pad = max(8, int((y2 - y1) * PAD_SHARE))
    first = (max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad))
    if min(first[2] - first[0], first[3] - first[1]) < 8:
        empty = np.zeros((max(0, first[3] - first[1]), max(0, first[2] - first[0])), bool)
        return (first, empty, empty) if with_letters else (first, empty)
    (bx1, by1, bx2, by2), seed, prob = _reach_cut_letters(image, first, pad)
    seed = _faint_letters(seed, prob)
    crop = image[by1:by2, bx1:bx2]
    ex1, ey1, ex2, ey2 = max(0, bx1 - RING), max(0, by1 - RING), min(w, bx2 + RING), min(h, by2 + RING)
    seed = _line_art(image[ey1:ey2, ex1:ex2], seed, (by1 - ey1, bx1 - ex1))
    ring = np.ones((ey2 - ey1, ex2 - ex1), bool)
    ring[by1 - ey1:by2 - ey1, bx1 - ex1:bx2 - ex1] = False
    bg = cv2.cvtColor(image[ey1:ey2, ex1:ex2], cv2.COLOR_BGR2LAB)[ring]
    size = text_size(seed, prob)
    grown = grow(crop, seed, bg, int(size * REACH))
    grown = _edge_fragments(crop, seed, grown, size, (by1 == 0, by2 == h))
    plate = _plate(crop, seed, grown, bg, size, (bx1 > 0, by1 > 0, bx2 < w, by2 < h))
    if plate is not None:
        grown = grown | plate
    return ((bx1, by1, bx2, by2), grown, seed) if with_letters else ((bx1, by1, bx2, by2), grown)


def still_reads(clean: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = box
    crop = clean[max(0, y1):y2, max(0, x1):x2]
    if min(crop.shape[:2]) < 8:
        return False
    seen = _prob(crop, 1.0) > THRESHOLD
    return int(seen.sum()) > max(40, LEFT_SHARE * seen.size)
