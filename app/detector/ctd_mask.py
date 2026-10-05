"""Erase masks from comic-text-detector letters, grown into outline and glow that differ from the background."""
from __future__ import annotations

import os
import threading

import cv2
import numpy as np

from app.config import CTD_MODEL
from app.knobs import knob

THRESHOLD = knob("mask.ink_threshold")
EDGE_FRAGMENT = 40.0  # Lab distance from the paper of a letter part cut by the image edge
FAINT = 0.12  # letter probability of marks that join the text only beside its letters
LINE_GAP = 0.15  # a row with less than this share of the fullest row's sure ink is between lines
CORE = 0.8  # letter probability of the surest pixels; glow and blur that weld letters into one blob fall below it
SCALES = (1.0, 0.5)  # the half-size pass catches very large lettering
STRIDE = 64  # the model's input sides must be multiples of this
PAD_SHARE = knob("mask.pad_share")
REACH = knob("mask.reach")
OUTLINE = knob("mask.outline")
EDGE = knob("mask.edge")
RING = knob("mask.ring")
LEFT_SHARE = knob("mask.left_share")
REACH_ROUNDS = knob("mask.reach_rounds")
CHAIN = knob("mask.chain")
KMEANS_SEED = 1234  # fixed seed for the background colour clusters, so one image always gives one mask
FRINGE = knob("mask.fringe")
FRINGE_NEAR = 0.25  # share of the reach a shadow may take without fading
FRINGE_FADE = 0.2  # Lab drop per pixel that marks a blur still fading out
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


def probability(img: np.ndarray) -> np.ndarray:
    """How sure the model is that each pixel is a letter, the higher of full and half size."""
    return np.max([_prob(img, s) for s in SCALES], axis=0)


def text_size(part: np.ndarray, prob: np.ndarray | None = None) -> int:
    """Letter height: the median line height of the model's surest pixels, else the median letter of a mask."""
    if prob is not None and prob.shape == part.shape:
        ink = (prob > CORE).sum(axis=1)
        # Lines set close together touch through a few strokes; a row that thin is the gap between them.
        rows = ink > LINE_GAP * ink.max()
        edges = np.flatnonzero(np.diff(np.concatenate(([0], rows.astype(np.int8), [0]))))
        lines = [end - start for start, end in zip(edges[::2], edges[1::2]) if end - start >= 6]
        if lines:
            return int(np.median(lines))
    _, _, stats, _ = cv2.connectedComponentsWithStats(part.astype(np.uint8))
    heights = stats[1:, cv2.CC_STAT_HEIGHT]
    heights = heights[heights >= 6]
    return int(np.median(heights)) if len(heights) else 0


def _centres(bg: np.ndarray) -> np.ndarray:
    """Up to four Lab colours of the background round a crop."""
    # k-means++ draws from OpenCV's per-thread RNG; without a fixed seed the mask depended on earlier calls.
    cv2.setRNGSeed(KMEANS_SEED)
    _, _, centers = cv2.kmeans(bg.astype(np.float32), min(4, len(bg)), None,
                               (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 2, cv2.KMEANS_PP_CENTERS)
    return centers


def _distance(a: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Lab distance of each pixel to the nearest background colour, one centre at a time to spare memory."""
    nearest = None
    for centre in centers:
        diff = a - centre
        square = np.einsum("...k,...k->...", diff, diff)
        nearest = square if nearest is None else np.minimum(nearest, square)
    return np.sqrt(nearest)


def _plate(img: np.ndarray, seed: np.ndarray, grown: np.ndarray, bg: np.ndarray, size: int,
           inner: tuple[bool, bool, bool, bool]):
    """A dark one-line plate the letters sit on, such as a scanlator badge, closed inside the crop; None for bubbles and art."""
    if len(bg) < 20 or not seed.any() or size < 4:
        return None
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    centers = _centres(bg)
    unlike = _distance(lab, centers) > np.percentile(_distance(bg.astype(np.float32), centers), 95) + PLATE_MARGIN
    unlike = cv2.morphologyEx(unlike.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8),
                              borderType=cv2.BORDER_CONSTANT, borderValue=0)
    # Thin panel lines touching the plate are not part of it.
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
    # Darker than paper, so bubbles and caption boxes keep their shape.
    if under.any() and np.median(lab[..., 0][under]) > PLATE_LIGHT:
        return None
    ys, xs = np.nonzero(part)
    sx = np.nonzero(seed)[1]
    height, width = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
    if height > PLATE_TALL * size or width > PLATE_WIDE * (sx.max() - sx.min() + 1):
        return None
    if part.sum() < PLATE_SOLID * height * width:
        return None
    # The plate's own thin rim, shaved off with the panel lines, comes back.
    part |= (cv2.dilate(part.astype(np.uint8), thin) > 0) & raw
    outline, _ = cv2.findContours(part.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cv2.drawContours(np.zeros(part.shape, np.uint8), outline, -1, 1, -1) > 0


def grow(img: np.ndarray, seed: np.ndarray, bg: np.ndarray, reach: int) -> np.ndarray:
    """Letters grown by an outline, then into smooth pixels unlike the background colours; growth that runs off is art."""
    if reach < 1 or not seed.any() or len(bg) < 20:
        return seed
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg = bg.astype(np.float32)
    centers = _centres(bg)

    def dist(a):
        return _distance(a, centers)

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
    return _fringe(lab, seed, region, reach)


def _fringe(lab: np.ndarray, seed: np.ndarray, region: np.ndarray, reach: int) -> np.ndarray:
    """A soft shadow or blur fading from the letters into one flat background joins the mask, so LaMa never redraws it."""
    width = max(3, int(reach * FRINGE_NEAR))
    far = max(width, int(reach * FRINGE))

    def around(px):
        return cv2.dilate(region.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1,) * 2)) > 0

    # The background is read past where a blur may still fade, as a heavy blur fills the band right by the letters.
    band = around(far + 2 * width) & ~around(far)
    if band.sum() < 50 or not seed.any():
        return region
    # The background right round the letters, when most of that band is one flat colour.
    bins, counts = np.unique(np.round(lab[band] / 4).astype(np.int32), axis=0, return_counts=True)
    back = bins[counts.argmax()].astype(np.float32) * 4
    to_back = np.linalg.norm(lab - back, axis=-1)
    tolerance = FRINGE_TOLERANCE
    if (to_back[band] <= FRINGE_TOLERANCE).mean() < FRINGE_FLAT_SHARE:
        # A textured background (stars, a nebula) takes only a glow that keeps fading, above its own spread.
        back = np.median(lab[band], axis=0)
        to_back = np.linalg.norm(lab - back, axis=-1)
        tolerance = max(FRINGE_TOLERANCE, float(np.percentile(to_back[band], 75)))
        width = 0
    ink = float(np.median(to_back[seed]))
    # Between background and ink, so neither the background nor ink-dark lines such as a balloon outline join.
    shade = (to_back > tolerance) & (to_back < FRINGE_INK_SHARE * ink)
    gap = cv2.distanceTransform((~region).astype(np.uint8), cv2.DIST_L2, 3)
    # Past the shadow's usual width only a fading blur goes on; flat art beside the letters stops it.
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


PLATE_MARGIN = 8  # Lab distance past the background's own spread that marks a plate pixel
PLATE_LIGHT = 200  # Lab lightness above which the plate is paper (a bubble or caption box), never erased
PLATE_TALL = 4.0  # a plate is at most this many letter heights tall: one line of text
PLATE_WIDE = 1.6  # and this many times their width
PLATE_SOLID = 0.6  # share of its bounding box a plate fills

SPECK = knob("mask.speck")
SPECK_HOPS = knob("mask.speck_hops")
DOT = knob("mask.dot")


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
            break
        keep = grown
    # A run of dots, like '...', joins a few steps past the letter it follows; flecks and outline scraps do not.
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
    """Which sides of the crop cut a letter-sized letter: left, top, right, bottom."""
    labels, big, _ = _parts(seed)
    edge = {side: np.unique(part) for side, part in
            (("l", labels[:, :2]), ("t", labels[:2]), ("r", labels[:, -2:]), ("b", labels[-2:]))}
    return tuple(bool(big[edge[side]].any()) for side in "ltrb")


def _reach_cut_letters(image: np.ndarray, first: tuple[int, int, int, int], step: int):
    """Grow the crop while its edge cuts letters, keeping only the text the first box holds; returns crop, seed and probability."""
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
    # The crop shrinks back to the first box plus the letters it had cut.
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
    """Faintly read marks on the text's own lines and within a letter gap of it, like a thin blurred '!', join the letters."""
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
    # A faint mark is letter-sized; a wide faint area is art the model half reads.
    keep = keep[(keep > 0) & (stats[keep, cv2.CC_STAT_HEIGHT] <= 2 * size) & (stats[keep, cv2.CC_STAT_WIDTH] <= 2 * size)]
    return seed | np.isin(labels, keep)


def _edge_fragments(img: np.ndarray, seed: np.ndarray, grown: np.ndarray, size: int,
                    at_edge: tuple[bool, bool]) -> np.ndarray:
    """Letter parts cut by the top or bottom of the image, too short for the model to read, join the mask."""
    if size < 4 or not seed.any() or not any(at_edge):
        return grown
    depth = max(3, size // 2)
    cols = np.flatnonzero(seed.any(axis=0))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    # The colour the letters sit on, not every colour round the box, as a balloon outline is one of those.
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
    (bx1, by1, bx2, by2), seed, prob = _reach_cut_letters(image, first, pad)
    seed = _faint_letters(seed, prob)
    crop = image[by1:by2, bx1:bx2]
    ex1, ey1, ex2, ey2 = max(0, bx1 - RING), max(0, by1 - RING), min(w, bx2 + RING), min(h, by2 + RING)
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
    """Letters the model still reads inside an erased block."""
    x1, y1, x2, y2 = box
    crop = clean[max(0, y1):y2, max(0, x1):x2]
    if min(crop.shape[:2]) < 8:
        return False
    seen = _prob(crop, 1.0) > THRESHOLD
    return int(seen.sum()) > max(40, LEFT_SHARE * seen.size)
