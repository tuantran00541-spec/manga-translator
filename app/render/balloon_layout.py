import math
from statistics import median

import cv2
import numpy as np

from app.detector.bubble_mask import MASK_ROWS, derive_bubble_mask, find_waist, mask_row_spans, split_bubble_mask
from app.mask_store import decode_mask_value
from app.ocr.reading_order import reconstruct_reading_order
from app.render.composition import _edge_distances, _interior
from app.render.source_size import source_cap_px
from app.render.text_fitting import _derived_mask, _mask_has_narrow_waist
from app.region_policy import geometry_center_in_regions, overlap_share


REGION_KEYS = ("x1", "y1", "x2", "y2")


def box_bounds(box):
    try:
        x1, y1, x2, y2 = (int(box[key]) for key in REGION_KEYS)
    except (KeyError, TypeError, ValueError):
        return None
    return (x1, y1, x2, y2) if x1 < x2 and y1 < y2 else None


def _union(bounds):
    return (min(b[0] for b in bounds), min(b[1] for b in bounds),
            max(b[2] for b in bounds), max(b[3] for b in bounds))


def anchor_coverage(mask, box):
    x1, y1, x2, y2 = box
    mx1, my1, mx2, my2 = map(int, mask["bbox"])
    if x2 <= x1 or y2 <= y1 or x2 <= mx1 or x1 >= mx2 or y2 <= my1 or y1 >= my2:
        return 0.0
    stride = max(1, (y2 - y1) // 16)
    widths = [sum(max(0, min(x2, right) - max(x1, left))
                  for left, right in mask_row_spans(mask, y))
              for y in range(y1 + stride // 2, y2, stride)]
    return sum(widths) / ((x2 - x1) * len(widths)) if widths else 0.0


def _contains_outline(mask, outline):
    x1, y1, x2, y2 = map(int, mask["bbox"])
    a, b, c, d = map(int, outline["bbox"])
    area = max(0, min(x2, c) - max(x1, a)) * max(0, min(y2, d) - max(y1, b))
    return area >= 0.88 * (c - a) * (d - b)


def verified_outline(image, anchors, outlines, load_source=None, *, min_coverage=0.72, search_region=None):
    union = _union(anchors)
    region = search_region or union

    def trace(canvas):
        if isinstance(canvas, np.ndarray):
            seed = ((union[0] + union[2]) // 2, (union[1] + union[3]) // 2)
            return derive_bubble_mask(canvas, region, seed=seed)
        return _derived_mask(canvas, region, union)

    def agrees(mask):
        return (mask is not None and mask["kind"] == "bubble"
                and all(anchor_coverage(mask, a) >= min_coverage for a in anchors)
                and all(outline is None or _contains_outline(mask, outline) for outline in outlines))

    existing = [outline for outline in outlines if outline is not None]
    if len(existing) >= 2 and all(outline["bbox"] == existing[0]["bbox"]
                                  and outline.get("rows") == existing[0].get("rows") for outline in existing):
        if agrees(existing[0]):
            return existing[0]
    mask = trace(image)
    if agrees(mask):
        return mask
    original = load_source() if load_source is not None else None
    if original is not None:
        mask = trace(original)
        if agrees(mask):
            return mask
    return None


def _chamber_mask(interior, frame):
    ys, xs = np.nonzero(interior)
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    if min(x2 - x1, y2 - y1) < MASK_ROWS:
        return None
    ox, oy = frame[:2]
    rows = []
    height = y2 - y1
    for i in range(MASK_ROWS):
        top = y1 + i * height // MASK_ROWS
        bottom = y1 + (i + 1) * height // MASK_ROWS
        edges = np.flatnonzero(np.diff(interior[top:bottom, x1:x2].all(axis=0), prepend=False, append=False))
        rows.append([[int(a), int(b)] for a, b in zip(edges[::2], edges[1::2])])
    return {"kind": "bubble", "bbox": [ox + x1, oy + y1, ox + x2, oy + y2],
            "cx": ox + round(float(xs.mean())), "cy": oy + round(float(ys.mean())), "rows": rows}


def balloon_chambers(mask):
    """Split a connected outline when it has two or more distinct interior cores."""
    frame = tuple(map(int, mask["bbox"]))
    if min(frame[2] - frame[0], frame[3] - frame[1]) < 48:
        return ()
    interior = _interior(frame, mask, False)
    distance = _edge_distances(interior)
    peak = float(distance.max())
    if peak < 12:
        return ()
    area = np.count_nonzero(interior)
    for ratio in (0.35, 0.48, 0.60, 0.68, 0.75):
        level = peak * ratio
        count, labels, stats, centres = cv2.connectedComponentsWithStats(
            (distance > level).astype(np.uint8), connectivity=8)
        if count <= 2:
            continue
        cores = []
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] < 0.022 * area:
                continue
            radius = float(distance[labels == label].max())
            if radius >= max(0.53 * peak, level + 0.12 * peak):
                cores.append((label, radius))
        if len(cores) != count - 1 or len(cores) < 2:
            continue
        if any(math.dist(centres[a], centres[b]) < 0.73 * (ra + rb)
               for i, (a, ra) in enumerate(cores) for b, rb in cores[i + 1:]):
            continue
        # Assign every part of the outline to its nearest interior core.
        distances = [cv2.distanceTransform((labels != label).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
                     for label, _ in cores]
        owners = np.argmin(distances, axis=0)
        chambers = tuple(_chamber_mask((owners == i) & (interior != 0), frame) for i in range(len(cores)))
        if all(chambers):
            return chambers
    if _mask_has_narrow_waist(mask):
        return split_bubble_mask(mask, find_waist(mask)) or ()
    return ()


def owning_chamber(chambers, anchor):
    shares = sorted(((anchor_coverage(mask, anchor), i) for i, mask in enumerate(chambers)), reverse=True)
    if shares and shares[0][0] >= 0.64:
        return shares[0][1]
    return None


def _anchor(box, bounds):
    x1, y1, x2, y2 = bounds
    letters = box.get("_letters_array")
    if letters is None:
        letters = decode_mask_value(box.get("letters_mask"))
    if letters is not None and letters.shape == (y2 - y1, x2 - x1) and letters.any():
        ys, xs = np.nonzero(letters)
        return x1 + int(xs.min()), y1 + int(ys.min()), x1 + int(xs.max()) + 1, y1 + int(ys.max()) + 1
    return bounds


def _outline(box, width, core):
    if not isinstance(box, dict) or box.get("kind") != "bubble":
        return None
    frame = box.get("bbox")
    if not isinstance(frame, (tuple, list)) or len(frame) != 4 or len(box.get("rows") or ()) != MASK_ROWS:
        return None
    x1, y1, x2, y2 = frame
    if 0 <= x1 < x2 <= width and core[0] < y1 < y2 < core[1]:
        return box
    return None


def _nearby(a, b, first, second):
    if first and (overlap_share(first["bbox"], b) or second and overlap_share(first["bbox"], second["bbox"])):
        return True
    if second and overlap_share(second["bbox"], a):
        return True
    ah, bh = a[3] - a[1], b[3] - b[1]
    return (max(a[0], b[0]) - min(a[2], b[2]) <= min(ah, bh)
            and max(a[1], b[1]) - min(a[3], b[3]) <= max(ah, bh))


def _has_lettering_background(clean, mask):
    _, top, _, bottom = mask["bbox"]
    samples = [clean[y, a:b:4] for y in range(top, bottom, 4) for a, b in mask_row_spans(mask, y)]
    if not samples:
        return False
    pixels = np.concatenate(samples)
    centre = np.median(pixels, axis=0)
    distance = np.max(np.abs(pixels - centre), axis=1)
    return np.count_nonzero(distance <= 32) >= 0.70 * len(distance)


def _plain_outline(mask):
    inside = _interior(mask["bbox"], mask, False)
    contours, _ = cv2.findContours(inside, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    hull = cv2.convexHull(np.concatenate(contours))
    return sum(cv2.contourArea(edge) for edge in contours) >= 0.925 * cv2.contourArea(hull)


def plan_balloon_regions(original, clean, boxes, *, core=None, preserve_regions=()):
    """Assign detector evidence to verified balloon interiors before translation."""
    height, width = original.shape[:2]
    core = core or (0, height)
    items = []
    for box in boxes:
        if (not box.get("id") or box.get("removed") or box.get("manual") or box.get("geometry_overridden")
                or box.get("ocr_eligible") is False or box.get("deferred_reason")
                or geometry_center_in_regions(box, preserve_regions)):
            continue
        bounds = box_bounds(box)
        if bounds and 0 <= bounds[0] < bounds[2] <= width and core[0] <= bounds[1] < bounds[3] <= core[1]:
            items.append((str(box["id"]), bounds, _anchor(box, bounds), _outline(box.get("bubble_mask"), width, core)))
    items.sort(key=lambda item: (item[2][1], item[2][0], item[0]))

    shapes = {}
    for i, (_, bounds, anchor, stored) in enumerate(items):
        if stored:
            shapes.setdefault(tuple(stored["bbox"]), stored)
            if anchor_coverage(stored, bounds) < 0.82:
                expanded = _outline(verified_outline(clean, (bounds,), (stored,), lambda: original), width, core)
                if expanded:
                    shapes[tuple(expanded["bbox"])] = expanded
        for _, _, other, other_mask in items[i + 1:]:
            if not (stored or other_mask) or not _nearby(anchor, other, stored, other_mask):
                continue
            common = verified_outline(clean, (anchor, other), (stored, other_mask), lambda: original)
            common = _outline(common, width, core)
            if common:
                shapes[tuple(common["bbox"])] = common

    proposals = []
    for frame, mask in shapes.items():
        contained = [i for i, item in enumerate(items) if anchor_coverage(mask, item[2]) >= 0.72]
        if not contained:
            continue
        chamber_groups = {}
        chambers = balloon_chambers(mask) if len(contained) > 1 else ()
        for i in contained:
            chamber = owning_chamber(chambers, items[i][2]) if chambers else 0
            if chamber is not None:
                chamber_groups.setdefault(chamber, []).append(i)
        area = (frame[2] - frame[0]) * (frame[3] - frame[1])
        for chamber, indices in chamber_groups.items():
            outline = chambers[chamber] if chambers else mask
            proposals.append((len(contained), area, outline, indices))

    plans = []
    assigned = set()
    for _, _, mask, indices in sorted(proposals, key=lambda proposal: proposal[:2], reverse=True):
        if assigned.intersection(indices):
            continue
        x1, y1, x2, y2 = mask["bbox"]
        if (any(overlap_share(mask["bbox"], region) for region in preserve_regions)
                or not _has_lettering_background(clean, mask)):
            continue
        fill = sum(b - a for row in mask["rows"] for a, b in row) / (MASK_ROWS * (x2 - x1))
        rectangular = fill >= 0.91
        if rectangular and len(indices) > 1:
            continue
        polygons = [[items[i][2][:2], items[i][2][2:]] for i in indices]
        order = reconstruct_reading_order(["x"] * len(indices), [], polygons, lang="")
        fragments = [items[indices[i]] for i in order["ordered_indices"]]
        caps = [cap for _, bounds, _, _ in fragments if (cap := source_cap_px(original, bounds)) is not None]
        plans.append({
            "sources": {oid: list(bounds) for oid, bounds, _, _ in fragments},
            "source_order": [oid for oid, _, _, _ in fragments],
            "letter_bounds": dict(zip(REGION_KEYS, _union([anchor for _, _, anchor, _ in fragments]))),
            "bubble_mask": mask,
            "container": "box" if rectangular else "bubble",
            "plain_outline": not rectangular and _plain_outline(mask),
            "source_cap_px": round(median(caps)) if caps else None,
        })
        assigned.update(indices)
    return sorted(plans, key=lambda plan: (plan["letter_bounds"]["y1"], plan["letter_bounds"]["x1"]))
