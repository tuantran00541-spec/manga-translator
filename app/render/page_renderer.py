import re
from collections import defaultdict, deque
from pathlib import Path
from statistics import median

import numpy as np
from PIL import Image
from fastapi import HTTPException

from app.logging_config import logger
from app.detector.bubble_mask import clip_bubble_mask, shift_bubble_mask
from app.render.balloon_layout import anchor_coverage, balloon_chambers, box_bounds, owning_chamber, verified_outline
from app.render.composition import _interior
from app.render.font_metrics import parse_color
from app.render.placement import (
    anchor_union, covers_source_ink, detector_anchors, detector_position, ink_window, source_ink_bounds,
    user_position,
)
from app.render.size_bands import preferred_dialogue_cap, uses_dialogue_size
from app.render.source_size import source_cap_px, source_ink_hex
from app.render.text_fitting import _derived_mask
from app.render.text_renderer import render_text_in_box
from app.region_policy import page_preserve_regions
from app.schemas import RenderRequest
from app.text_objects import source_box_ids
from app.translation.vision import PLACEHOLDER_TRANSLATIONS, collapse_repeated_lines


def style_get(d: dict, idx: int, default=None):
    if not d:
        return default
    if idx in d:
        return d[idx]
    s = str(idx)
    if s in d:
        return d[s]
    return default


def _is_auto_style(value) -> bool:
    return value is None or value == "" or (
        isinstance(value, str) and value.strip().lower() == "auto"
    )


def _resolve_ocr_style(requested, stored, ocr_value, fallback):
    value = requested if requested is not None else stored
    if _is_auto_style(value) and ocr_value not in (None, ""):
        return ocr_value
    return fallback if value is None else value


def _valid_region(region: object) -> bool:
    if not isinstance(region, dict):
        return False
    try:
        return (
            int(region["x1"]) < int(region["x2"])
            and int(region["y1"]) < int(region["y2"])
        )
    except (KeyError, TypeError, ValueError):
        return False


def _protected_boxes(regions, width, height):
    for region in regions:
        if _valid_region(region):
            x1, y1, x2, y2 = (int(region[key]) for key in ("x1", "y1", "x2", "y2"))
            box = (max(0, x1), max(0, y1), min(width, x2), min(height, y2))
            if box[0] < box[2] and box[1] < box[3]:
                yield box


def _rank_renders(rendered: list) -> list:
    def stats_of(item):
        return item[3] if len(item) > 3 and isinstance(item[3], dict) else {}

    def is_corrected(item):
        return (stats_of(item).get("fit_box") != stats_of(item).get("attempt_box")
                and stats_of(item).get("fit_box") is not None)

    corrected = [item for item in rendered if is_corrected(item)]
    pool = corrected if corrected else rendered
    if corrected and all(stats_of(item).get("below_share")
                         or stats_of(item).get("spills_letter_box")
                         for item in corrected):
        pool = pool + [item for item in rendered
                       if not is_corrected(item)
                       and not stats_of(item).get("below_share")
                       and not stats_of(item).get("spills_letter_box")]
    pool.sort(key=lambda item: (bool(stats_of(item).get("spills_letter_box")),
                                bool(stats_of(item).get("below_share")),
                                stats_of(item).get("layout_score", float("inf")), item[0],
                                -(stats_of(item).get("size") or 0), item[1]))
    return pool


def _ink_mask(base: Image.Image, candidate: Image.Image):
    return np.any(np.asarray(candidate) != np.asarray(base), axis=2)


def _mask_bbox(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def _render_diff(base, candidate, stats):
    extent = stats.get("draw_box")
    if extent is None:
        ink = _ink_mask(base, candidate)
        return ink, _mask_bbox(ink)
    x1, y1, x2, y2 = extent
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(base.width, x2), min(base.height, y2)
    mask = np.zeros((base.height, base.width), dtype=bool)
    if x1 >= x2 or y1 >= y2:
        return mask, None
    bounds = (x1, y1, x2, y2)
    diff = _ink_mask(base.crop(bounds), candidate.crop(bounds))
    mask[y1:y2, x1:x2] = diff
    box = _mask_bbox(diff)
    return mask, (x1 + box[0], y1 + box[1], x1 + box[2], y1 + box[3]) if box is not None else None


def _paste_ink(image, candidate, mask, box):
    if box is not None:
        x1, y1, x2, y2 = box
        image.paste(candidate.crop(box), box, Image.fromarray(mask[y1:y2, x1:x2]))


def _boxes_overlap(a, b):
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def _bubble_outline(mask, region):
    if not isinstance(mask, dict) or mask.get("kind") != "bubble":
        return None
    try:
        x1, y1, x2, y2 = (int(v) for v in mask["bbox"])
    except (KeyError, TypeError, ValueError):
        return None
    if _boxes_overlap((x1, y1, x2, y2), region):
        return (x1, y1, x2, y2)
    return None


def _placement_outline(clean, anchors, mask, source_loader, verified, search_region, core):
    anchor = anchor_union(anchors)
    if _bubble_outline(mask, anchor) and all(anchor_coverage(mask, spot) >= .72 for spot in anchors):
        outline = mask
    elif search_region is not None:
        outline = verified_outline(clean, anchors, (), source_loader, search_region=search_region)
        verified = False
    else:
        return None
    bounds = _bubble_outline(outline, anchor)
    if bounds is None or not (core[0] <= bounds[1] < bounds[3] <= core[1]):
        return None
    if verified:
        return outline
    chambers = balloon_chambers(outline)
    if not chambers:
        return outline
    owner = owning_chamber(chambers, anchor)
    if owner is not None and all(anchor_coverage(chambers[owner], spot) >= .72 for spot in anchors):
        return chambers[owner]
    return None


def _ink_inside_placement(changed, box, bounds, outline):
    if box is None:
        return True
    x1, y1, x2, y2 = box
    left, top, right, bottom = bounds
    if x1 < left or y1 < top or x2 > right or y2 > bottom:
        return False
    return outline is None or not np.any(changed[y1:y2, x1:x2] & (_interior(box, outline, False) == 0))


INK_GAP = 4
INK_SHIFT_CAP = 64


def _fit_around_ink(base, text, box, style, placed_boxes, placed_mask):
    if str(style["bg_color"] or "").lower() not in ("", "none", "transparent"):
        return None
    reserved = tuple((x1 - INK_GAP, y1 - INK_GAP, x2 + INK_GAP, y2 + INK_GAP)
                     for x1, y1, x2, y2 in placed_boxes)
    stats = {"attempt_box": box}
    try:
        candidate = _render_in_region(base.copy(), text, box, stats=stats,
                                      **(style | {"reserved_boxes": reserved}))
    except ValueError:
        return None
    mask, ink = _render_diff(base, candidate, stats)
    if ink is None or np.any(mask & placed_mask):
        return None
    return candidate, mask, ink, stats


def _clearing_shift(cand_box, placed_boxes, height: int, frame=None, anchor_box=None,
                    placed_tops=None, cand_top=None):
    def clears(box):
        return not any(px1 < box[2] and box[0] < px2 and py1 < box[3] and box[1] < py2
                       for px1, py1, px2, py2 in placed_boxes)

    def allowed(dy):
        moved = (cand_box[0], cand_box[1] + dy, cand_box[2], cand_box[3] + dy)
        if moved[1] < 0 or moved[3] > height:
            return False
        if not clears(moved):
            return False
        if frame is not None:
            anchor = anchor_box if anchor_box is not None else cand_box
            if anchor[1] + dy < frame[1] or anchor[3] + dy > frame[3]:
                return False
        if placed_tops is not None and cand_top is not None:
            new_top = cand_top + dy
            for pt in placed_tops:
                if (cand_top - pt) * (new_top - pt) < 0:
                    return False
        return True

    if clears(cand_box):
        return 0
    ups = [py1 - INK_GAP - cand_box[3] for px1, py1, px2, py2 in placed_boxes
           if px1 < cand_box[2] and cand_box[0] < px2 and py1 < cand_box[3] and cand_box[1] < py2]
    downs = [py2 + INK_GAP - cand_box[1] for px1, py1, px2, py2 in placed_boxes
             if px1 < cand_box[2] and cand_box[0] < px2 and py1 < cand_box[3] and cand_box[1] < py2]
    options = []
    if ups:
        dy = min(ups)
        if dy < 0 and allowed(dy):
            options.append(dy)
    if downs:
        dy = max(downs)
        if dy > 0 and allowed(dy):
            options.append(dy)
    if not options:
        return None
    return min(options, key=abs)


def _rerender_non_colliding(
    image: Image.Image,
    text: str,
    sbox: tuple[int, int, int, int],
    style: dict,
    placed_mask,
    frame_bound=None,
):
    if sbox[2] - sbox[0] < 16 or sbox[3] - sbox[1] < 16:
        return None
    sstats: dict = {"attempt_box": sbox}
    sub_mask = None
    if isinstance(style.get("bubble_mask"), dict):
        sub_mask = clip_bubble_mask(style["bubble_mask"], sbox)
        if style["bubble_mask"].get("kind") == "bubble" and sub_mask is None:
            return None
    sub_style = {**style, "bubble_mask": sub_mask, "letters": None,
                 "letters_box": None, "source_letter_box": None}
    try:
        scandidate = _render_in_region(image.copy(), text, sbox, stats=sstats, **sub_style)
    except ValueError:
        return None
    smask, sbox_ink = _render_diff(image, scandidate, sstats)
    if sbox_ink is None or int((smask & placed_mask).sum()) != 0:
        return None
    if frame_bound is not None and not (
        frame_bound[0] <= sbox_ink[0]
        and frame_bound[1] <= sbox_ink[1]
        and sbox_ink[2] <= frame_bound[2]
        and sbox_ink[3] <= frame_bound[3]
    ):
        return None
    return (scandidate, smask, sbox_ink, sstats)


def _stats_verdict_fields(stats: dict, frame, attempt: int) -> dict:
    return {
        "attempt": attempt,
        "size": stats.get("size"),
        "lines": stats.get("lines"),
        "frame": frame,
        "frame_source": stats.get("frame_source"),
        "lobe": stats.get("lobe"),
        "fallbacks": stats.get("fallbacks", []),
        "stages_skipped": stats.get("stages_skipped", []),
        "band_narrowing": stats.get("band_narrowing", {}),
        "below_share": bool(stats.get("below_share")),
        "spills_letter_box": bool(stats.get("spills_letter_box")),
        "fill_fallback": stats.get("fill_fallback"),
        "edge_clearance": stats.get("edge_clearance"),
        "layout_score": stats.get("layout_score"),
        "preferred_size": stats.get("preferred_size"),
    }


def _frame_iou(a, b) -> float:
    if a is None or b is None:
        return 0.0
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _render_region_for_text_object(obj: dict) -> dict:
    region = obj.get("region") or {}
    ocr_region = obj.get("ocr_text_region")
    if (
        obj.get("auto_generated")
        and region == obj.get("auto_geometry")
        and _valid_region(ocr_region)
        and not _valid_region(obj.get("letter_bounds"))
        and not isinstance(obj.get("bubble_mask"), dict)
        and not obj.get("container")
        and obj.get("source_cap_px") is None
    ):
        return ocr_region
    return region


def _uses_auto_geometry(obj: dict) -> bool:
    return bool(obj.get("auto_generated") and obj.get("region") == obj.get("auto_geometry")
                and not obj.get("seam_owner"))


def _reflow_dialogue(text: str) -> str:
    paragraphs = re.split(r"\n[ \t]*\n", text.replace("\r\n", "\n"))
    return "\n\n".join(" ".join(paragraph.split()) for paragraph in paragraphs if paragraph.strip())


def _object_translation(req, obj):
    return style_get(req.translations, obj.get("id")) or obj.get("translation", "")


def _same_lettering_style(obj, other, style_maps):
    fields = ("style", "speaker", "enlarge")
    if ((obj.get("typography_role") or "dialogue") != (other.get("typography_role") or "dialogue")
            or any(obj.get(field) != other.get(field) for field in fields)
            or any(style_get(m, obj["id"]) != style_get(m, other["id"]) for m in style_maps)):
        return False
    profiles = (obj.get("ocr_color_profile"), other.get("ocr_color_profile"))
    a, b = [profile if isinstance(profile, dict) else {} for profile in profiles]
    if any(a.get(key) != b.get(key) for key in ("gradient", "stroke_color", "glow_color")):
        return False
    ac = (parse_color(obj.get("lettering_color"), default=None)
          or parse_color(obj.get("ocr_text_color") or a.get("fill"), default=None))
    bc = (parse_color(other.get("lettering_color"), default=None)
          or parse_color(other.get("ocr_text_color") or b.get("fill"), default=None))
    if ac is None or bc is None or max(abs(x - y) for x, y in zip(ac, bc)) <= 24:
        return True
    return all(max(c) - min(c) <= 12 and max(c) <= 140 for c in (ac, bc))


def _fragment_direction(a, b, text, other_text, frame):
    min_w = min(a[2] - a[0], b[2] - b[0])
    min_h = min(a[3] - a[1], b[3] - b[1])
    left, right = sorted((a, b))
    y_overlap = min(a[3], b[3]) - max(a[1], b[1])
    if (y_overlap >= 0.62 * min_h and -0.18 * min_w <= right[0] - left[2] <= min_h
            and max(len(text.split()), len(other_text.split())) <= 5):
        return "row"
    top, bottom = sorted((a, b), key=lambda box: box[1])
    overlap_x = min(a[2], b[2]) - max(a[0], b[0])
    if overlap_x < 0.55 * min_w or bottom[1] - top[3] < -0.22 * min_h:
        return None
    upper = text if top == a else other_text
    limit = (0.22 if upper.endswith((".", "!", "?", "…")) else 0.35) * (frame[3] - frame[1])
    return "stack" if bottom[1] - top[3] <= max(0.5 * min_h, limit) else None


def _nearby_balloon_anchors(a, b, am, bm):
    if am is not None and (_boxes_overlap(am["bbox"], b) or
                           (bm is not None and _boxes_overlap(am["bbox"], bm["bbox"]))):
        return True
    if bm is not None and _boxes_overlap(bm["bbox"], a):
        return True
    min_h = min(a[3] - a[1], b[3] - b[1])
    max_h = max(a[3] - a[1], b[3] - b[1])
    return (max(a[0], b[0]) - min(a[2], b[2]) <= min_h
            and max(a[1], b[1]) - min(a[3], b[3]) <= max_h)


def _plan_balloon_lettering(image, req, objects, style_maps, source_image_loader, core, source_anchors=None):
    candidates = []
    for obj in objects:
        oid = obj.get("id")
        mask = obj.get("bubble_mask")
        translation = _object_translation(req, obj)
        region = obj.get("letter_bounds")
        if not _valid_region(region):
            region = obj.get("region")
        if (not oid or obj.get("verified_balloon") or not _uses_auto_geometry(obj) or not _valid_region(region)
                or str(obj.get("container") or "").lower() not in ("bubble", "spiky")
                or not translation
                or translation.strip().lower() in PLACEHOLDER_TRANSLATIONS):
            continue
        text = _reflow_dialogue(translation)
        if source_anchors is not None:
            if oid not in source_anchors:
                continue
            bounds = anchor_union(source_anchors[oid])
        else:
            bounds = tuple(int(region[key]) for key in ("x1", "y1", "x2", "y2"))
        if not (0 <= bounds[0] < bounds[2] <= image.width and core[0] <= bounds[1] < bounds[3] <= core[1]):
            continue
        can_join = (translation == obj.get("auto_translation") and obj.get("font_selection_mode") != "user"
                    and obj.get("typography_role") in (None, "", "dialogue", "thought", "whisper", "shout", "dark_threat")
                    and obj.get("ocr_orientation") != "vertical"
                    and "\n" not in text and len(text) <= 150)
        candidates.append((obj, bounds, mask if _bubble_outline(mask, bounds) else None, text, can_join))

    available = {}
    outlines = {}
    adjacent = {}
    for i, (obj, a, am, text, can_join) in enumerate(candidates):
        for j in range(i + 1, len(candidates)):
            other, b, bm, other_text, other_can_join = candidates[j]
            if (am is None and bm is None) or not _nearby_balloon_anchors(a, b, am, bm):
                continue
            common = verified_outline(image, (a, b), (am, bm), source_image_loader)
            if common is None or not (core[0] <= common["bbox"][1] < common["bbox"][3] <= core[1]):
                continue
            key = tuple(common["bbox"])
            if key not in available:
                available[key] = balloon_chambers(common)
            pieces = available[key]
            if pieces:
                first, second = owning_chamber(pieces, a), owning_chamber(pieces, b)
                if first is not None and second is not None and first != second:
                    outlines[obj["id"]] = pieces[first]
                    outlines[other["id"]] = pieces[second]
                continue
            direction = _fragment_direction(a, b, text, other_text, key)
            if (not direction or not (can_join and other_can_join) or text == other_text
                    or not _same_lettering_style(obj, other, style_maps)):
                continue
            adjacent.setdefault(i, []).append((j, common, direction))
            adjacent.setdefault(j, []).append((i, common, direction))

    groups = {}
    joined = set()
    for i, (obj, _, _, _, _) in enumerate(candidates):
        if i in joined or obj["id"] in outlines or i not in adjacent:
            continue
        _, common, direction = adjacent[i][0]
        group = [i]
        for member in group:
            for neighbour, mask, order in adjacent[member]:
                if (order != direction or neighbour in group or neighbour in joined
                        or candidates[neighbour][0]["id"] in outlines
                        or _frame_iou(mask["bbox"], common["bbox"]) < 0.9
                        or not _same_lettering_style(obj, candidates[neighbour][0], style_maps)
                        or anchor_coverage(common, candidates[neighbour][1]) < 0.72):
                    continue
                group.append(neighbour)
        if len(group) < 2:
            continue
        axis = 0 if direction == "row" else 1
        items = [candidates[j] for j in sorted(group, key=lambda j: candidates[j][1][axis])]
        words = " ".join(item[3] for item in items)
        if len(words) > 350:
            continue
        union = tuple((min if k < 2 else max)(item[1][k] for item in items) for k in range(4))
        region = dict(zip(("x1", "y1", "x2", "y2"), union))
        caps = [item[0]["source_cap_px"] for item in items
                if isinstance(item[0].get("source_cap_px"), (int, float)) and item[0]["source_cap_px"] > 0]
        merged = {**obj, "region": region, "letter_bounds": region, "letters_mask": None,
                  "bubble_mask": common, "translation": words, "auto_translation": words}
        if caps:
            merged["source_cap_px"] = round(median(caps))
        groups[obj["id"]] = (merged, tuple(candidates[j][0]["id"] for j in group if j != i))
        joined.update(group)
    return groups, outlines


LETTER_MARGIN = 0.15
FIT_INSET = 0.04


def letter_box(obj: dict, region: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    bounds = obj.get("letter_bounds")
    if not _valid_region(bounds):
        return None
    x1, y1, x2, y2 = (int(bounds[k]) for k in ("x1", "y1", "x2", "y2"))
    margin = max(8, int(LETTER_MARGIN * min(x2 - x1, y2 - y1)))
    box = (max(region[0], x1 - margin), max(region[1], y1 - margin),
           min(region[2], x2 + margin), min(region[3], y2 + margin))
    return box if box[0] < box[2] and box[1] < box[3] and box != tuple(region) else None


def fit_attempts(obj: dict, coords: tuple[int, int, int, int]) -> list:
    tight = letter_box(obj, coords)
    if not tight:
        return [coords]
    half = tuple((a + b) // 2 for a, b in zip(tight, coords))
    attempts = [tight, half, coords] if half != tight else [tight, coords]
    if str(obj.get("container") or "").lower() == "bubble":
        tall = (tight[0], coords[1], tight[2], coords[3])
        if tall not in attempts:
            attempts.insert(1, tall)
    bounds = obj.get("letter_bounds")
    pull = 0
    if _valid_region(bounds):
        pull = max(6, int(FIT_INSET * (int(bounds["x2"]) - int(bounds["x1"]))))
    if pull:
        attempts = [box if box[0] == tight[0] or box[2] - box[0] <= 2 * pull
                    else (box[0] + pull, box[1], box[2] - pull, box[3]) for box in attempts]
    return attempts


def _render_in_region(image: Image.Image, text: str, box: tuple[int, int, int, int], **style) -> Image.Image:
    width, height = image.size
    x1, x2 = max(0, box[0]), min(width, box[2])
    top, bottom = max(0, -box[1]), max(0, box[3] - height)
    if not (top or bottom):
        return render_text_in_box(image, text, (x1, box[1], x2, box[3]), **style)
    pixels = np.asarray(image)
    pad = ((top, bottom), (0, 0)) + (((0, 0),) if pixels.ndim == 3 else ())
    canvas = Image.fromarray(np.pad(pixels, pad, mode="edge"), mode=image.mode)
    padded_style = style
    if top and isinstance(style.get("bubble_mask"), dict):
        padded_style = {**padded_style, "bubble_mask": shift_bubble_mask(style["bubble_mask"], dy=top)}
    if top and isinstance(style.get("letters_box"), tuple):
        padded_style = {**padded_style,
                        "letters_box": (style["letters_box"][0], style["letters_box"][1] + top,
                                        style["letters_box"][2], style["letters_box"][3] + top)}
    if top and isinstance(style.get("source_letter_box"), tuple):
        sb = style["source_letter_box"]
        padded_style = {**padded_style,
                        "source_letter_box": (sb[0], sb[1] + top, sb[2], sb[3] + top)}
    if top and style.get("reserved_boxes"):
        padded_style = {**padded_style, "reserved_boxes":
                        tuple((a, b + top, c, d + top) for a, b, c, d in style["reserved_boxes"])}
    canvas = render_text_in_box(canvas, text, (x1, box[1] + top, x2, box[3] + top), **padded_style)
    stats = style.get("stats")
    if isinstance(stats, dict):
        for key in ("ink_box", "draw_box"):
            if stats.get(key) is None:
                continue
            ix1, iy1, ix2, iy2 = stats[key]
            ix1, iy1 = max(ix1, 0), max(iy1, top)
            ix2, iy2 = min(ix2, width), min(iy2, top + height)
            stats[key] = (ix1, iy1 - top, ix2, iy2 - top) if ix2 > ix1 and iy2 > iy1 else None
        if top:
            for key in ("frame", "fit_box"):
                if stats.get(key) is None:
                    continue
                fx1, fy1, fx2, fy2 = stats[key]
                stats[key] = (fx1, fy1 - top, fx2, fy2 - top)
    return canvas.crop((0, top, width, top + height))


def cleanup_tmp(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass


def render_boxes_legacy(
    image: Image.Image,
    req: RenderRequest,
    page: dict,
    drafts: dict,
    colors_dict: dict,
    fonts_dict: dict,
    font_sizes_dict: dict,
    bolds_dict: dict,
    stroke_w_dict: dict,
    stroke_c_dict: dict,
    bg_colors_dict: dict,
    radii_dict: dict,
    *,
    source_image_loader=None,
    verdicts=None,
) -> int:
    rendered_count = 0
    protected = tuple(_protected_boxes(page_preserve_regions(page), *image.size))
    clean = image.copy() if source_image_loader is not None else image
    box_indices = set()
    for box_key in req.translations.keys():
        try:
            box_indices.add(int(box_key))
        except (TypeError, ValueError):
            continue

    for draft_key, d_val in drafts.items():
        parts = str(draft_key).split("_")
        if len(parts) == 2 and parts[0] == str(req.page_index):
            try:
                b_idx = int(parts[1])
                if isinstance(d_val, dict) and d_val.get("text"):
                    box_indices.add(b_idx)
            except ValueError:
                pass

    for box_idx in sorted(box_indices):
        if box_idx < 0 or box_idx >= len(page["boxes"]):
            continue
        box = page["boxes"][box_idx]
        if not isinstance(box, dict) or box.get("removed"):
            continue

        draft_key = f"{req.page_index}_{box_idx}"
        draft_item = drafts.get(draft_key, {})

        translation = style_get(req.translations, box_idx)
        if translation is None or translation == "":
            translation = draft_item.get("text", "")
        translation = collapse_repeated_lines(translation)
        if not (translation or "").strip():
            continue

        coords_raw = (
            int(box["x1"]),
            int(box["y1"]),
            int(box["x2"]),
            int(box["y2"]),
        )
        img_w, img_h = image.size
        x1 = max(0, min(img_w, coords_raw[0]))
        y1 = max(0, min(img_h, coords_raw[1]))
        x2 = max(0, min(img_w, coords_raw[2]))
        y2 = max(0, min(img_h, coords_raw[3]))
        if x2 <= x1 or y2 <= y1:
            logger.warning(
                "Chapter {} page {} box {} invalid coordinates ({},{},{},{}) for image size ({},{}), skipping",
                req.chapter_id,
                req.page_index,
                box_idx,
                x1, y1, x2, y2,
                img_w, img_h,
            )
            continue
        coords = (x1, y1, x2, y2)
        locked = source_image_loader is not None and detector_position(box, {})
        if locked:
            anchor = source_ink_bounds(box, source_image_loader(), clean)
            if anchor is None:
                if verdicts is not None:
                    verdicts.append({"id": str(box_idx), "guard": "placement_unverified", "review": True})
                continue
            coords = ink_window(anchor, coords)

        box_color = style_get(colors_dict, box_idx)
        if box_color is None:
            box_color = draft_item.get("color", "auto")

        box_font = style_get(fonts_dict, box_idx)
        if box_font is None:
            box_font = draft_item.get("font", "default")

        box_size = style_get(font_sizes_dict, box_idx)
        if box_size is None:
            box_size = draft_item.get("fontSize", "auto")

        bold_val = style_get(bolds_dict, box_idx)
        if bold_val is None:
            bold_val = draft_item.get("bold", False)
        box_bold = bool(bold_val) if bold_val is not None else False

        stroke_w = style_get(stroke_w_dict, box_idx)
        if stroke_w is None:
            stroke_w = draft_item.get("strokeWidth", "auto")

        stroke_c = style_get(stroke_c_dict, box_idx)
        if stroke_c is None:
            stroke_c = draft_item.get("strokeColor", "auto")

        bg_c = style_get(bg_colors_dict, box_idx)
        if bg_c is None:
            bg_c = draft_item.get("bgColor", "transparent")

        r_raw = style_get(radii_dict, box_idx)
        if r_raw is None:
            r_raw = draft_item.get("cornerRadius", 0)
        stats = {"attempt_box": coords}
        try:
            r_val = int(r_raw or 0)
        except (TypeError, ValueError):
            r_val = 0

        try:
            legacy_profile = (
                box.get("ocr_color_profile")
                if _is_auto_style(box_color) and isinstance(box.get("ocr_color_profile"), dict)
                else None
            )
            candidate = render_text_in_box(
                image.copy() if locked or protected else image,
                translation.strip(),
                coords,
                fill=box_color,
                font_size=box_size,
                is_bold=box_bold,
                font_name=box_font or "default",
                stroke_width=stroke_w,
                stroke_color=stroke_c,
                bg_color=bg_c,
                corner_radius=r_val,
                color_profile=legacy_profile,
                adapt_contrast=_is_auto_style(box_color),
                lock_frame=locked,
                stats=stats,
            )
            if protected and any(_ink_mask(image.crop(area), candidate.crop(area)).any() for area in protected):
                if verdicts is not None:
                    verdicts.append({"id": str(box_idx), "guard": "skipped_collision", "review": True})
                continue
            if locked:
                changed, ink = _render_diff(image, candidate, stats)
                if not _ink_inside_placement(changed, ink, coords, None):
                    if verdicts is not None:
                        verdicts.append({"id": str(box_idx), "guard": "placement_outside_bounds", "review": True})
                    continue
            if locked or protected:
                image.paste(candidate)
            rendered_count += 1
        except Exception as e:
            if locked and isinstance(e, ValueError) and "minimum readable" in str(e):
                if verdicts is not None:
                    verdicts.append({"id": str(box_idx), "guard": "placement_overset", "review": True})
                continue
            logger.opt(exception=True).error(
                "Chapter {} page {} box {} operation 'render_text_in_box' failed: {}",
                req.chapter_id,
                req.page_index,
                box_idx,
                e,
            )
            raise HTTPException(
                500,
                f"Chèn chữ thất bại (ô {box_idx})",
            ) from e

    return rendered_count


def render_text_objects(
    image: Image.Image,
    req: RenderRequest,
    text_objects: list[dict],
    colors_dict: dict,
    fonts_dict: dict,
    font_sizes_dict: dict,
    bolds_dict: dict,
    stroke_w_dict: dict,
    stroke_c_dict: dict,
    bg_colors_dict: dict,
    radii_dict: dict,
    horizontal_aligns_dict: dict,
    vertical_aligns_dict: dict,
    core_y1: int | None = None,
    core_y2: int | None = None,
    verdicts: list | None = None,
    size_bands: dict | None = None,
    source_image_loader=None,
    protected_regions=(),
    source_boxes: dict | None = None,
) -> int:
    rendered_count = 0
    img_w, img_h = image.size
    seen_keys = set()
    placed_mask = np.zeros((img_h, img_w), dtype=bool)
    protected_boxes = tuple(_protected_boxes(protected_regions, img_w, img_h))
    placed_boxes = list(protected_boxes)
    for x1, y1, x2, y2 in protected_boxes:
        placed_mask[y1:y2, x1:x2] = True
    placed_frames: list[tuple] = []
    def _order_key(obj):
        region = _render_region_for_text_object(obj)
        framed = 0 if obj.get("verified_balloon") else (
            1 if str(obj.get("container") or "").lower() in ("bubble", "spiky", "box", "screen") else 2)
        try:
            return (framed, int(region["y1"]), int(region["x1"]))
        except (KeyError, TypeError, ValueError):
            return (framed, 0, 0)

    objects = sorted(text_objects, key=_order_key)
    clean = image.copy() if len(objects) > 1 else image
    style_maps = (colors_dict, fonts_dict, font_sizes_dict, bolds_dict, stroke_w_dict, stroke_c_dict,
                  bg_colors_dict, radii_dict, horizontal_aligns_dict, vertical_aligns_dict)
    core = (core_y1 if core_y1 is not None else 0, core_y2 if core_y2 is not None else img_h)
    anchors_by_id = None
    if source_boxes is not None:
        anchors_by_id = {}
        supported = []
        owners = defaultdict(list)
        for obj in objects:
            oid = obj.get("id")
            text = (_object_translation(req, obj) or "").strip()
            if oid and text and text.lower() not in PLACEHOLDER_TRANSLATIONS and detector_position(obj, source_boxes):
                raw = source_image_loader() if source_image_loader is not None else None
                anchors = detector_anchors(obj, source_boxes, raw, clean)
                region = box_bounds(obj.get("region"))
                if not anchors or region is None or not all(covers_source_ink(region, anchor) for anchor in anchors):
                    if verdicts is not None:
                        verdicts.append({"id": oid, "guard": "placement_unverified", "review": True})
                    continue
                anchors_by_id[oid] = anchors
                for ref in source_box_ids(obj):
                    owners[ref].append(oid)
            supported.append(obj)
        ambiguous = {oid for ids in owners.values() if len(ids) > 1 for oid in ids}
        if verdicts is not None:
            verdicts.extend({"id": oid, "guard": "placement_ambiguous", "review": True} for oid in sorted(ambiguous))
        for oid in ambiguous:
            anchors_by_id.pop(oid, None)
        objects = [obj for obj in supported if obj.get("id") not in ambiguous]
    groups, chambers = _plan_balloon_lettering(clean, req, objects, style_maps, source_image_loader, core, anchors_by_id)
    consumed = set()
    pending = deque(objects)
    while pending:
        original = obj = pending.popleft()
        oid = obj.get("id")
        if not oid or oid in consumed:
            continue
        pair = groups.pop(oid, None)
        if pair:
            obj, joined_ids = pair
            translation = obj["translation"]
        else:
            translation = _object_translation(req, obj)
        auto_text = bool(obj.get("auto_translation") and translation == obj["auto_translation"])
        translation = collapse_repeated_lines(translation)
        if not (translation or "").strip() or str(translation).strip().lower() in PLACEHOLDER_TRANSLATIONS:
            continue

        region = obj.get("region") if anchors_by_id is not None else _render_region_for_text_object(obj)
        try:
            coords_raw = (
                int(region["x1"]), int(region["y1"]),
                int(region["x2"]), int(region["y2"]),
            )
        except (KeyError, TypeError, ValueError):
            logger.warning(
                "Chapter {} page {} object {} has malformed region, skipping",
                req.chapter_id, req.page_index, oid,
            )
            continue

        x1 = max(0, min(img_w, coords_raw[0]))
        y1 = max(0, min(img_h, coords_raw[1]))
        x2 = max(0, min(img_w, coords_raw[2]))
        y2 = max(0, min(img_h, coords_raw[3]))
        if x2 <= x1 or y2 <= y1:
            logger.warning(
                "Chapter {} page {} object {} invalid coordinates ({},{},{},{}) for image size ({},{}), skipping",
                req.chapter_id, req.page_index, oid,
                x1, y1, x2, y2, img_w, img_h,
            )
            continue
        dup_key = (coords_raw, translation.strip())
        if dup_key in seen_keys:
            if verdicts is not None:
                verdicts.append({"id": oid, "guard": "dedup_exact"})
            continue
        seen_keys.add(dup_key)
        clamp_top = core_y1 is None or core_y1 <= 0
        clamp_bottom = core_y2 is None or core_y2 >= img_h
        coords = (
            x1,
            y1 if clamp_top else coords_raw[1],
            x2,
            y2 if clamp_bottom else coords_raw[3],
        )
        if (obj.get("container") == "screen" and core_y1 is not None and core_y2 is not None):
            clipped_y1 = max(coords_raw[1], core_y1)
            clipped_y2 = min(coords_raw[3], core_y2)
            if clipped_y2 - clipped_y1 < 0.5 * (coords_raw[3] - coords_raw[1]):
                if verdicts is not None:
                    verdicts.append({"id": oid, "guard": "screen_sliver_skipped"})
                continue
            coords = (x1, clipped_y1, x2, clipped_y2)

        obj_style = obj.get("style") or {}
        requested_color = style_get(colors_dict, oid)
        stored_color = obj_style.get("color", "auto")
        user_color_auto = _is_auto_style(
            requested_color if requested_color is not None else stored_color
        )
        raw_color_profile = obj.get("ocr_color_profile")
        profile_fill_default = (
            raw_color_profile.get("fill")
            if isinstance(raw_color_profile, dict) and raw_color_profile.get("fill")
            else None
        )
        lettering_c = obj.get("lettering_color")
        ocr_color_default = (
            lettering_c
            if lettering_c and str(lettering_c).lower() != "auto"
            else (obj.get("ocr_text_color") or profile_fill_default or lettering_c)
        )
        box_color = _resolve_ocr_style(
            requested_color,
            stored_color,
            ocr_color_default,
            "auto",
        )
        active_color_profile = (
            raw_color_profile
            if user_color_auto and isinstance(raw_color_profile, dict)
            else None
        )
        box_font = style_get(fonts_dict, oid)
        if box_font is None:
            box_font = obj_style.get("font", "default")
        container = str(obj.get("container") or "").lower()
        verified_balloon = bool(obj.get("verified_balloon") and _uses_auto_geometry(obj))
        effective_source_cap = obj.get("source_cap_px")
        role_key = str(obj.get("typography_role") or "").lower()
        requested_size = style_get(font_sizes_dict, oid, obj_style.get("fontSize", "auto"))
        if (_is_auto_style(requested_size) and uses_dialogue_size(obj)
                and isinstance(size_bands, dict) and "dialogue" in size_bands):
            effective_source_cap = preferred_dialogue_cap(size_bands["dialogue"])
        elif effective_source_cap is None and isinstance(size_bands, dict) and role_key in size_bands:
            band = size_bands[role_key]
            if isinstance(band, (list, tuple)) and len(band) == 2:
                effective_source_cap = preferred_dialogue_cap(band)
        if (effective_source_cap is None and container == "free" and auto_text
                and _uses_auto_geometry(obj) and _is_auto_style(requested_size)
                and source_image_loader is not None):
            source = source_image_loader()
            if source is not None:
                bgr = np.asarray(source.crop((x1, y1, x2, y2)).convert("RGB"))[:, :, ::-1]
                effective_source_cap = source_cap_px(
                    bgr, (0, 0, x2 - x1, y2 - y1),
                    min_pieces=1 if len("".join(translation.split())) <= 2 else 3,
                )
        ocr_size_default = None if effective_source_cap is not None else obj.get("ocr_font_size")
        box_size = _resolve_ocr_style(
            style_get(font_sizes_dict, oid),
            obj_style.get("fontSize", "auto"),
            ocr_size_default,
            "auto",
        )
        compose_layout = bool((original.get("auto_generated") or original.get("origin") == "manual" or auto_text)
                              and _is_auto_style(requested_size) and not original.get("seam_owner"))
        bold_val = style_get(bolds_dict, oid)
        if bold_val is None:
            bold_val = obj_style.get("bold", False)
        box_bold = bool(bold_val) if bold_val is not None else False
        stroke_w = style_get(stroke_w_dict, oid)
        if stroke_w is None:
            stroke_w = obj_style.get("strokeWidth", "auto")
        stroke_c = style_get(stroke_c_dict, oid)
        if stroke_c is None:
            stroke_c = obj_style.get("strokeColor", "auto")
        bg_c = style_get(bg_colors_dict, oid)
        if bg_c is None:
            bg_c = obj_style.get("bgColor", "transparent")
        r_raw = style_get(radii_dict, oid)
        if r_raw is None:
            r_raw = obj_style.get("cornerRadius", 0)
        try:
            r_val = int(r_raw or 0)
        except (TypeError, ValueError):
            r_val = 0

        h_align = style_get(horizontal_aligns_dict, oid)
        if h_align is None:
            h_align = obj_style.get("horizontalAlign", "center")
        v_align = style_get(vertical_aligns_dict, oid)
        if v_align is None:
            v_align = obj_style.get("verticalAlign", "middle")

        obj_shape = str(obj.get("shape") or "rectangle").lower()
        if obj_shape not in ("rectangle", "ellipse"):
            obj_shape = "rectangle"

        style = dict(
            fill=box_color,
            font_size=box_size,
            is_bold=box_bold,
            font_name=box_font,
            stroke_width=stroke_w,
            stroke_color=stroke_c,
            bg_color=bg_c,
            corner_radius=r_val,
            shape=obj_shape,
            horizontal_align=h_align,
            vertical_align=v_align,
            source_cap_px=effective_source_cap,
            container=obj.get("container"),
            typography_role=role_key if auto_text and _uses_auto_geometry(original) else None,
            enlarge=bool(obj.get("enlarge")),
            bubble_mask=obj.get("bubble_mask"),
            color_profile=active_color_profile,
            adapt_contrast=user_color_auto,
            keep_compounds=(auto_text or pair is not None) and container in ("bubble", "spiky"),
            optical_size_hint=ocr_size_default,
            verified_balloon=verified_balloon,
            reserved_boxes=protected_boxes,
        )
        bounds = obj.get("letter_bounds")
        if _valid_region(bounds):
            anchor = tuple(int(bounds[k]) for k in ("x1", "y1", "x2", "y2"))
            if _boxes_overlap(anchor, coords_raw):
                style["source_letter_box"] = anchor
        if (user_color_auto and _is_auto_style(lettering_c) and auto_text
                and _uses_auto_geometry(original) and source_image_loader is not None
                and not (active_color_profile or {}).get("gradient")):
            colour = parse_color(box_color, default=None)
            neutral_hint = colour is not None and max(colour) - min(colour) <= 12 and 35 <= max(colour) <= 175
            if ((neutral_hint and container in ("bubble", "spiky", "box"))
                    or (container == "free" and role_key == "sfx")):
                source = source_image_loader()
                ax1, ay1, ax2, ay2 = style.get("source_letter_box") or (x1, y1, x2, y2)
                if source is not None and 0 <= ax1 < ax2 <= source.width and 0 <= ay1 < ay2 <= source.height:
                    crop = np.asarray(source.crop((ax1, ay1, ax2, ay2)).convert("RGB"))
                    ink = source_ink_hex(crop[:, :, ::-1], (0, 0, crop.shape[1], crop.shape[0]))
                    if ink:
                        style["fill"] = ink
        if obj.get("letters_mask"):
            from app.mask_store import decode_mask_value

            decoded_letters = decode_mask_value(obj.get("letters_mask"))
            if decoded_letters is not None:
                style["letters"] = decoded_letters
                reg = obj.get("region")
                if _valid_region(reg) and decoded_letters.shape == (
                    int(reg["y2"]) - int(reg["y1"]),
                    int(reg["x2"]) - int(reg["x1"]),
                ):
                    style["letters_box"] = (
                        int(reg["x1"]),
                        int(reg["y1"]),
                        int(reg["x2"]),
                        int(reg["y2"]),
                    )
                else:
                    style["letters_box"] = coords_raw
        moved_by_user = user_position(original)
        if moved_by_user:
            style.update(bubble_mask=None, container=None, source_letter_box=None,
                         letters=None, letters_box=None)
        if oid in chambers:
            style.update(bubble_mask=chambers[oid], letters=None, letters_box=None)
        mask = style["bubble_mask"]
        outline_box = _bubble_outline(mask, coords_raw)
        stale_outline = (isinstance(mask, dict) and mask.get("kind") == "bubble" and outline_box is None
                         and not original.get("seam_owner"))
        authorized = None
        anchors = ()
        if anchors_by_id is not None:
            anchors = anchors_by_id.get(oid, ())
            if pair:
                anchors += tuple(spot for item in joined_ids for spot in anchors_by_id[item])
        if anchors:
            anchor = anchor_union(anchors)
            seam = bool(original.get("seam_owner") or coords[1] < core[0] or coords[3] > core[1])
            outline_limits = (min(0, coords[1]), max(img_h, coords[3])) if seam else core
            mask = _placement_outline(clean, anchors, style["bubble_mask"], source_image_loader,
                                      verified_balloon or bool(pair) or oid in chambers,
                                      coords if not seam and container in ("bubble", "spiky") else None,
                                      outline_limits)
            group_ids = {oid, *joined_ids} if pair else {oid}
            if mask is not None and any(
                    other_id not in group_ids and any(anchor_coverage(mask, spot) >= .72 for spot in spots)
                    for other_id, spots in anchors_by_id.items()):
                mask = clip_bubble_mask(mask, ink_window(anchor, coords))
            if mask is None:
                anchor_area = max(1, (anchor[2] - anchor[0]) * (anchor[3] - anchor[1]))
                coords_area = max(1, (coords[2] - coords[0]) * (coords[3] - coords[1]))
                if not (
                    (seam and _boxes_overlap(coords, anchor))
                    or container in ("box", "screen", "bubble", "spiky")
                    or anchor_area < 0.50 * coords_area
                ):
                    coords = ink_window(anchor, coords)
            if coords[0] >= coords[2] or coords[1] >= coords[3]:
                if verdicts is not None:
                    verdicts.append({"id": oid, "guard": "placement_unverified", "review": True})
                continue
            style.update(source_letter_box=anchor, verified_balloon=mask is not None, bubble_mask=mask, lock_frame=True)
            has_outline = mask is not None
            authorized = tuple(mask["bbox"]) if has_outline else coords
        else:
            if stale_outline:
                style["bubble_mask"] = mask = {"kind": "rect", "bbox": list(coords)}
            has_outline = isinstance(mask, dict) and mask.get("kind") == "bubble"
            anchor = style.get("source_letter_box") or coords
            if (container in ("bubble", "spiky") and _uses_auto_geometry(obj)
                    and core[0] <= anchor[1] < anchor[3] <= core[1]
                    and (not has_outline or (not verified_balloon and oid not in chambers
                                              and anchor_coverage(mask, anchor) < .75))):
                expected = (mask,) if has_outline else ()
                recovered = verified_outline(clean, (anchor,), expected, source_image_loader)
                if recovered is not None and core[0] <= recovered["bbox"][1] < recovered["bbox"][3] <= core[1]:
                    style["bubble_mask"] = mask = recovered
                    has_outline = True
                elif not has_outline:
                    style["bubble_mask"] = {"kind": "rect", "bbox": list(coords)}
            elif mask is None and style["container"] in ("bubble", "spiky"):
                shared = _derived_mask(clean, coords, style.get("source_letter_box"))
                if isinstance(shared, dict) and shared.get("kind") == "bubble":
                    style["bubble_mask"] = shared
                    has_outline = True
        outline_box = style["bubble_mask"]["bbox"] if has_outline else coords
        fit_y1, fit_y2 = min(coords[1], outline_box[1]), max(coords[3], outline_box[3])
        style["compose_layout"] = (compose_layout and h_align == "center" and v_align == "middle"
                                   and (core_y1 or 0) <= fit_y1 < fit_y2 <= (core_y2 or img_h))
        render_base = clean if style["compose_layout"] else image
        attempts = [coords] if authorized is not None or has_outline or moved_by_user else fit_attempts(obj, coords)
        render_text = translation.strip()
        if auto_text and container in ("bubble", "spiky"):
            render_text = _reflow_dialogue(render_text)
        if (container == "free"
                and "\n" in render_text and obj.get("source_cap_px")
                and not (isinstance(box_size, (int, float)) and box_size > 0)
                and not (isinstance(box_size, str) and box_size.isdigit()
                         and int(box_size) > 0)):
            joined = " ".join(render_text.split())
            probe_stats: dict = {}
            try:
                _render_in_region(render_base.copy(), joined, attempts[-1],
                                  stats=probe_stats, **style)
            except ValueError:
                probe_stats = {}
            if probe_stats.get("lines", 0) and probe_stats["lines"] < len(render_text.splitlines()):
                render_text = joined
        try:
            rendered = []
            last_error = None
            floor_only = True
            for attempt, box in enumerate(attempts):
                attempt_stats: dict = {"attempt_box": box}
                try:
                    candidate = _render_in_region(render_base.copy(), render_text, box,
                                                  stats=attempt_stats, **style)
                except ValueError as exc:
                    if "minimum readable" not in str(exc):
                        floor_only = False
                    last_error = exc
                    continue
                letter_spot = style.get("source_letter_box")
                ink_box = attempt_stats.get("ink_box")
                spills = False
                if (attempt_stats.get("frame_source") != "mask"
                        and letter_spot is not None and ink_box is not None):
                    ix1, iy1, ix2, iy2 = ink_box
                    ox1, oy1 = max(letter_spot[0], ix1), max(letter_spot[1], iy1)
                    ox2, oy2 = min(letter_spot[2], ix2), min(letter_spot[3], iy2)
                    overlap = max(0, ox2 - ox1) * max(0, oy2 - oy1)
                    spills = overlap * 2 < (ix2 - ix1) * (iy2 - iy1)
                attempt_stats["spills_letter_box"] = spills
                rendered.append((attempt_stats.get("lines", 0), attempt, candidate, attempt_stats))
            if not rendered:
                if pair:
                    seen_keys.discard(dup_key)
                    pending.appendleft(original)
                    continue
                if floor_only and last_error is not None:
                    logger.warning(
                        "Chapter {} page {} object {} below readable floor, left for review",
                        req.chapter_id, req.page_index, oid)
                    if verdicts is not None:
                        guard = "below_readable_floor"
                        if anchors:
                            guard = "balloon_overset" if has_outline else "placement_overset"
                        elif verified_balloon:
                            guard = "balloon_overset"
                        verdicts.append({"id": oid, "guard": guard,
                                         "review": True})
                    continue
                if last_error is not None:
                    raise last_error
            else:
                painted = False
                best = None
                verdict: dict = {"id": oid}
                if render_text != translation.strip():
                    verdict["reflowed" if container in ("bubble", "spiky") else "joined_free"] = True
                for item in _rank_renders(rendered):
                    candidate = item[2]
                    item_stats = item[3] if len(item) > 3 and isinstance(item[3], dict) else {}
                    mask, box = _render_diff(render_base, candidate, item_stats)
                    if authorized is not None and not _ink_inside_placement(mask, box, authorized, style["bubble_mask"]):
                        verdict.update(guard="placement_outside_bounds", review=True)
                        continue
                    overlap = 0
                    if box is not None:
                        for px1, py1, px2, py2 in placed_boxes:
                            overlap += (max(0, min(box[2], px2) - max(box[0], px1))
                                        * max(0, min(box[3], py2) - max(box[1], py1)))
                    collision = int((mask & placed_mask).sum()) if box is not None else 0
                    cand_frame = item_stats.get("frame")
                    dedup_frame = False
                    shared_frame = False
                    for placed_frame, placed_words in placed_frames:
                        if _frame_iou(cand_frame, placed_frame) >= 0.9:
                            if placed_words == translation.strip() and not verified_balloon:
                                dedup_frame = True
                            else:
                                shared_frame = True
                    if dedup_frame:
                        verdict["guard"] = "dedup_frame"
                        continue
                    if shared_frame:
                        verdict["shared_frame"] = True
                    guard = "clear"
                    if item_stats.get("layout_score") is not None and overlap:
                        reflow = _fit_around_ink(render_base, render_text, attempts[item[1]],
                                                 style, placed_boxes, placed_mask)
                        if reflow is not None and (authorized is None or _ink_inside_placement(
                                reflow[1], reflow[2], authorized, style["bubble_mask"])):
                            candidate, mask, box, item_stats = reflow
                            cand_frame = item_stats.get("frame")
                            collision = overlap = 0
                            guard = "recomposed"
                        else:
                            collision = max(collision, overlap)
                    if collision == 0:
                        _paste_ink(image, candidate, mask, box)
                        if box is not None:
                            placed_boxes.append(box)
                        placed_mask |= mask
                        if cand_frame is not None:
                            placed_frames.append((cand_frame, translation.strip()))
                        verdict.update(guard=guard, bbox_overlap=overlap)
                        verdict.update(_stats_verdict_fields(item_stats, cand_frame, item[1]))
                        painted = True
                        break
                    key = (collision, overlap)
                    if best is None or key < best[0]:
                        best = (key, candidate, mask, box, item_stats, item[1])
                if not painted and best is not None and best[4].get("layout_score") is None and authorized is None:
                    _, candidate, mask, box, item_stats, best_attempt = best
                    cand_frame = item_stats.get("frame")
                    cand_fit = item_stats.get("fit_box")
                    dy = None
                    if box is not None:
                        dy = _clearing_shift(
                            box, placed_boxes, img_h, frame=cand_frame,
                            anchor_box=cand_fit,
                            placed_tops=[b[1] for b in placed_boxes],
                            cand_top=box[1])
                    if box is not None and dy and abs(dy) <= INK_SHIFT_CAP:
                        base_box = item_stats.get("attempt_box") or cand_fit or cand_frame or box
                        shift_box = (base_box[0], base_box[1] + dy,
                                     base_box[2], base_box[3] + dy)
                        shifted_res = _rerender_non_colliding(
                            image, render_text, shift_box, style, placed_mask,
                            frame_bound=cand_frame)
                        if shifted_res is not None:
                            scandidate, smask, sbox_ink, sstats = shifted_res
                            image.paste(scandidate)
                            placed_boxes.append(sbox_ink)
                            placed_mask |= smask
                            sframe = sstats.get("frame") or cand_frame
                            if sframe is not None:
                                placed_frames.append((sframe, translation.strip()))
                            verdict.update({"guard": "shifted", "shift_dy": dy})
                            verdict.update(_stats_verdict_fields(sstats, sframe, best_attempt))
                            painted = True
                    if not painted:
                        stacked = None
                        if cand_frame is not None and cand_fit is not None:
                            blockers = [b for b in placed_boxes
                                        if b[0] < cand_fit[2] and cand_fit[0] < b[2]
                                        and b[1] < cand_fit[3] and cand_fit[1] < b[3]]
                            if blockers:
                                below_top = max(b[3] for b in blockers) + INK_GAP
                                above_bottom = min(b[1] for b in blockers) - INK_GAP
                                tries = []
                                if below_top < cand_frame[3]:
                                    tries.append((cand_frame[0], below_top,
                                                  cand_frame[2], cand_frame[3]))
                                if cand_frame[1] < above_bottom:
                                    tries.append((cand_frame[0], cand_frame[1],
                                                  cand_frame[2], above_bottom))
                                frame_mid = (cand_frame[1] + cand_frame[3]) // 2
                                block_mid = (min(b[1] for b in blockers)
                                             + max(b[3] for b in blockers)) // 2
                                if frame_mid < block_mid:
                                    tries.reverse()
                                left_end = min(b[0] for b in blockers) - INK_GAP
                                right_start = max(b[2] for b in blockers) + INK_GAP
                                if cand_frame[0] < left_end:
                                    tries.append((cand_frame[0], cand_frame[1],
                                                  left_end, cand_frame[3]))
                                if right_start < cand_frame[2]:
                                    tries.append((right_start, cand_frame[1],
                                                  cand_frame[2], cand_frame[3]))
                                for sbox in tries:
                                    res = _rerender_non_colliding(
                                        image, render_text, sbox, style,
                                        placed_mask, frame_bound=cand_frame)
                                    if res is not None:
                                        stacked = res
                                        break
                        if stacked is not None:
                            scandidate, smask, sbox_ink, sstats = stacked
                            image.paste(scandidate)
                            placed_boxes.append(sbox_ink)
                            placed_mask |= smask
                            sframe = sstats.get("frame") or cand_frame
                            placed_frames.append((sframe, translation.strip()))
                            verdict.update({"guard": "stacked"})
                            verdict.update(_stats_verdict_fields(sstats, sframe, best_attempt))
                            painted = True
                        else:
                            logger.warning(
                                "Chapter {} page {} object {} skipped due to unresolvable collision",
                                req.chapter_id, req.page_index, oid)
                            verdict.update({
                                "guard": "skipped_collision",
                                "review": True,
                            })
                            verdict.update(_stats_verdict_fields(item_stats, cand_frame, best_attempt))
                            verdict["fallbacks"] = (
                                list(verdict.get("fallbacks") or []) + ["skipped_collision"]
                            )
                if not painted and best is not None and (authorized is not None or best[4].get("layout_score") is not None):
                    verdict.update(guard="skipped_collision", review=True)
                    verdict.update(_stats_verdict_fields(best[4], best[4].get("frame"), best[5]))
                if stale_outline:
                    verdict["stale_outline"] = True
                if oid in chambers:
                    verdict["owned_chamber"] = True
                if not painted and pair:
                    seen_keys.discard(dup_key)
                    pending.appendleft(original)
                    continue
                if verdicts is not None:
                    verdicts.append(verdict)
                if painted:
                    rendered_count += 1
                    if pair:
                        consumed.update(joined_ids)
                        rendered_count += len(joined_ids)
                        if verdicts is not None:
                            verdicts.extend({"id": other_id, "guard": "joined_in_bubble", "with": oid}
                                            for other_id in joined_ids)
                continue
        except Exception as e:
            logger.opt(exception=True).error(
                "Chapter {} page {} object {} operation 'render_text_in_box' failed: {}",
                req.chapter_id, req.page_index, oid, e,
            )
            raise HTTPException(500, f"Chèn chữ thất bại (vùng {oid})") from e

    return rendered_count
