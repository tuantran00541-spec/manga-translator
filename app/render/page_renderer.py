import re
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image
from fastapi import HTTPException

from app.logging_config import logger
from app.detector.bubble_mask import clip_bubble_mask, find_waist, mask_row_spans, shift_bubble_mask
from app.render.font_metrics import parse_color
from app.render.source_size import source_cap_px
from app.render.text_fitting import _derived_mask, _mask_has_narrow_waist, _recover_bubble_mask
from app.render.text_renderer import render_text_in_box
from app.schemas import RenderRequest
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


INK_GAP = 4
INK_SHIFT_CAP = 64


def _fit_around_ink(base, text, box, style, placed_boxes, placed_mask):
    if str(style["bg_color"] or "").lower() not in ("", "none", "transparent"):
        return None
    reserved = tuple((x1 - INK_GAP, y1 - INK_GAP, x2 + INK_GAP, y2 + INK_GAP)
                     for x1, y1, x2, y2 in placed_boxes)
    stats = {"attempt_box": box}
    try:
        candidate = _render_in_region(base.copy(), text, box, stats=stats, reserved_boxes=reserved, **style)
    except ValueError:
        return None
    mask = _ink_mask(base, candidate)
    ink = _mask_bbox(mask)
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
    smask = _ink_mask(image, scandidate)
    sbox_ink = _mask_bbox(smask)
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
    fields = ("style", "container", "typography_role", "speaker", "enlarge")
    if (any(obj.get(field) != other.get(field) for field in fields)
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
    return ac is None or bc is None or max(abs(x - y) for x, y in zip(ac, bc)) <= 24


def _fragment_outline(image, first, second, source_image_loader):
    obj, a, am, text = first
    _, b, bm, other_text = second
    min_height = min(a[3] - a[1], b[3] - b[1])
    y_overlap = min(a[3], b[3]) - max(a[1], b[1])
    union = (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
    if am is not None and bm is not None and y_overlap >= 0.65 * min_height:
        left, right = sorted((a, b))
        ma, mb = sorted((am, bm))
        if (len(text.split()) > 4 or len(other_text.split()) > 4
                or not -0.1 * min_height <= right[0] - left[2] <= 0.6 * min_height
                or abs(ma[2] - mb[0]) > 4 or max(abs(ma[1] - mb[1]), abs(ma[3] - mb[3])) > 8):
            return None
        mask = _derived_mask(image, union, union)
        outline = (ma[0], min(ma[1], mb[1]), mb[2], max(ma[3], mb[3]))
        if mask["kind"] != "bubble" or _frame_iou(mask["bbox"], outline) < 0.85 or find_waist(mask):
            return None
        cx1, cx2 = (left[0] + left[2]) // 2, (right[0] + right[2]) // 2
        cy = (max(a[1], b[1]) + min(a[3], b[3])) // 2
        if any(lo <= cx1 < cx2 < hi for lo, hi in mask_row_spans(mask, cy)):
            return mask, union, a == left
    elif am is not None and bm is None:
        x_overlap = min(a[2], b[2]) - max(a[0], b[0])
        if (not -0.15 * min_height <= b[1] - a[3] <= 0.5 * min_height
                or x_overlap < 0.65 * min(a[2] - a[0], b[2] - b[0])
                or not am[1] <= (b[1] + b[3]) // 2 < am[3]
                or _mask_has_narrow_waist(obj["bubble_mask"])):
            return None
        src = source_image_loader() if source_image_loader is not None else None
        recovered = _recover_bubble_mask(src or image, b, b)
        if recovered is not None and _frame_iou(recovered["bbox"], am) >= 0.9:
            return obj["bubble_mask"], union, True
    return None


def _paired_fragments(image, req, objects, style_maps, source_image_loader):
    candidates = []
    for obj in objects:
        oid = obj.get("id")
        mask = obj.get("bubble_mask")
        translation = _object_translation(req, obj)
        region = obj.get("letter_bounds")
        if not _valid_region(region):
            region = obj.get("region")
        if (not oid or not _uses_auto_geometry(obj) or not _valid_region(region)
                or obj.get("font_selection_mode") == "user"
                or str(obj.get("container") or "").lower() not in ("bubble", "spiky")
                or obj.get("typography_role") not in (None, "", "dialogue", "thought", "whisper", "shout", "dark_threat")
                or translation != obj.get("auto_translation") or not translation
                or len(translation) > 150
                or translation.strip().lower() in PLACEHOLDER_TRANSLATIONS):
            continue
        text = _reflow_dialogue(translation)
        if not text or "\n" in text:
            continue
        bounds = tuple(int(region[key]) for key in ("x1", "y1", "x2", "y2"))
        if not (0 <= bounds[1] < bounds[3] <= image.height):
            continue
        outline = _bubble_outline(mask, bounds)
        candidates.append((obj, bounds, outline, text))

    pairs = {}
    paired = set()
    for i, first in enumerate(candidates):
        obj, _, _, text = first
        oid = obj["id"]
        if oid in paired:
            continue
        for second in candidates[i + 1:]:
            other, _, _, other_text = second
            other_id = other["id"]
            if other_id in paired or text == other_text or not _same_lettering_style(obj, other, style_maps):
                continue
            proof = _fragment_outline(image, first, second, source_image_loader)
            if proof is None:
                continue
            combined, union, in_reading_order = proof
            region = dict(zip(("x1", "y1", "x2", "y2"), union))
            words = f"{text} {other_text}" if in_reading_order else f"{other_text} {text}"
            merged = {**obj, "region": region, "letter_bounds": region, "letters_mask": None,
                      "bubble_mask": combined, "translation": words}
            pairs[oid] = (merged, other_id)
            paired.update((oid, other_id))
            break
    return pairs


LETTER_MARGIN = 0.15
FIT_INSET = 0.04
SOURCE_SIZE_SHARE = 0.85


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
    canvas = render_text_in_box(canvas, text, (x1, box[1] + top, x2, box[3] + top), **padded_style)
    stats = style.get("stats")
    if isinstance(stats, dict) and stats.get("ink_box") is not None:
        ix1, iy1, ix2, iy2 = stats["ink_box"]
        ix1, iy1 = max(ix1, 0), max(iy1, top)
        ix2, iy2 = min(ix2, width), min(iy2, top + height)
        if ix2 > ix1 and iy2 > iy1:
            stats["ink_box"] = (ix1, iy1 - top, ix2, iy2 - top)
        else:
            stats["ink_box"] = None
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
) -> int:
    rendered_count = 0
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
        if box.get("removed"):
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
            image = render_text_in_box(
                image,
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
            )
            rendered_count += 1
        except Exception as e:
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
) -> int:
    rendered_count = 0
    img_w, img_h = image.size
    seen_keys = set()
    placed_mask = np.zeros((img_h, img_w), dtype=bool)
    placed_boxes = []
    placed_frames: list[tuple] = []
    def _order_key(obj):
        region = _render_region_for_text_object(obj)
        framed = 0 if str(obj.get("container") or "").lower() in (
            "bubble", "spiky", "box", "screen") else 1
        try:
            return (framed, int(region["y1"]), int(region["x1"]))
        except (KeyError, TypeError, ValueError):
            return (framed, 0, 0)

    objects = sorted(text_objects, key=_order_key)
    clean = image.copy() if len(objects) > 1 else image
    style_maps = (colors_dict, fonts_dict, font_sizes_dict, bolds_dict, stroke_w_dict, stroke_c_dict,
                  bg_colors_dict, radii_dict, horizontal_aligns_dict, vertical_aligns_dict)
    pairs = _paired_fragments(image, req, objects, style_maps, source_image_loader)
    consumed = set()
    pending = deque(objects)
    while pending:
        original = obj = pending.popleft()
        oid = obj.get("id")
        if not oid or oid in consumed:
            continue
        pair = pairs.pop(oid, None)
        if pair:
            obj, other_id = pair
            translation = obj["translation"]
        else:
            translation = _object_translation(req, obj)
        auto_text = bool(obj.get("auto_translation") and translation == obj["auto_translation"])
        translation = collapse_repeated_lines(translation)
        if not (translation or "").strip() or str(translation).strip().lower() in PLACEHOLDER_TRANSLATIONS:
            continue

        region = _render_region_for_text_object(obj)
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
        effective_source_cap = obj.get("source_cap_px")
        role_key = str(obj.get("typography_role") or "").lower()
        if effective_source_cap is None and isinstance(size_bands, dict) and role_key in size_bands:
            band = size_bands[role_key]
            if isinstance(band, (list, tuple)) and len(band) == 2:
                from app.render.size_bands import BAND_LO_FACTOR

                effective_source_cap = round(float(band[0]) / BAND_LO_FACTOR)
        requested_size = style_get(font_sizes_dict, oid, obj_style.get("fontSize", "auto"))
        if (effective_source_cap is None and container == "free" and auto_text
                and _uses_auto_geometry(obj) and _is_auto_style(requested_size)
                and source_image_loader is not None):
            source = source_image_loader()
            if source is not None:
                bgr = np.asarray(source)[y1:y2, x1:x2, ::-1]
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
            enlarge=bool(obj.get("enlarge")),
            bubble_mask=obj.get("bubble_mask"),
            color_profile=active_color_profile,
            adapt_contrast=user_color_auto,
            keep_compounds=(auto_text or pair is not None) and container in ("bubble", "spiky"),
            optical_size_hint=ocr_size_default,
        )
        bounds = obj.get("letter_bounds")
        if _valid_region(bounds):
            anchor = tuple(int(bounds[k]) for k in ("x1", "y1", "x2", "y2"))
            if _boxes_overlap(anchor, coords_raw):
                style["source_letter_box"] = anchor
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
        moved_by_user = (original.get("auto_generated") and _valid_region(original.get("auto_geometry"))
                         and original.get("region") != original["auto_geometry"] and not original.get("seam_owner"))
        if moved_by_user:
            style.update(bubble_mask=None, container=None, source_letter_box=None,
                         letters=None, letters_box=None)
        mask = style["bubble_mask"]
        outline_box = _bubble_outline(mask, coords_raw)
        stale_outline = (isinstance(mask, dict) and mask.get("kind") == "bubble" and outline_box is None
                         and not original.get("seam_owner"))
        if stale_outline:
            style["bubble_mask"] = mask = {"kind": "rect", "bbox": list(coords)}
        has_outline = isinstance(mask, dict) and mask.get("kind") == "bubble"
        if (not has_outline and container in ("bubble", "spiky")
                and _uses_auto_geometry(obj) and y1 == coords[1] and y2 == coords[3]):
            source_image = source_image_loader() if source_image_loader is not None else None
            mask = _recover_bubble_mask(source_image or clean, coords, style.get("source_letter_box"))
            if mask is None and source_image is not None:
                mask = _recover_bubble_mask(clean, coords, style.get("source_letter_box"))
            if mask is not None:
                style["bubble_mask"] = mask
                has_outline = True
            elif style["bubble_mask"] is None:
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
        attempts = [coords] if has_outline or moved_by_user else fit_attempts(obj, coords)
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
                        verdicts.append({"id": oid, "guard": "below_readable_floor",
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
                    mask = _ink_mask(render_base, candidate)
                    box = _mask_bbox(mask)
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
                            if placed_words == translation.strip():
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
                        if reflow is not None:
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
                if not painted and best is not None and best[4].get("layout_score") is None:
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
                            image, render_text, shift_box, style, placed_mask)
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
                if not painted and best is not None and best[4].get("layout_score") is not None:
                    verdict.update(guard="skipped_collision", review=True)
                    verdict.update(_stats_verdict_fields(best[4], best[4].get("frame"), best[5]))
                if stale_outline:
                    verdict["stale_outline"] = True
                if not painted and pair:
                    seen_keys.discard(dup_key)
                    pending.appendleft(original)
                    continue
                if verdicts is not None:
                    verdicts.append(verdict)
                if painted:
                    rendered_count += 1
                    if pair:
                        consumed.add(other_id)
                        rendered_count += 1
                        if verdicts is not None:
                            verdicts.append({"id": other_id, "guard": "joined_in_bubble", "with": oid})
                continue
            rendered_count += 1
        except Exception as e:
            logger.opt(exception=True).error(
                "Chapter {} page {} object {} operation 'render_text_in_box' failed: {}",
                req.chapter_id, req.page_index, oid, e,
            )
            raise HTTPException(500, f"Chèn chữ thất bại (vùng {oid})") from e

    return rendered_count
