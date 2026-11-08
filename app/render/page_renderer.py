from pathlib import Path

import numpy as np
from PIL import Image
from fastapi import HTTPException

from app.logging_config import logger
from app.render.text_renderer import render_text_in_box
from app.schemas import RenderRequest


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
    """Attempts best first, by the chooser's rule below."""
    def stats_of(item):
        return item[3] if len(item) > 3 and isinstance(item[3], dict) else {}

    corrected = [item for item in rendered
                 if stats_of(item).get("fit_box") != stats_of(item).get("attempt_box")
                 and stats_of(item).get("fit_box") is not None]
    pool = corrected if corrected else rendered
    pool.sort(key=lambda item: (bool(stats_of(item).get("below_share")), item[0],
                                -(stats_of(item).get("size") or 0), item[1]))
    return pool


def _choose_render(rendered: list) -> Image.Image:
    """The attempt that letters fewest lines; ties letter largest.

    Ranking is fit and readability only: an attempt the bubble
    measurement never corrected — its fit box is the raw region — is not
    trusted over one the measurement placed inside the bubble, however
    few lines it letters: the raw box can straddle an outline the
    measured box avoids. Below-share attempts are a last resort, so a
    text that only fits small still letters rather than vanishing.
    Among equal line counts, the larger size wins: a tight box that wraps
    to the same lines at a smaller size only strands the text small in a
    big bubble. The ladder's tight-to-region order is generation order,
    not preference — it survives only as the final tiebreak. Separation
    is not this ranking's job: the guard below vetoes colliding attempts
    in ranked order instead of re-ranking them.
    """
    return _rank_renders(rendered)[0][2]


def _ink_mask(base: Image.Image, candidate: Image.Image):
    """Pixels where a candidate differs from the image it would paint onto."""
    return (np.abs(np.asarray(candidate).astype(int)
                   - np.asarray(base).astype(int)).max(axis=2) > 0)


def _mask_bbox(mask):
    """Exclusive-end bbox of a mask, or None when it holds no ink."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


INK_GAP = 4  # neighbouring texts keep this clear space between ink boxes
INK_SHIFT_CAP = 64  # a nudge past this stops being a separation and starts relocating the text


def _clearing_shift(cand_box, placed_boxes, height: int, frame=None, anchor_box=None,
                    placed_tops=None, cand_top=None):
    """Smallest vertical move taking an ink box clear of placed ink, or None.

    A move counts only when the moved box clears every placed box, not just
    the one it collided with, and stays on the canvas; a move that trades
    one collision for another is no move. It must also stay inside the
    object's frame — judged on the fit box, the unit the fit decided —
    and must not carry the block's top past a neighbour's top: a shift
    that inverts which block reads first trades an overlap for a
    reading-order break. Without a frame the canvas check alone stands,
    and the caller records that.
    """
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


def _frame_iou(a, b) -> float:
    """Overlap share of two frame boxes, for same-frame identity."""
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
    ):
        return ocr_region
    return region


LETTER_MARGIN = 0.15  # room round the source lettering, as a share of its shorter side
FIT_INSET = 0.04  # of the lettering width, pulled in on each side of a roomier attempt
# A box that letters the translation below this share of the source size gives way to a roomier one.
SOURCE_SIZE_SHARE = 0.85


def letter_box(obj: dict, region: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    """Where the source lettering was, with some room, inside the region; None when unknown."""
    bounds = obj.get("letter_bounds")
    if not _valid_region(bounds):
        return None
    x1, y1, x2, y2 = (int(bounds[k]) for k in ("x1", "y1", "x2", "y2"))
    margin = max(8, int(LETTER_MARGIN * min(x2 - x1, y2 - y1)))
    box = (max(region[0], x1 - margin), max(region[1], y1 - margin),
           min(region[2], x2 + margin), min(region[3], y2 + margin))
    return box if box[0] < box[2] and box[1] < box[3] and box != tuple(region) else None


def fit_attempts(obj: dict, coords: tuple[int, int, int, int]) -> list:
    """Boxes tried for an object's lettering: the source lettering's spot, then roomier.

    A bubble gives room downward before it gives any sideways, so the
    tall attempt keeps the lettering's width and takes the region's
    height. Wider attempts are pulled in on each side: a box flush with
    the region edge letters onto the outline where the interior narrows.
    The frame is still measured first inside each attempt, so this ladder
    only chooses starting boxes, never the frame itself.
    """
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
    """Letter ``box`` even where it runs past the top or bottom of the slice, clipped to the slice.

    A text crossing a slice cut is lettered by both slices; each lays it out in the
    whole box, so the two halves meet exactly.
    """
    width, height = image.size
    x1, x2 = max(0, box[0]), min(width, box[2])
    top, bottom = max(0, -box[1]), max(0, box[3] - height)
    if not (top or bottom):
        return render_text_in_box(image, text, (x1, box[1], x2, box[3]), **style)
    pixels = np.asarray(image)
    pad = ((top, bottom), (0, 0)) + (((0, 0),) if pixels.ndim == 3 else ())
    canvas = Image.fromarray(np.pad(pixels, pad, mode="edge"), mode=image.mode)
    canvas = render_text_in_box(canvas, text, (x1, box[1] + top, x2, box[3] + top), **style)
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
) -> int:
    """Paint ``text_objects`` onto ``image`` in one decision order.

    Frame first (measured in the fit), then fit inside it, then rank
    attempts by fit and readability, then separate by pixel ink, then
    paint. Each stage consumes the previous stage's output: the fit
    never re-decides the frame, the ranking never re-decides the fit,
    and the guard vetoes colliding attempts in ranked order instead of
    re-ranking them. Every fallback a stage takes is recorded in
    ``verdicts`` when given, one entry per object.
    """
    rendered_count = 0
    img_w, img_h = image.size
    seen_keys = set()
    placed_mask = np.zeros((img_h, img_w), dtype=bool)
    placed_boxes = []
    placed_frames: list[tuple] = []
    # Paint in reading order, not manifest order: a block painted first
    # owns its ink, and a later block fits around it instead of the other
    # way round. Ties keep manifest order, so identical duplicates are
    # unaffected by the sort. Framed blocks go before frameless text:
    # a bubble's rays and bands measure the image it is drawn on, so a
    # frameless glyph painted first reads as an outline to them, while
    # a frameless fit is a rectangle fit and dodges placed ink cleanly.
    def _order_key(obj):
        region = _render_region_for_text_object(obj)
        framed = 0 if str(obj.get("container") or "").lower() in (
            "bubble", "spiky", "box", "screen") else 1
        try:
            return (framed, int(region["y1"]), int(region["x1"]))
        except (KeyError, TypeError, ValueError):
            return (framed, 0, 0)

    for obj in sorted(text_objects, key=_order_key):
        oid = obj.get("id")
        if not oid:
            continue
        translation = style_get(req.translations, oid)
        if translation is None or translation == "":
            translation = obj.get("translation", "")
        if not (translation or "").strip():
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
        # The same region lettered twice with the same words is one text
        # rendered over itself, so the second copy is skipped outright.
        dup_key = (coords_raw, translation.strip())
        if dup_key in seen_keys:
            if verdicts is not None:
                verdicts.append({"id": oid, "guard": "dedup_exact"})
            continue
        seen_keys.add(dup_key)
        # The layout uses the whole region even past the slice edge; only drawing is clipped.
        coords = (x1, coords_raw[1], x2, coords_raw[3])
        # A system window cut by a slice edge is the exception: its text
        # centred on the whole window lands on the frame's border in the
        # cropped page, so the window is fitted to the kept part instead.
        # A sliver of window keeps nothing: its full text lives on the
        # neighbouring slice's page, where the kept part is bigger.
        if (obj.get("container") == "screen" and core_y1 is not None and core_y2 is not None):
            clipped_y1 = max(coords_raw[1], core_y1)
            clipped_y2 = min(coords_raw[3], core_y2)
            if clipped_y2 - clipped_y1 < 0.5 * (coords_raw[3] - coords_raw[1]):
                if verdicts is not None:
                    verdicts.append({"id": oid, "guard": "screen_sliver_skipped"})
                continue
            coords = (x1, clipped_y1, x2, clipped_y2)

        obj_style = obj.get("style") or {}
        box_color = _resolve_ocr_style(
            style_get(colors_dict, oid),
            obj_style.get("color", "auto"),
            obj.get("ocr_text_color") or obj.get("lettering_color"),
            "auto",
        )
        box_font = style_get(fonts_dict, oid)
        if box_font is None:
            box_font = obj_style.get("font", "default")
        box_size = _resolve_ocr_style(
            style_get(font_sizes_dict, oid),
            obj_style.get("fontSize", "auto"),
            obj.get("ocr_font_size"),
            "auto",
        )
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
            source_cap_px=obj.get("source_cap_px"),
            container=obj.get("container"),
            enlarge=bool(obj.get("enlarge")),
        )
        bounds = obj.get("letter_bounds")
        if _valid_region(bounds):
            style["source_letter_box"] = tuple(int(bounds[k]) for k in ("x1", "y1", "x2", "y2"))
        # The lettering goes where the source lettering was, then roomier
        # boxes; the attempt that letters fewest lines wins, ties largest.
        attempts = fit_attempts(obj, coords)
        # Frameless text only: a stored break sized for a narrow box can
        # strand a wide line as two small ones. The joined line is tried
        # on the roomiest attempt at source share; it wins only when it
        # fits there at share and actually drops a line, otherwise the
        # stored breaks stand exactly as before.
        render_text = translation.strip()
        if (str(obj.get("container") or "").lower() == "free"
                and "\n" in render_text and obj.get("source_cap_px")
                and not (isinstance(box_size, (int, float)) and box_size > 0)
                and not (isinstance(box_size, str) and box_size.isdigit()
                         and int(box_size) > 0)):
            joined = " ".join(render_text.split())
            probe_stats: dict = {}
            try:
                _render_in_region(image.copy(), joined, attempts[-1],
                                  min_source_share=SOURCE_SIZE_SHARE,
                                  stats=probe_stats, **style)
            except ValueError:
                probe_stats = {}
            if probe_stats.get("lines", 0) and probe_stats["lines"] < len(render_text.splitlines()):
                render_text = joined
        try:
            rendered = []
            last_error = None
            floor_only = True
            ratio_only = True
            for attempt, box in enumerate(attempts):
                share = SOURCE_SIZE_SHARE if attempt < len(attempts) - 1 else 0.0
                attempt_stats: dict = {"attempt_box": box}
                try:
                    candidate = _render_in_region(image.copy(), render_text, box,
                                                  min_source_share=share, stats=attempt_stats, **style)
                except ValueError as exc:
                    if "well below" in str(exc) and share:
                        # A small fitting attempt beats no attempt at all;
                        # it only wins when nothing at source share fits.
                        attempt_stats["below_share"] = True
                        try:
                            candidate = _render_in_region(image.copy(), render_text, box,
                                                          min_source_share=0.0, stats=attempt_stats, **style)
                        except ValueError as exc2:
                            if "minimum readable" not in str(exc2):
                                floor_only = False
                            if "source floor share" not in str(exc2):
                                ratio_only = False
                            last_error = exc2
                            continue
                    else:
                        if "minimum readable" not in str(exc):
                            floor_only = False
                        if "source floor share" not in str(exc):
                            ratio_only = False
                        last_error = exc
                        continue
                rendered.append((attempt_stats.get("lines", 0), attempt, candidate, attempt_stats))
            if not rendered:
                # Text that fits nowhere at the readable floor is flagged,
                # not shrunk past the floor: the raise below carries the
                # object id, so the caller restores its original pixels
                # and lists it for review instead of leaving a blank hole.
                if floor_only and last_error is not None:
                    logger.warning(
                        "Chapter {} page {} object {} below readable floor, left for review",
                        req.chapter_id, req.page_index, oid)
                    if verdicts is not None:
                        verdicts.append({"id": oid, "guard": "below_readable_floor",
                                         "review": True})
                elif ratio_only and last_error is not None:
                    logger.warning(
                        "Chapter {} page {} object {} below source floor share, left for review",
                        req.chapter_id, req.page_index, oid)
                    if verdicts is not None:
                        verdicts.append({"id": oid, "guard": "below_source_floor",
                                         "review": True})
                if last_error is not None:
                    raise last_error
            else:
                # The caller renders in place, so paint the chosen attempt
                # back onto the image it passed in. The guard vetoes in
                # ranked order: the first attempt whose ink pixels touch no
                # placed ink is painted. Bounding boxes are the reader's
                # eye in neither direction — tall blocks in overlapping
                # bubbles share y-bands at different x, and glyphs can
                # collide inside disjoint-looking boxes — so boxes judge
                # nothing here, they only break ties between colliding
                # attempts. A candidate for a frame already painted with
                # the same words is the same text twice, whatever its
                # region said, and gives way to the next attempt.
                painted = False
                best = None
                verdict: dict = {"id": oid}
                if render_text != translation.strip():
                    verdict["joined_free"] = True
                for item in _rank_renders(rendered):
                    candidate = item[2]
                    item_stats = item[3] if len(item) > 3 and isinstance(item[3], dict) else {}
                    mask = _ink_mask(image, candidate)
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
                    if collision == 0:
                        image.paste(candidate)
                        if box is not None:
                            placed_boxes.append(box)
                        placed_mask |= mask
                        if cand_frame is not None:
                            placed_frames.append((cand_frame, translation.strip()))
                        verdict.update({
                            "guard": "clear",
                            "attempt": item[1],
                            "size": item_stats.get("size"),
                            "lines": item_stats.get("lines"),
                            "frame": cand_frame,
                            "frame_source": item_stats.get("frame_source"),
                            "lobe": item_stats.get("lobe"),
                            "fallbacks": item_stats.get("fallbacks", []),
                            "stages_skipped": item_stats.get("stages_skipped", []),
                            "band_narrowing": item_stats.get("band_narrowing", {}),
                            "below_share": bool(item_stats.get("below_share")),
                            "fill_fallback": item_stats.get("fill_fallback"),
                            "bbox_overlap": overlap,
                        })
                        painted = True
                        break
                    key = (collision, overlap)
                    if best is None or key < best[0]:
                        best = (key, candidate, mask, box, item_stats, item[1])
                if not painted and best is not None:
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
                        shifted = (box[0], box[1] + dy, box[2], box[3] + dy)
                        canvas = np.asarray(image).copy()
                        ys, xs = np.nonzero(mask)
                        canvas[ys + dy, xs] = np.asarray(candidate)[ys, xs]
                        image.paste(Image.fromarray(canvas))
                        placed_boxes.append(shifted)
                        moved = np.zeros_like(mask)
                        moved[ys + dy, xs] = True
                        placed_mask |= moved
                        if cand_frame is not None:
                            placed_frames.append((cand_frame, translation.strip()))
                        verdict.update({
                            "guard": "shifted", "shift_dy": dy,
                            "attempt": best_attempt,
                            "size": item_stats.get("size"),
                            "lines": item_stats.get("lines"),
                            "frame": cand_frame,
                            "frame_source": item_stats.get("frame_source"),
                            "lobe": item_stats.get("lobe"),
                            "fallbacks": item_stats.get("fallbacks", []),
                            "stages_skipped": item_stats.get("stages_skipped", []),
                            "band_narrowing": item_stats.get("band_narrowing", {}),
                            "below_share": bool(item_stats.get("below_share")),
                            "fill_fallback": item_stats.get("fill_fallback"),
                        })
                        painted = True
                    else:
                        # No shift clears the ink, so the block is
                        # re-fitted into the free part of its own frame
                        # — below the blocking ink when the frame sits
                        # below it, above when it sits above — instead
                        # of being painted over its neighbour. A stacked
                        # fit must read clear of placed ink and stay
                        # inside the frame, or it is no fit at all.
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
                                for sbox in tries:
                                    if sbox[2] - sbox[0] < 16 or sbox[3] - sbox[1] < 16:
                                        continue
                                    sstats: dict = {"attempt_box": sbox}
                                    scandidate = None
                                    for share in (SOURCE_SIZE_SHARE, 0.0):
                                        try:
                                            scandidate = _render_in_region(
                                                image.copy(), render_text, sbox,
                                                min_source_share=share, stats=sstats, **style)
                                            break
                                        except ValueError:
                                            scandidate = None
                                    if scandidate is None:
                                        continue
                                    smask = _ink_mask(image, scandidate)
                                    sbox_ink = _mask_bbox(smask)
                                    if sbox_ink is None:
                                        continue
                                    if int((smask & placed_mask).sum()) != 0:
                                        continue
                                    if not (cand_frame[0] <= sbox_ink[0]
                                            and cand_frame[1] <= sbox_ink[1]
                                            and sbox_ink[2] <= cand_frame[2]
                                            and sbox_ink[3] <= cand_frame[3]):
                                        continue
                                    stacked = (scandidate, smask, sbox_ink, sstats)
                                    break
                        if stacked is not None:
                            scandidate, smask, sbox_ink, sstats = stacked
                            image.paste(scandidate)
                            placed_boxes.append(sbox_ink)
                            placed_mask |= smask
                            sframe = sstats.get("frame") or cand_frame
                            placed_frames.append((sframe, translation.strip()))
                            verdict.update({
                                "guard": "stacked",
                                "attempt": best_attempt,
                                "size": sstats.get("size"),
                                "lines": sstats.get("lines"),
                                "frame": sframe,
                                "frame_source": sstats.get("frame_source"),
                                "lobe": sstats.get("lobe"),
                                "fallbacks": sstats.get("fallbacks", []),
                                "stages_skipped": sstats.get("stages_skipped", []),
                                "band_narrowing": sstats.get("band_narrowing", {}),
                                "below_share": bool(sstats.get("below_share")),
                                "fill_fallback": sstats.get("fill_fallback"),
                            })
                            painted = True
                        else:
                            # No in-frame, order-preserving move clears
                            # the ink, so the collision is painted and
                            # labelled rather than hidden behind a
                            # worse relocation.
                            logger.warning(
                                "Chapter {} page {} object {} paints colliding ink (no clearing shift)",
                                req.chapter_id, req.page_index, oid)
                            image.paste(candidate)
                            if box is not None:
                                placed_boxes.append(box)
                            placed_mask |= mask
                            if cand_frame is not None:
                                placed_frames.append((cand_frame, translation.strip()))
                            verdict.update({
                                "guard": "painted_colliding",
                                "attempt": best_attempt,
                                "size": item_stats.get("size"),
                                "lines": item_stats.get("lines"),
                                "frame": cand_frame,
                                "frame_source": item_stats.get("frame_source"),
                                "lobe": item_stats.get("lobe"),
                                "fallbacks": (item_stats.get("fallbacks", [])
                                              + ["painted_colliding"]),
                                "stages_skipped": item_stats.get("stages_skipped", []),
                                "band_narrowing": item_stats.get("band_narrowing", {}),
                                "below_share": bool(item_stats.get("below_share")),
                                "fill_fallback": item_stats.get("fill_fallback"),
                            })
                            painted = True
                if verdicts is not None:
                    verdicts.append(verdict)
                if not painted:
                    # Every attempt was the same frame and words as a
                    # placed object: nothing painted, nothing counted.
                    continue
                if painted:
                    rendered_count += 1
                continue
            rendered_count += 1
        except Exception as e:
            logger.opt(exception=True).error(
                "Chapter {} page {} object {} operation 'render_text_in_box' failed: {}",
                req.chapter_id, req.page_index, oid, e,
            )
            raise HTTPException(500, f"Chèn chữ thất bại (vùng {oid})") from e

    return rendered_count
