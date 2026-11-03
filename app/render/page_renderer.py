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


FRAME_EDGE_STEP = 30  # a printed border steps this far in grey over two pixels; soft shading does not
FRAME_MIN_RUN = 0.5  # a border line spans at least this share of the region side
FRAME_MIN_SIZE = 0.5  # a detected frame keeps at least this share of the region
FRAME_INSET = 3  # room for the border line itself and its shadow


def _border_run(line: np.ndarray) -> int:
    """The longest run in a boolean row or column, bridging hairline gaps from compression."""
    best = run = gap = 0
    for on in line:
        if on:
            run += gap + 1
            gap = 0
            best = max(best, run)
        elif run:
            gap += 1
            if gap > 3:
                run = gap = 0
    return best


def detect_printed_frame(image: Image.Image, region: tuple[int, int, int, int]):
    """The printed rectangular border inside a detector region, or None when unsure.

    A detector region overshoots the box it found, so lettering sized to the region
    crosses the printed border. The border is the first long edge line met scanning
    inward from each region edge; all four sides must agree, or nothing is claimed
    and rendering falls back to the region.
    """
    x1, y1, x2, y2 = (int(v) for v in region)
    if x2 - x1 < 40 or y2 - y1 < 40:
        return None
    max_inset = min(120, max(20, int(0.25 * min(x2 - x1, y2 - y1))))
    gray = np.asarray(image.convert("L"))
    img_h, img_w = gray.shape[:2]
    cx1, cy1 = max(0, x1 - max_inset), max(0, y1 - max_inset)
    cx2, cy2 = min(img_w, x2 + max_inset), min(img_h, y2 + max_inset)
    crop = gray[cy1:cy2, cx1:cx2]
    rx1, ry1, rx2, ry2 = x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1
    # Borders are edges, not darkness: a thin light-grey line still steps sharply.
    vedge = np.zeros(crop.shape, dtype=bool)
    vedge[1:-1] = np.abs(crop[2:].astype(int) - crop[:-2].astype(int)) > FRAME_EDGE_STEP
    hedge = np.zeros(crop.shape, dtype=bool)
    hedge[:, 1:-1] = np.abs(crop[:, 2:].astype(int) - crop[:, :-2].astype(int)) > FRAME_EDGE_STEP

    def scan_side(fixed_range, along, horizontal: bool, inward: int):
        """The inner edge of the first long edge line met scanning inward, or None."""
        lo, hi = min(fixed_range), max(fixed_range)
        need = FRAME_MIN_RUN * (along[1] - along[0])

        def qualifies(pos):
            line = vedge[pos, along[0]:along[1]] if horizontal else hedge[along[0]:along[1], pos]
            return _border_run(line) >= need

        start = fixed_range[0] if inward > 0 else fixed_range[0] - 1
        for d in range(max_inset + 1):
            pos = start + d * inward
            if pos < lo or pos >= hi:
                return None
            if qualifies(pos):
                edge = pos
                # A double border reads as two lines a few pixels apart; the text
                # belongs inside the inner one, so keep walking while lines recur.
                while True:
                    nxt = edge + inward
                    while nxt >= lo and nxt < hi and not qualifies(nxt):
                        nxt += inward
                        if abs(nxt - edge) > 12:
                            break
                    if nxt < lo or nxt >= hi or abs(nxt - edge) > 12 or not qualifies(nxt):
                        break
                    edge = nxt
                    while True:
                        nxt = edge + inward
                        if nxt < lo or nxt >= hi or not qualifies(nxt):
                            break
                        edge = nxt
                return edge + 1 if inward > 0 else edge
        return None

    top = scan_side((ry1, ry2), (rx1, rx2), True, 1)
    bottom = scan_side((ry2, ry1), (rx1, rx2), True, -1)
    left = scan_side((rx1, rx2), (ry1, ry2), False, 1)
    right = scan_side((rx2, rx1), (ry1, ry2), False, -1)
    if None in (top, bottom, left, right):
        return None
    frame = (left + FRAME_INSET + cx1, top + FRAME_INSET + cy1,
             right - FRAME_INSET + cx1, bottom - FRAME_INSET + cy1)
    if (frame[2] - frame[0] < FRAME_MIN_SIZE * (x2 - x1)
            or frame[3] - frame[1] < FRAME_MIN_SIZE * (y2 - y1)
            or frame[0] < x1 or frame[1] < y1 or frame[2] > x2 or frame[3] > y2):
        return None
    return frame


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
    image.paste(canvas.crop((0, top, width, top + height)))
    return image


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
) -> int:
    rendered_count = 0
    img_w, img_h = image.size
    for obj in text_objects:
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
                "Chapter {} page {} object {} has malformed region",
                req.chapter_id, req.page_index, oid,
            )
            raise HTTPException(422, f"Vùng chữ không hợp lệ ({oid})")

        x1 = max(0, min(img_w, coords_raw[0]))
        y1 = max(0, min(img_h, coords_raw[1]))
        x2 = max(0, min(img_w, coords_raw[2]))
        y2 = max(0, min(img_h, coords_raw[3]))
        if x2 <= x1 or y2 <= y1:
            logger.warning(
                "Chapter {} page {} object {} invalid coordinates ({},{},{},{}) for image size ({},{})",
                req.chapter_id, req.page_index, oid,
                x1, y1, x2, y2, img_w, img_h,
            )
            raise HTTPException(422, f"Vùng chữ nằm ngoài ảnh ({oid})")
        # The layout uses the whole region even past the slice edge; only drawing is clipped.
        coords = (x1, coords_raw[1], x2, coords_raw[3])

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
        requested_size = style_get(font_sizes_dict, oid)
        stored_size = obj_style.get("fontSize", "auto")
        box_size = _resolve_ocr_style(
            requested_size,
            stored_size,
            obj.get("ocr_font_size"),
            "auto",
        )
        size_is_hint = _is_auto_style(requested_size if requested_size is not None else stored_size) and bool(obj.get("ocr_font_size"))
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

        container = str(obj.get("container") or "").lower()
        # A curved container narrows its lines toward the poles; a rectangular one
        # is clamped to its printed border when one is found. Neither is guessed:
        # an unknown container keeps the region behaviour it always had.
        taper = container in ("bubble", "spiky") or obj_shape == "ellipse"
        style = dict(
            fill=box_color,
            font_size=box_size,
            size_is_hint=size_is_hint,
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
            enlarge=bool(obj.get("enlarge")),
            taper=taper,
        )

        def _attempts_for(box):
            tight = letter_box(obj, box)
            if not tight:
                return [box]
            half = tuple((a + b) // 2 for a, b in zip(tight, box))
            return [tight, half, box] if half != tight else [tight, box]

        # The lettering goes where the source lettering was, then halfway to the region, then the region,
        # taking the first that holds it near the source size.
        ladders = [_attempts_for(coords)]
        if not taper and container in ("box", "screen"):
            frame = detect_printed_frame(image, coords)
            if frame:
                clamped = (max(coords[0], frame[0]), max(coords[1], frame[1]),
                           min(coords[2], frame[2]), min(coords[3], frame[3]))
                if clamped[0] < clamped[2] and clamped[1] < clamped[3] and clamped != coords:
                    # The in-frame ladder runs first; when nothing fits readably inside
                    # the frame, the region ladder below keeps the old behaviour.
                    ladders.insert(0, _attempts_for(clamped))
        try:
            rendered = False
            for attempts in ladders:
                for attempt, box in enumerate(attempts):
                    share = SOURCE_SIZE_SHARE if attempt < len(attempts) - 1 else 0.0
                    try:
                        image = _render_in_region(image, translation.strip(), box, min_source_share=share, **style)
                        rendered = True
                        break
                    except ValueError:
                        if attempt == len(attempts) - 1:
                            break
                if rendered:
                    break
            if not rendered:
                raise ValueError("Translation does not fit in the region")
            rendered_count += 1
        except Exception as e:
            logger.opt(exception=True).error(
                "Chapter {} page {} object {} operation 'render_text_in_box' failed: {}",
                req.chapter_id, req.page_index, oid, e,
            )
            raise HTTPException(500, f"Chèn chữ thất bại (vùng {oid})") from e

    return rendered_count
