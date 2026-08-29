from pathlib import Path

from PIL import Image
from fastapi import APIRouter, HTTPException

from app.logging_config import logger
from app.render.text_renderer import list_available_fonts, render_text_in_box
from app.schemas import RenderRequest

router = APIRouter(prefix="/api", tags=["render"])


def _style_get(d: dict, idx: int, default=None):
    if not d:
        return default
    if idx in d:
        return d[idx]
    s = str(idx)
    if s in d:
        return d[s]
    return default


def _cleanup_tmp(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass


@router.get("/fonts")
def get_available_fonts() -> list[dict[str, str]]:
    return list_available_fonts()


def _render_boxes_legacy(
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

        translation = _style_get(req.translations, box_idx)
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
                "Chapter %s page %s box %s invalid coordinates (%s,%s,%s,%s) for image size (%s,%s), skipping",
                req.chapter_id,
                req.page_index,
                box_idx,
                x1, y1, x2, y2,
                img_w, img_h,
            )
            continue
        coords = (x1, y1, x2, y2)

        box_color = _style_get(colors_dict, box_idx)
        if box_color is None:
            box_color = draft_item.get("color", "auto")

        box_font = _style_get(fonts_dict, box_idx)
        if box_font is None:
            box_font = draft_item.get("font", "default")

        box_size = _style_get(font_sizes_dict, box_idx)
        if box_size is None:
            box_size = draft_item.get("fontSize", "auto")

        bold_val = _style_get(bolds_dict, box_idx)
        if bold_val is None:
            bold_val = draft_item.get("bold", False)
        box_bold = bool(bold_val) if bold_val is not None else False

        stroke_w = _style_get(stroke_w_dict, box_idx)
        if stroke_w is None:
            stroke_w = draft_item.get("strokeWidth", "auto")

        stroke_c = _style_get(stroke_c_dict, box_idx)
        if stroke_c is None:
            stroke_c = draft_item.get("strokeColor", "auto")

        bg_c = _style_get(bg_colors_dict, box_idx)
        if bg_c is None:
            bg_c = draft_item.get("bgColor", "transparent")

        r_raw = _style_get(radii_dict, box_idx)
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
            logger.error(
                "Chapter %s page %s box %s operation 'render_text_in_box' failed: %s",
                req.chapter_id,
                req.page_index,
                box_idx,
                e,
                exc_info=True,
            )
            raise HTTPException(
                500,
                f"Chèn chữ thất bại (ô {box_idx})",
            ) from e

    return rendered_count


def _render_text_objects(
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
        translation = _style_get(req.translations, oid)
        if translation is None or translation == "":
            translation = obj.get("translation", "")
        if not (translation or "").strip():
            continue

        region = obj.get("region") or {}
        try:
            coords_raw = (
                int(region["x1"]), int(region["y1"]),
                int(region["x2"]), int(region["y2"]),
            )
        except (KeyError, TypeError, ValueError):
            logger.warning(
                "Chapter %s page %s object %s has malformed region, skipping",
                req.chapter_id, req.page_index, oid,
            )
            continue

        x1 = max(0, min(img_w, coords_raw[0]))
        y1 = max(0, min(img_h, coords_raw[1]))
        x2 = max(0, min(img_w, coords_raw[2]))
        y2 = max(0, min(img_h, coords_raw[3]))
        if x2 <= x1 or y2 <= y1:
            logger.warning(
                "Chapter %s page %s object %s invalid coordinates (%s,%s,%s,%s) for image size (%s,%s), skipping",
                req.chapter_id, req.page_index, oid,
                x1, y1, x2, y2, img_w, img_h,
            )
            continue
        coords = (x1, y1, x2, y2)

        obj_style = obj.get("style") or {}
        box_color = _style_get(colors_dict, oid)
        if box_color is None:
            box_color = obj_style.get("color", "auto")
        box_font = _style_get(fonts_dict, oid)
        if box_font is None:
            box_font = obj_style.get("font", "default")
        box_size = _style_get(font_sizes_dict, oid)
        if box_size is None:
            box_size = obj_style.get("fontSize", "auto")
        bold_val = _style_get(bolds_dict, oid)
        if bold_val is None:
            bold_val = obj_style.get("bold", False)
        box_bold = bool(bold_val) if bold_val is not None else False
        stroke_w = _style_get(stroke_w_dict, oid)
        if stroke_w is None:
            stroke_w = obj_style.get("strokeWidth", "auto")
        stroke_c = _style_get(stroke_c_dict, oid)
        if stroke_c is None:
            stroke_c = obj_style.get("strokeColor", "auto")
        bg_c = _style_get(bg_colors_dict, oid)
        if bg_c is None:
            bg_c = obj_style.get("bgColor", "transparent")
        r_raw = _style_get(radii_dict, oid)
        if r_raw is None:
            r_raw = obj_style.get("cornerRadius", 0)
        try:
            r_val = int(r_raw or 0)
        except (TypeError, ValueError):
            r_val = 0

        h_align = _style_get(horizontal_aligns_dict, oid)
        if h_align is None:
            h_align = obj_style.get("horizontalAlign", "center")
        v_align = _style_get(vertical_aligns_dict, oid)
        if v_align is None:
            v_align = obj_style.get("verticalAlign", "middle")

        obj_shape = str(obj.get("shape") or "rectangle").lower()
        if obj_shape not in ("rectangle", "ellipse"):
            obj_shape = "rectangle"

        try:
            image = render_text_in_box(
                image,
                translation.strip(),
                coords,
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
            )
            rendered_count += 1
        except Exception as e:
            logger.error(
                "Chapter %s page %s object %s operation 'render_text_in_box' failed: %s",
                req.chapter_id, req.page_index, oid, e, exc_info=True,
            )
            raise HTTPException(500, f"Chèn chữ thất bại (vùng {oid})") from e

    return rendered_count
