import copy
import os
import uuid
from pathlib import Path
from PIL import Image
from fastapi import APIRouter, HTTPException

from app.config import OUTPUT_DIR
from app.logging_config import logger
from app.manifest_utils import get_manifest_lock, load_manifest_raw, save_manifest_raw
from app.render.text_renderer import list_available_fonts, render_text_in_box
from app.schemas import RenderRequest
from app.security import validate_chapter_id

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


@router.post("/render")
def render_page(req: RenderRequest) -> dict:
    validate_chapter_id(req.chapter_id)
    with get_manifest_lock(req.chapter_id):
        manifest = load_manifest_raw(req.chapter_id)
        if req.page_index < 0 or req.page_index >= len(manifest.get("pages", [])):
            raise HTTPException(400, f"Invalid page_index: {req.page_index}")
        page = copy.deepcopy(manifest["pages"][req.page_index])
        boxes_snapshot = copy.deepcopy(page.get("boxes", []))
        text_objects_snapshot = copy.deepcopy(page.get("text_objects", []))
        clean_snapshot = page.get("clean")
        original_snapshot = page.get("original")
        skipped_snapshot = page.get("skipped", False)
        drafts = copy.deepcopy(manifest.get("drafts", {}))
        _page_prefix = f"{req.page_index}_"
        page_drafts_snapshot = {
            k: v for k, v in drafts.items()
            if str(k).startswith(_page_prefix)
        }

    logger.info(
        f"Chapter {req.chapter_id} page {req.page_index}: "
        f"rendering translations (req keys: {len(req.translations)})"
    )

    base_image_path = page.get("clean") or page.get("original")
    if not base_image_path:
        raise HTTPException(400, "Page has no image to render onto")
    base_path = Path(base_image_path)
    if not base_path.is_file():
        raise HTTPException(
            404,
            f"Base image not found: {base_path.name}. "
            "Hãy chạy xử lý trang (process) trước khi chèn chữ.",
        )

    try:
        image = Image.open(base_path).convert("RGB")
    except Exception as e:
        logger.exception("Failed to open base image %s", base_path)
        raise HTTPException(500, f"Cannot open base image: {e}") from e

    colors_dict = req.colors or {}
    fonts_dict = req.fonts or {}
    font_sizes_dict = req.font_sizes or {}
    bolds_dict = req.bolds or {}
    stroke_w_dict = req.stroke_widths or {}
    stroke_c_dict = req.stroke_colors or {}
    bg_colors_dict = req.bg_colors or {}
    radii_dict = req.corner_radii or {}
    horizontal_aligns_dict = req.horizontal_aligns or {}
    vertical_aligns_dict = req.vertical_aligns or {}

    rendered_count = 0
    text_objects = page.get("text_objects") or []
    if text_objects:
        rendered_count = _render_text_objects(
            image, req, text_objects,
            colors_dict, fonts_dict, font_sizes_dict, bolds_dict,
            stroke_w_dict, stroke_c_dict, bg_colors_dict, radii_dict,
            horizontal_aligns_dict, vertical_aligns_dict,
        )
    else:
        rendered_count = _render_boxes_legacy(
            image, req, page, drafts,
            colors_dict, fonts_dict, font_sizes_dict, bolds_dict,
            stroke_w_dict, stroke_c_dict, bg_colors_dict, radii_dict,
        )

    out_dir = OUTPUT_DIR / req.chapter_id
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / f"page_{req.page_index:03d}.png"
    tmp_path = out_dir / f"page_{req.page_index:03d}.rendering.{uuid.uuid4().hex[:12]}.tmp"

    try:
        image.save(tmp_path, format="PNG")
    except Exception as e:
        logger.error(
            "Chapter %s page %s operation 'render_page' save failed: %s",
            req.chapter_id,
            req.page_index,
            e,
            exc_info=True,
        )
        _cleanup_tmp(tmp_path)
        raise HTTPException(500, "Cannot save rendered image") from e

    state_changed = False
    try:
        with get_manifest_lock(req.chapter_id):
            m = load_manifest_raw(req.chapter_id)
            if 0 <= req.page_index < len(m.get("pages", [])):
                cur_page = m["pages"][req.page_index]
                cur_page_drafts = {
                    k: v for k, v in m.get("drafts", {}).items()
                    if str(k).startswith(_page_prefix)
                }
                if (
                    cur_page.get("boxes", []) != boxes_snapshot
                    or cur_page.get("text_objects", []) != text_objects_snapshot
                    or cur_page.get("clean") != clean_snapshot
                    or cur_page.get("original") != original_snapshot
                    or cur_page.get("skipped", False) != skipped_snapshot
                    or cur_page_drafts != page_drafts_snapshot
                ):
                    state_changed = True
            else:
                state_changed = True

            if not state_changed:
                os.replace(tmp_path, final_path)
                cur_page["rendered"] = True
                if text_objects:
                    cur_text_objects = cur_page.setdefault("text_objects", [])
                    for obj in cur_text_objects:
                        oid = obj.get("id")
                        if not oid:
                            continue
                        t = _style_get(req.translations, oid)
                        if t is None:
                            t = obj.get("translation", "")
                        if (t or "").strip():
                            obj["translation"] = t.strip()
                else:
                    m_drafts = m.setdefault("drafts", {})
                    for box_key, translation in req.translations.items():
                        if (translation or "").strip():
                            dk = f"{req.page_index}_{box_key}"
                            d = m_drafts.setdefault(dk, {})
                            d["text"] = translation.strip()
                            if _style_get(colors_dict, box_key) is not None:
                                d["color"] = _style_get(colors_dict, box_key)
                            if _style_get(fonts_dict, box_key) is not None:
                                d["font"] = _style_get(fonts_dict, box_key)
                            if _style_get(font_sizes_dict, box_key) is not None:
                                d["fontSize"] = _style_get(font_sizes_dict, box_key)
                            if _style_get(bolds_dict, box_key) is not None:
                                d["bold"] = bool(_style_get(bolds_dict, box_key))
                            if _style_get(stroke_w_dict, box_key) is not None:
                                d["strokeWidth"] = _style_get(stroke_w_dict, box_key)
                            if _style_get(stroke_c_dict, box_key) is not None:
                                d["strokeColor"] = _style_get(stroke_c_dict, box_key)
                            if _style_get(bg_colors_dict, box_key) is not None:
                                d["bgColor"] = _style_get(bg_colors_dict, box_key)
                            if _style_get(radii_dict, box_key) is not None:
                                d["cornerRadius"] = _style_get(radii_dict, box_key)
                save_manifest_raw(req.chapter_id, m)
    finally:
        _cleanup_tmp(tmp_path)

    if state_changed:
        logger.warning(
            "Chapter %s page %s: page state changed during render, discarding stale output",
            req.chapter_id,
            req.page_index,
        )
        return {
            "output": f"/api/image/{req.chapter_id}/{req.page_index}/rendered",
            "warning": "Vùng thoại đã bị sửa trong quá trình chèn chữ. Vui lòng chèn chữ lại.",
        }

    logger.info(
        "Chapter %s page %s: rendered %s box(es) -> %s",
        req.chapter_id,
        req.page_index,
        rendered_count,
        final_path.name,
    )
    return {"output": f"/api/image/{req.chapter_id}/{req.page_index}/rendered"}
