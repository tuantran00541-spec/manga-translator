"""API Router for text rendering and font management."""

import copy
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
    """Lookup style value by box index — keys may be int or str after JSON/Pydantic."""
    if not d:
        return default
    if idx in d:
        return d[idx]
    s = str(idx)
    if s in d:
        return d[s]
    return default


@router.get("/fonts")
def get_available_fonts() -> list[dict[str, str]]:
    return list_available_fonts()


@router.post("/render")
def render_page(req: RenderRequest) -> dict:
    validate_chapter_id(req.chapter_id)
    with get_manifest_lock(req.chapter_id):
        manifest = load_manifest_raw(req.chapter_id)
        if req.page_index < 0 or req.page_index >= len(manifest.get("pages", [])):
            raise HTTPException(400, f"Invalid page_index: {req.page_index}")
        page = copy.deepcopy(manifest["pages"][req.page_index])
        boxes_snapshot = copy.deepcopy(page.get("boxes", []))
        clean_snapshot = page.get("clean")
        original_snapshot = page.get("original")
        skipped_snapshot = page.get("skipped", False)
        drafts = copy.deepcopy(manifest.get("drafts", {}))

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

    out_dir = OUTPUT_DIR / req.chapter_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"page_{req.page_index:03d}.png"
    try:
        image.save(out_path)
    except Exception as e:
        logger.error(
            "Chapter %s page %s operation 'render_page' save failed: %s",
            req.chapter_id,
            req.page_index,
            e,
            exc_info=True,
        )
        raise HTTPException(500, "Cannot save rendered image") from e

    state_changed = False
    with get_manifest_lock(req.chapter_id):
        m = load_manifest_raw(req.chapter_id)
        if 0 <= req.page_index < len(m.get("pages", [])):
            cur_page = m["pages"][req.page_index]
            if (
                cur_page.get("boxes", []) != boxes_snapshot
                or cur_page.get("clean") != clean_snapshot
                or cur_page.get("original") != original_snapshot
                or cur_page.get("skipped", False) != skipped_snapshot
            ):
                state_changed = True
            else:
                cur_page["rendered"] = True
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

    if state_changed:
        logger.warning(
            "Chapter %s page %s: page state changed during render, skipping rendered=True",
            req.chapter_id,
            req.page_index,
        )
        return {
            "output": f"/api/image/{req.chapter_id}/{req.page_index}/rendered",
            "warning": "Vùng thoại đã bị sửa trong quá trình chèn chữ.",
        }

    logger.info(
        "Chapter %s page %s: rendered %s box(es) -> %s",
        req.chapter_id,
        req.page_index,
        rendered_count,
        out_path.name,
    )
    return {"output": f"/api/image/{req.chapter_id}/{req.page_index}/rendered"}
