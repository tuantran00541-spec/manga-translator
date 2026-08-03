"""API Router for text rendering and font management."""

from pathlib import Path
from PIL import Image
from fastapi import APIRouter, HTTPException

from app.config import OUTPUT_DIR
from app.logging_config import logger
from app.manifest_utils import load_manifest_raw, save_manifest_raw
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
    manifest = load_manifest_raw(req.chapter_id)
    if req.page_index < 0 or req.page_index >= len(manifest["pages"]):
        raise HTTPException(400, f"Invalid page_index: {req.page_index}")
    page = manifest["pages"][req.page_index]
    logger.info(
        f"Chapter {req.chapter_id} page {req.page_index}: "
        f"rendering {len(req.translations)} translations"
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
    for box_key, translation in req.translations.items():
        try:
            box_idx = int(box_key)
        except (TypeError, ValueError):
            continue
        if box_idx < 0 or box_idx >= len(page["boxes"]):
            continue
        box = page["boxes"][box_idx]
        if box.get("removed"):
            continue
        if not (translation or "").strip():
            continue

        coords = (
            int(box["x1"]),
            int(box["y1"]),
            int(box["x2"]),
            int(box["y2"]),
        )
        box_color = _style_get(colors_dict, box_idx, "auto")
        box_font = _style_get(fonts_dict, box_idx, "default")
        box_size = _style_get(font_sizes_dict, box_idx, "auto")
        bold_val = _style_get(bolds_dict, box_idx, False)
        box_bold = bool(bold_val) if bold_val is not None else False
        stroke_w = _style_get(stroke_w_dict, box_idx, "auto")
        stroke_c = _style_get(stroke_c_dict, box_idx, "auto")
        bg_c = _style_get(bg_colors_dict, box_idx, "transparent")
        r_raw = _style_get(radii_dict, box_idx, 0)
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
            logger.exception(
                "Render failed for chapter %s page %s box %s",
                req.chapter_id,
                req.page_index,
                box_idx,
            )
            raise HTTPException(
                500,
                f"Chèn chữ thất bại (ô {box_idx}): {e}",
            ) from e

    out_dir = OUTPUT_DIR / req.chapter_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"page_{req.page_index:03d}.png"
    try:
        image.save(out_path)
    except Exception as e:
        logger.exception("Failed to save rendered page %s", out_path)
        raise HTTPException(500, f"Cannot save rendered image: {e}") from e

    from app.manifest_utils import get_manifest_lock
    with get_manifest_lock(req.chapter_id):
        m = load_manifest_raw(req.chapter_id)
        if 0 <= req.page_index < len(m.get("pages", [])):
            m["pages"][req.page_index]["rendered"] = True
            save_manifest_raw(req.chapter_id, m)

    logger.info(
        "Chapter %s page %s: rendered %s box(es) -> %s",
        req.chapter_id,
        req.page_index,
        rendered_count,
        out_path.name,
    )
    return {"output": f"/api/image/{req.chapter_id}/{req.page_index}/rendered"}
