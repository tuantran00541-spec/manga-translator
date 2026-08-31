from __future__ import annotations

import copy
import os
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException
from PIL import Image

from app.config import OUTPUT_DIR
from app.logging_config import logger
from app.manifest_utils import bump_page_revision, get_manifest_lock, load_manifest_raw, save_manifest_raw
from app.render.identity import render_input_signature, stamp_render_artifact
from app.render.page_renderer import cleanup_tmp, render_boxes_legacy, render_text_objects, style_get
from app.schemas import RenderRequest
from app.security import validate_chapter_id

router = APIRouter(prefix="/api", tags=["render"])


def _request_style_maps(req: RenderRequest) -> dict[str, dict]:
    return {
        "colors": req.colors or {},
        "fonts": req.fonts or {},
        "font_sizes": req.font_sizes or {},
        "bolds": req.bolds or {},
        "stroke_widths": req.stroke_widths or {},
        "stroke_colors": req.stroke_colors or {},
        "bg_colors": req.bg_colors or {},
        "corner_radii": req.corner_radii or {},
        "horizontal_aligns": req.horizontal_aligns or {},
        "vertical_aligns": req.vertical_aligns or {},
    }


def _render_snapshot(image: Image.Image, req: RenderRequest, page: dict, drafts: dict, styles: dict[str, dict]) -> int:
    text_objects = page.get("text_objects") or []
    if text_objects:
        active_text_objects = [
            obj
            for obj in text_objects
            if isinstance(obj, dict) and not obj.get("source_missing")
        ]
        if not active_text_objects:
            return 0
        return render_text_objects(
            image,
            req,
            active_text_objects,
            styles["colors"],
            styles["fonts"],
            styles["font_sizes"],
            styles["bolds"],
            styles["stroke_widths"],
            styles["stroke_colors"],
            styles["bg_colors"],
            styles["corner_radii"],
            styles["horizontal_aligns"],
            styles["vertical_aligns"],
        )
    return render_boxes_legacy(
        image,
        req,
        page,
        drafts,
        styles["colors"],
        styles["fonts"],
        styles["font_sizes"],
        styles["bolds"],
        styles["stroke_widths"],
        styles["stroke_colors"],
        styles["bg_colors"],
        styles["corner_radii"],
    )


def _set_style_value(style: dict, field: str, value, *, stringify: bool = False) -> None:
    if value is None:
        return
    style[field] = str(value) if stringify else value


def _persist_text_object_state(page: dict, req: RenderRequest, styles: dict[str, dict]) -> None:
    for obj in page.get("text_objects") or []:
        if not isinstance(obj, dict) or obj.get("source_missing"):
            continue
        oid = obj.get("id")
        if not oid:
            continue
        translation = style_get(req.translations, oid)
        if translation is not None and str(translation).strip():
            obj["translation"] = str(translation).strip()

        style = obj.setdefault("style", {})
        _set_style_value(style, "color", style_get(styles["colors"], oid))
        _set_style_value(style, "font", style_get(styles["fonts"], oid))
        _set_style_value(style, "fontSize", style_get(styles["font_sizes"], oid), stringify=True)
        _set_style_value(style, "bold", style_get(styles["bolds"], oid))
        _set_style_value(style, "strokeWidth", style_get(styles["stroke_widths"], oid), stringify=True)
        _set_style_value(style, "strokeColor", style_get(styles["stroke_colors"], oid))
        _set_style_value(style, "bgColor", style_get(styles["bg_colors"], oid))
        _set_style_value(style, "cornerRadius", style_get(styles["corner_radii"], oid), stringify=True)
        _set_style_value(style, "horizontalAlign", style_get(styles["horizontal_aligns"], oid))
        _set_style_value(style, "verticalAlign", style_get(styles["vertical_aligns"], oid))


def _persist_legacy_drafts(manifest: dict, req: RenderRequest, styles: dict[str, dict]) -> None:
    drafts = manifest.setdefault("drafts", {})
    for box_key, translation in req.translations.items():
        if not (translation or "").strip():
            continue
        draft = drafts.setdefault(f"{req.page_index}_{box_key}", {})
        draft["text"] = translation.strip()
        mappings = (
            ("color", styles["colors"]),
            ("font", styles["fonts"]),
            ("fontSize", styles["font_sizes"]),
            ("bold", styles["bolds"]),
            ("strokeWidth", styles["stroke_widths"]),
            ("strokeColor", styles["stroke_colors"]),
            ("bgColor", styles["bg_colors"]),
            ("cornerRadius", styles["corner_radii"]),
        )
        for field, values in mappings:
            value = style_get(values, box_key)
            if value is not None:
                draft[field] = bool(value) if field == "bold" else value


def _load_render_snapshot(req: RenderRequest) -> tuple[dict, dict, str]:
    with get_manifest_lock(req.chapter_id):
        manifest = load_manifest_raw(req.chapter_id)
        pages = manifest.get("pages", [])
        if req.page_index < 0 or req.page_index >= len(pages):
            raise HTTPException(400, f"Invalid page_index: {req.page_index}")
        page = copy.deepcopy(pages[req.page_index])
        drafts = copy.deepcopy(manifest.get("drafts", {}))
        try:
            signature = render_input_signature(manifest, req.page_index)
        except FileNotFoundError as exc:
            raise HTTPException(404, "Base image not found. Hãy chạy xử lý trang trước khi kết xuất.") from exc
    return page, drafts, signature


def _commit_render(
    req: RenderRequest,
    styles: dict[str, dict],
    snapshot_signature: str,
    tmp_path: Path,
    final_path: Path,
) -> tuple[bool, int | None]:
    with get_manifest_lock(req.chapter_id):
        manifest = load_manifest_raw(req.chapter_id)
        try:
            current_signature = render_input_signature(manifest, req.page_index)
        except (IndexError, OSError):
            return False, None
        if current_signature != snapshot_signature:
            return False, None

        page = manifest["pages"][req.page_index]
        if page.get("text_objects"):
            _persist_text_object_state(page, req, styles)
        else:
            _persist_legacy_drafts(manifest, req, styles)

        final_signature = render_input_signature(manifest, req.page_index)
        os.replace(tmp_path, final_path)
        render_revision = bump_page_revision(page, "render_revision")
        stamp_render_artifact(
            page,
            input_signature=final_signature,
            output_path=final_path,
        )
        save_manifest_raw(req.chapter_id, manifest)
        return True, render_revision


@router.post(
    "/render",
    responses={
        400: {"description": "Invalid render request"},
        404: {"description": "Base image missing"},
        409: {"description": "Page changed while render was running"},
        500: {"description": "Render failed"},
    },
)
def render_page(req: RenderRequest) -> dict:
    validate_chapter_id(req.chapter_id)
    page, drafts, snapshot_signature = _load_render_snapshot(req)
    base_value = page.get("clean") or page.get("original")
    if not base_value:
        raise HTTPException(400, "Page has no image to render onto")
    base_path = Path(str(base_value))

    try:
        image = Image.open(base_path).convert("RGB")
    except FileNotFoundError as exc:
        raise HTTPException(404, "Base image not found") from exc
    except Exception as exc:
        logger.opt(exception=True).error(
            "Chapter {} page {} cannot open render base: {}",
            req.chapter_id,
            req.page_index,
            exc,
        )
        raise HTTPException(500, "Cannot open base image") from exc

    styles = _request_style_maps(req)
    rendered_count = _render_snapshot(image, req, page, drafts, styles)

    out_dir = OUTPUT_DIR / req.chapter_id
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / f"page_{req.page_index:03d}.png"
    tmp_path = out_dir / f"page_{req.page_index:03d}.rendering.{uuid.uuid4().hex[:12]}.tmp"
    try:
        image.save(tmp_path, format="PNG")
        committed, render_revision = _commit_render(
            req,
            styles,
            snapshot_signature,
            tmp_path,
            final_path,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.opt(exception=True).error(
            "Chapter {} page {} render commit failed: {}",
            req.chapter_id,
            req.page_index,
            exc,
        )
        raise HTTPException(500, "Cannot save rendered image") from exc
    finally:
        cleanup_tmp(tmp_path)

    if not committed:
        logger.warning(
            "Chapter {} page {} changed during render; stale output discarded",
            req.chapter_id,
            req.page_index,
        )
        raise HTTPException(
            409,
            "Dữ liệu trang đã thay đổi trong lúc kết xuất. Vui lòng kết xuất lại.",
        )

    logger.info(
        "Chapter {} page {} rendered {} object(s) at revision {}",
        req.chapter_id,
        req.page_index,
        rendered_count,
        render_revision,
    )
    return {
        "output": f"/api/image/{req.chapter_id}/{req.page_index}/rendered",
        "committed": True,
        "render_revision": render_revision,
    }
