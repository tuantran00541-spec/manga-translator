from __future__ import annotations

import io
import os
import uuid
import zipfile
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from PIL import Image

from app.config import OUTPUT_DIR
from app.manifest_utils import get_manifest_lock, load_manifest_raw, save_manifest_raw, urlify_manifest
from app.routers.image import _current_rendered_path, _fallback_page_path
from app.routers.render45 import render_page
from app.schemas import RenderRequest
from app.security import validate_chapter_id, validate_image_size
from app.text_objects import ensure_page_text_objects


router = APIRouter(prefix="/api", tags=["export"])


def _render_request_from_page(chapter_id: str, page_index: int, page: dict) -> RenderRequest:
    translations: dict[str, str] = {}
    colors: dict[str, str] = {}
    fonts: dict[str, str] = {}
    font_sizes: dict[str, int | str] = {}
    bolds: dict[str, bool] = {}
    stroke_widths: dict[str, int | str] = {}
    stroke_colors: dict[str, str] = {}
    bg_colors: dict[str, str] = {}
    corner_radii: dict[str, int] = {}
    horizontal_aligns: dict[str, str] = {}
    vertical_aligns: dict[str, str] = {}

    for obj in page.get("text_objects") or []:
        if not isinstance(obj, dict) or not obj.get("id"):
            continue
        oid = str(obj["id"])
        translations[oid] = str(obj.get("translation") or "")
        style = obj.get("style") or {}
        colors[oid] = str(style.get("color") or "auto")
        fonts[oid] = str(style.get("font") or "default")
        font_sizes[oid] = style.get("fontSize", "auto")
        bolds[oid] = bool(style.get("bold", False))
        stroke_widths[oid] = style.get("strokeWidth", "auto")
        stroke_colors[oid] = str(style.get("strokeColor") or "auto")
        bg_colors[oid] = str(style.get("bgColor") or "transparent")
        try:
            corner_radii[oid] = int(style.get("cornerRadius") or 0)
        except (TypeError, ValueError):
            corner_radii[oid] = 0
        h_align = str(style.get("horizontalAlign") or "center")
        v_align = str(style.get("verticalAlign") or "middle")
        horizontal_aligns[oid] = h_align if h_align in {"left", "center", "right"} else "center"
        vertical_aligns[oid] = v_align if v_align in {"top", "middle", "bottom"} else "middle"

    return RenderRequest(
        chapter_id=chapter_id,
        page_index=page_index,
        translations=translations,
        colors=colors,
        fonts=fonts,
        font_sizes=font_sizes,
        bolds=bolds,
        stroke_widths=stroke_widths,
        stroke_colors=stroke_colors,
        bg_colors=bg_colors,
        corner_radii=corner_radii,
        horizontal_aligns=horizontal_aligns,
        vertical_aligns=vertical_aligns,
    )


def _stitch_pngs(paths: list[Path]) -> bytes:
    if not paths:
        raise ValueError("No images to stitch")
    images = [Image.open(path).convert("RGB") for path in paths]
    try:
        widths = {image.width for image in images}
        if len(widths) != 1:
            raise ValueError("Slice widths do not match")
        if len(images) == 1:
            buffer = io.BytesIO()
            images[0].save(buffer, format="PNG")
            return buffer.getvalue()
        width = images[0].width
        total_height = sum(image.height for image in images)
        canvas = Image.new("RGB", (width, total_height), "white")
        y = 0
        for image in images:
            canvas.paste(image, (0, y))
            y += image.height
        buffer = io.BytesIO()
        canvas.save(buffer, format="PNG")
        return buffer.getvalue()
    finally:
        for image in images:
            image.close()


@router.post("/render/chapter")
def render_chapter(chapter_id: str) -> dict:
    validate_chapter_id(chapter_id)
    with get_manifest_lock(chapter_id):
        manifest = load_manifest_raw(chapter_id)
        changed = False
        for page in manifest.get("pages", []):
            if page.get("skipped"):
                continue
            _, page_changed = ensure_page_text_objects(page)
            changed = changed or page_changed
        if changed:
            save_manifest_raw(chapter_id, manifest)

    rendered = 0
    skipped = 0
    for page_index, page in enumerate(manifest.get("pages", [])):
        if page.get("skipped"):
            skipped += 1
            continue
        request = _render_request_from_page(chapter_id, page_index, page)
        render_page(request)
        rendered += 1

    latest = load_manifest_raw(chapter_id)
    result = urlify_manifest(latest)
    result["chapter_render"] = {
        "rendered": rendered,
        "skipped": skipped,
        "total": len(latest.get("pages", [])),
        "download_url": f"/api/export/{chapter_id}.zip",
    }
    return result


@router.get(
    "/export/{chapter_id}.zip",
    responses={409: {"description": "At least one rendered page is stale or unavailable"}},
)
def export_chapter(chapter_id: str):
    validate_chapter_id(chapter_id)
    out_dir = OUTPUT_DIR / chapter_id
    out_dir.mkdir(parents=True, exist_ok=True)
    final_archive = out_dir / f"chapter_{chapter_id}.zip"
    tmp_archive = out_dir / f"chapter_{chapter_id}.export.{uuid.uuid4().hex[:12]}.tmp"

    with get_manifest_lock(chapter_id):
        manifest = load_manifest_raw(chapter_id)
        groups: dict[int, list[tuple[int, int, Path]]] = {}
        for page_index, page in enumerate(manifest.get("pages", [])):
            if page.get("skipped"):
                path = _fallback_page_path(chapter_id, page)
            else:
                path = _current_rendered_path(chapter_id, page_index, manifest)
                if path is None:
                    raise HTTPException(
                        409,
                        f"Page {page_index + 1} is not currently rendered. Render the chapter again before export.",
                    )
            if path is None or not path.is_file():
                raise HTTPException(404, f"Page {page_index + 1} image is unavailable")
            validate_image_size(path)
            source_page = int(page.get("source_page", page_index))
            slice_index = int(page.get("slice_index", 0))
            groups.setdefault(source_page, []).append((slice_index, page_index, path))

        try:
            with zipfile.ZipFile(tmp_archive, "w", compression=zipfile.ZIP_STORED) as archive:
                for output_index, source_page in enumerate(sorted(groups), start=1):
                    items = sorted(groups[source_page], key=lambda item: (item[0], item[1]))
                    payload = _stitch_pngs([item[2] for item in items])
                    archive.writestr(f"page_{output_index:03d}.png", payload)
            os.replace(tmp_archive, final_archive)
        finally:
            if tmp_archive.exists():
                try:
                    tmp_archive.unlink()
                except OSError:
                    pass

    return FileResponse(
        final_archive,
        filename=f"manga-translator-{chapter_id}.zip",
        media_type="application/zip",
    )
