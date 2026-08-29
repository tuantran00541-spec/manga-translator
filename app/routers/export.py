from __future__ import annotations

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
from app.routers.render_commit import render_page
from app.schemas import RenderRequest
from app.security import validate_chapter_id, validate_image_size
from app.text_objects import ensure_page_text_objects


router = APIRouter(prefix="/api", tags=["export"])


def _safe_int(value, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(fallback)


def _file_revision(path: Path) -> tuple[int, int, int]:
    st = path.stat()
    return int(st.st_size), int(st.st_mtime_ns), int(st.st_ctime_ns)


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
        corner_radii[oid] = _safe_int(style.get("cornerRadius"), 0)
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


def _normalized_core_range(height: int, core: tuple[int, int] | None) -> tuple[int, int]:
    if core is None:
        return 0, height
    y1, y2 = core
    y1 = max(0, min(int(y1), height))
    y2 = max(y1, min(int(y2), height))
    return y1, y2


def _stitch_png_to_file(
    paths: list[Path],
    output_path: Path,
    core_ranges: list[tuple[int, int] | None] | None = None,
) -> None:
    """Stitch slices with only one decoded source image resident at a time."""
    if not paths:
        raise ValueError("No images to stitch")
    if core_ranges is not None and len(core_ranges) != len(paths):
        raise ValueError("Core range count does not match image count")

    metadata: list[tuple[int, int, int]] = []
    for index, path in enumerate(paths):
        core = core_ranges[index] if core_ranges is not None else None
        with Image.open(path) as image:
            width, height = image.size
        y1, y2 = _normalized_core_range(height, core)
        metadata.append((width, y1, y2))

    widths = {width for width, _, _ in metadata}
    if len(widths) != 1:
        raise ValueError("Slice widths do not match")

    if len(paths) == 1:
        width, y1, y2 = metadata[0]
        with Image.open(paths[0]) as source:
            image = source.convert("RGB")
            try:
                if y1 == 0 and y2 == image.height:
                    image.save(output_path, format="PNG")
                else:
                    cropped = image.crop((0, y1, width, y2))
                    try:
                        cropped.save(output_path, format="PNG")
                    finally:
                        cropped.close()
            finally:
                image.close()
        return

    width = metadata[0][0]
    total_height = sum(y2 - y1 for _, y1, y2 in metadata)
    canvas = Image.new("RGB", (width, total_height), "white")
    try:
        target_y = 0
        for path, (_, y1, y2) in zip(paths, metadata):
            with Image.open(path) as source:
                image = source.convert("RGB")
                try:
                    if y1 == 0 and y2 == image.height:
                        canvas.paste(image, (0, target_y))
                    else:
                        cropped = image.crop((0, y1, width, y2))
                        try:
                            canvas.paste(cropped, (0, target_y))
                        finally:
                            cropped.close()
                finally:
                    image.close()
            target_y += y2 - y1
        canvas.save(output_path, format="PNG")
    finally:
        canvas.close()


def _export_path_for_page(chapter_id: str, page_index: int, page: dict, manifest: dict) -> Path:
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
    return path


def _core_range(page: dict) -> tuple[int, int] | None:
    core = page.get("stitch_core") if isinstance(page.get("stitch_core"), dict) else None
    if core is None:
        return None
    try:
        return int(core.get("core_y1", 0)), int(core.get("core_y2", 0))
    except (TypeError, ValueError):
        return None


def _snapshot_export_inputs(chapter_id: str) -> list[dict]:
    """Capture canonical export inputs while holding the manifest lock briefly."""
    with get_manifest_lock(chapter_id):
        manifest = load_manifest_raw(chapter_id)
        snapshot: list[dict] = []
        for page_index, page in enumerate(manifest.get("pages", [])):
            path = _export_path_for_page(chapter_id, page_index, page, manifest)
            try:
                revision = _file_revision(path)
            except OSError as exc:
                raise HTTPException(409, f"Page {page_index + 1} changed before export started") from exc
            snapshot.append(
                {
                    "page_index": page_index,
                    "source_page": _safe_int(page.get("source_page"), page_index),
                    "slice_index": _safe_int(page.get("slice_index"), 0),
                    "path": path,
                    "revision": revision,
                    "core_range": _core_range(page),
                }
            )
        return snapshot


def _export_snapshot_is_current(chapter_id: str, snapshot: list[dict]) -> bool:
    """Revalidate page/file and stitch metadata after expensive encoding."""
    manifest = load_manifest_raw(chapter_id)
    pages = manifest.get("pages", [])
    if len(snapshot) != len(pages):
        return False
    for item in snapshot:
        page_index = int(item["page_index"])
        if page_index < 0 or page_index >= len(pages):
            return False
        page = pages[page_index]
        if (
            _safe_int(page.get("source_page"), page_index) != int(item["source_page"])
            or _safe_int(page.get("slice_index"), 0) != int(item["slice_index"])
            or _core_range(page) != item["core_range"]
        ):
            return False
        try:
            path = _export_path_for_page(chapter_id, page_index, page, manifest)
            revision = _file_revision(path)
        except (HTTPException, OSError):
            return False
        if path != item["path"] or revision != item["revision"]:
            return False
    return True


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

    snapshot = _snapshot_export_inputs(chapter_id)
    groups: dict[int, list[dict]] = {}
    for item in snapshot:
        path = item["path"]
        validate_image_size(path)
        groups.setdefault(int(item["source_page"]), []).append(item)

    try:
        with zipfile.ZipFile(tmp_archive, "w", compression=zipfile.ZIP_STORED) as archive:
            for output_index, source_page in enumerate(sorted(groups), start=1):
                items = sorted(
                    groups[source_page],
                    key=lambda item: (int(item["slice_index"]), int(item["page_index"])),
                )
                page_tmp = out_dir / (
                    f"page_{output_index:03d}.export.{uuid.uuid4().hex[:12]}.png"
                )
                try:
                    _stitch_png_to_file(
                        [item["path"] for item in items],
                        page_tmp,
                        [item["core_range"] for item in items],
                    )
                    archive.write(page_tmp, arcname=f"page_{output_index:03d}.png")
                finally:
                    if page_tmp.exists():
                        try:
                            page_tmp.unlink()
                        except OSError:
                            pass

        with get_manifest_lock(chapter_id):
            if not _export_snapshot_is_current(chapter_id, snapshot):
                raise HTTPException(
                    409,
                    "Chapter changed while export was running. Export again to include the latest edits.",
                )
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
