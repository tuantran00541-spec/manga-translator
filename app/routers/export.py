from __future__ import annotations

import copy
import uuid
import zipfile
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from PIL import Image

from app.config import OUTPUT_DIR
from app.editorial_gate import editorial_preflight
from app.export_pages import FORMATS, ExportOptions, finish_page
from app.logging_config import logger
from app.manifest_utils import atomic_replace, get_manifest_lock, load_manifest_raw, save_manifest_raw, urlify_manifest
from app.routers.image import _current_rendered_path, _fallback_page_path
from app.routers.render_commit import render_page
from app.schemas import RenderRequest
from app.security import validate_chapter_id, validate_image_size
from app.text_objects import ensure_page_text_objects
from app.region_policy import text_object_in_preserve_region


router = APIRouter(prefix="/api", tags=["export"])


BLOCKER_LABELS = {
    "untranslated_story_object": "chưa dịch",
    "story_object_missing_ocr": "chưa có chữ gốc",
    "unaccounted_story_candidate": "chữ chưa có vùng",
}


def _editorial_gate_or_409(
    manifest: dict,
    *,
    require_final_approval: bool = False,
    force: bool = False,
) -> dict:
    preflight = editorial_preflight(
        manifest,
        require_final_approval=require_final_approval,
    )
    if preflight["ok"] or force:
        return preflight
    counts: dict[str, int] = {}
    for blocker in preflight["blockers"]:
        label = BLOCKER_LABELS.get(blocker["kind"], "cần xem lại")
        counts[label] = counts.get(label, 0) + 1
    parts = ", ".join(f"{n} vùng {label}" for label, n in counts.items())
    raise HTTPException(
        409,
        {
            "code": "editorial_preflight",
            "blocker_count": preflight["blocker_count"],
            "counts": counts,
            "message": f"Chương còn {parts}.",
        },
    )


def _preflight_copy(
    manifest: dict,
    *,
    require_final_approval: bool = True,
) -> dict:
    working = copy.deepcopy(manifest)
    for page in working.get("pages", []):
        if isinstance(page, dict) and not page.get("skipped"):
            ensure_page_text_objects(page)
    return editorial_preflight(
        working,
        require_final_approval=require_final_approval,
    )


def _safe_int(value, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(fallback)


def _export_source_page(page: dict, page_index: int) -> int:
    value = page.get("source_page")
    if isinstance(value, bool):
        value = None
    try:
        return int(value) if value is not None else -(page_index + 1)
    except (TypeError, ValueError):
        return -(page_index + 1)


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
        if (
            not isinstance(obj, dict)
            or not obj.get("id")
            or obj.get("source_missing")
            or text_object_in_preserve_region(page, obj)
        ):
            continue
        oid = str(obj["id"])
        translations[oid] = str(obj.get("translation") or "")
        style = obj.get("style") or {}
        colors[oid] = str(style.get("color") or "auto")
        mode = obj.get("font_selection_mode")
        font = style.get("font")
        if mode == "auto":
            fonts[oid] = "auto"
        elif mode == "user":
            fonts[oid] = str(font or "default")
        elif not obj.get("font_ai_id") and font and str(font) != "default":
            fonts[oid] = str(font)
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


def _source_core_metadata(page: dict) -> tuple[tuple[int, int] | None, int | None]:
    core = page.get("stitch_core") if isinstance(page.get("stitch_core"), dict) else None
    if core is None:
        return None, None
    try:
        source_range = (
            int(core["core_source_y1"]),
            int(core["core_source_y2"]),
        )
        source_height = int(core["source_height"])
    except (KeyError, TypeError, ValueError):
        return None, None
    return source_range, source_height


def _validate_stitch_group(source_page: int, items: list[dict]) -> None:
    if not items:
        raise HTTPException(409, f"Source page {source_page + 1} has no export slices")

    slice_indices = [int(item["slice_index"]) for item in items]
    expected_indices = list(range(len(items)))
    if slice_indices != expected_indices:
        raise HTTPException(
            409,
            f"Source page {source_page + 1} has invalid slice ordering/coverage: "
            f"expected {expected_indices}, got {slice_indices}.",
        )

    source_metadata = [item.get("source_core_range") for item in items]
    source_heights = [item.get("source_height") for item in items]

    if len(items) == 1 and source_metadata[0] is None and source_heights[0] is None:
        if items[0].get("stitch_malformed"):
            raise HTTPException(
                409,
                f"Source page {source_page + 1} has malformed stitch metadata (slice {int(items[0]['page_index']) + 1}).",
            )
        return
    if any(value is None for value in source_metadata) or any(value is None for value in source_heights):
        bad = [int(item["page_index"]) + 1 for item, meta, height in zip(items, source_metadata, source_heights)
               if meta is None or height is None]
        raise HTTPException(
            409,
            f"Source page {source_page + 1} is missing stitch ownership metadata (slices {bad}).",
        )

    heights = {int(value) for value in source_heights}
    if len(heights) != 1:
        raise HTTPException(
            409,
            f"Source page {source_page + 1} has inconsistent source heights across slices "
            f"{[int(item['page_index']) + 1 for item in items]}.",
        )
    source_height = heights.pop()
    if source_height <= 0:
        raise HTTPException(409, f"Source page {source_page + 1} has invalid source height "
                                 f"(slices {[int(item['page_index']) + 1 for item in items]})")

    expected_source_y = 0
    for item in items:
        source_y1, source_y2 = item["source_core_range"]
        if source_y1 != expected_source_y or source_y2 <= source_y1:
            relation = "gap" if source_y1 > expected_source_y else "overlap"
            raise HTTPException(
                409,
                f"Source page {source_page + 1} stitch ownership has a {relation} at "
                f"y={expected_source_y}: next core is [{source_y1}, {source_y2}) "
                f"(slice {int(item['page_index']) + 1}).",
            )

        physical_core = item.get("core_range")
        if physical_core is None:
            raise HTTPException(
                409,
                f"Source page {source_page + 1} is missing physical core metadata "
                f"(slice {int(item['page_index']) + 1}).",
            )
        core_y1, core_y2 = physical_core
        if core_y2 <= core_y1 or (core_y2 - core_y1) != (source_y2 - source_y1):
            raise HTTPException(
                409,
                f"Source page {source_page + 1} has inconsistent physical/source core lengths "
                f"(slice {int(item['page_index']) + 1}).",
            )
        expected_source_y = source_y2

    if expected_source_y != source_height:
        raise HTTPException(
            409,
            f"Source page {source_page + 1} stitch ownership ends at y={expected_source_y}, "
            f"expected source height {source_height} "
            f"(slices {[int(item['page_index']) + 1 for item in items]}).",
        )


def _snapshot_export_inputs(chapter_id: str, force: bool = False) -> list[dict]:
    with get_manifest_lock(chapter_id):
        manifest = load_manifest_raw(chapter_id)
        for page in manifest.get("pages", []):
            if not isinstance(page, dict) or page.get("skipped"):
                continue
            ensure_page_text_objects(page)
        _editorial_gate_or_409(manifest, require_final_approval=True, force=force)
        snapshot: list[dict] = []
        for page_index, page in enumerate(manifest.get("pages", [])):
            path = _export_path_for_page(chapter_id, page_index, page, manifest)
            try:
                revision = _file_revision(path)
            except OSError as exc:
                raise HTTPException(409, f"Page {page_index + 1} changed before export started") from exc
            source_core_range, source_height = _source_core_metadata(page)
            source_page = _export_source_page(page, page_index)
            if source_page < 0:
                logger.warning("Chapter {} page {} has no usable source_page; exporting it on its own",
                               chapter_id, page_index + 1)
            snapshot.append(
                {
                    "page_index": page_index,
                    "source_page": source_page,
                    "slice_index": _safe_int(page.get("slice_index"), 0),
                    "path": path,
                    "revision": revision,
                    "core_range": _core_range(page),
                    "source_core_range": source_core_range,
                    "source_height": source_height,
                    "stitch_malformed": ("stitch_core" in page
                                         and source_core_range is None and source_height is None),
                }
            )
        return snapshot


def _export_snapshot_is_current(chapter_id: str, snapshot: list[dict]) -> bool:
    manifest = load_manifest_raw(chapter_id)
    pages = manifest.get("pages", [])
    if len(snapshot) != len(pages):
        return False
    for item in snapshot:
        page_index = int(item["page_index"])
        if page_index < 0 or page_index >= len(pages):
            return False
        page = pages[page_index]
        source_core_range, source_height = _source_core_metadata(page)
        if (
            _export_source_page(page, page_index) != int(item["source_page"])
            or _safe_int(page.get("slice_index"), 0) != int(item["slice_index"])
            or _core_range(page) != item["core_range"]
            or source_core_range != item["source_core_range"]
            or source_height != item["source_height"]
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


@router.get("/export/{chapter_id}/preflight")
def export_preflight(chapter_id: str) -> dict:
    validate_chapter_id(chapter_id)
    with get_manifest_lock(chapter_id):
        manifest = load_manifest_raw(chapter_id)
        return _preflight_copy(manifest)


def _restore_skipped_and_rerender(
    chapter_id: str, page_index: int, skipped: list[str]
) -> tuple[int, list[str], str | None]:
    from app.dependencies import pipeline

    total_restored = 0
    pending = [str(oid) for oid in skipped]
    for _ in range(3):
        with get_manifest_lock(chapter_id):
            page = load_manifest_raw(chapter_id)["pages"][page_index]
            matched = [
                item for item in page.get("text_objects") or []
                if isinstance(item, dict)
                and str(item.get("id")) in pending
                and isinstance(item.get("region"), dict)
            ]
            regions = [dict(item["region"]) for item in matched]
            matched_ids = {str(item["id"]) for item in matched}
        if not regions:
            return total_restored, [], None
        try:
            pipeline.preserve_and_reinpaint(chapter_id, page_index, regions)
        except (ValueError, RuntimeError, OSError) as exc:
            return total_restored, pending, str(exc)[:150]
        total_restored += len(regions)
        with get_manifest_lock(chapter_id):
            from app.text_objects import clear_page_object_translations

            manifest = load_manifest_raw(chapter_id)
            page = manifest["pages"][page_index]
            if clear_page_object_translations(page, matched_ids):
                save_manifest_raw(chapter_id, manifest)
            request = _render_request_from_page(chapter_id, page_index, page)
        rerender = render_page(request)
        pending = [str(oid) for oid in (rerender.get("skipped_objects") or [])]
        if not pending:
            return total_restored, [], None
    return total_restored, pending, None


@router.post("/render/chapter")
def render_chapter(chapter_id: str, force: bool = False) -> dict:
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
        preflight = _editorial_gate_or_409(manifest, force=force)
        total = len(manifest.get("pages", []))

    rendered = 0
    reused = 0
    skipped = 0
    restored_regions = 0
    restore_failures: list[dict] = []
    review_list: list[dict] = []
    for page_index in range(total):
        with get_manifest_lock(chapter_id):
            latest_page_manifest = load_manifest_raw(chapter_id)
            pages = latest_page_manifest.get("pages", [])
            if page_index >= len(pages):
                raise HTTPException(409, "Chapter page list changed while rendering")
            page = pages[page_index]
            if page.get("skipped"):
                skipped += 1
                continue
            if _current_rendered_path(
                chapter_id,
                page_index,
                latest_page_manifest,
            ) is not None:
                reused += 1
                continue
            request = _render_request_from_page(chapter_id, page_index, page)
        result = render_page(request)
        rendered += 1
        skipped_ids = [str(oid) for oid in (result.get("skipped_objects") or [])]
        if skipped_ids:
            restored, rerender_skipped, restore_error = _restore_skipped_and_rerender(
                chapter_id, page_index, skipped_ids)
            restored_regions += restored
            if restore_error is not None:
                restore_failures.append({
                    "page": page_index + 1, "ids": skipped_ids, "reason": restore_error,
                })
            else:
                for oid in skipped_ids:
                    review_list.append({
                        "page": page_index + 1, "id": oid,
                        "reason": "Bản dịch không vừa vùng chữ; đã giữ ảnh gốc",
                    })
            for oid in rerender_skipped:
                review_list.append({
                    "page": page_index + 1, "id": oid,
                    "reason": "Bản dịch không vừa vùng chữ; đã giữ ảnh gốc",
                })

    latest = load_manifest_raw(chapter_id)
    result = urlify_manifest(latest)
    result["chapter_render"] = {
        "rendered": rendered,
        "reused": reused,
        "skipped": skipped,
        "total": len(latest.get("pages", [])),
        "restored_regions": restored_regions,
        "restore_failures": restore_failures,
        "review_list": review_list,
        "download_url": f"/api/export/{chapter_id}.zip" + ("?force=true" if force else ""),
        "editorial_preflight": preflight,
    }
    return result


@router.get(
    "/export/{chapter_id}.zip",
    responses={409: {"description": "At least one rendered page is stale or unavailable"}},
)
def export_chapter(chapter_id: str, force: bool = False, format: str = "png", quality: int = 90, width: int = 0,
                   split: int = 0, prefix: str = "page_", archive: str = "zip"):
    validate_chapter_id(chapter_id)
    try:
        options = ExportOptions.parse(format, quality, width, split, prefix, archive)
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
    final_archive = write_chapter_archive(chapter_id, force, options)
    return FileResponse(
        final_archive,
        filename=f"manga-translator-{chapter_id}.{options.archive}",
        media_type="application/vnd.comicbook+zip" if options.archive == "cbz" else "application/zip",
    )


def write_chapter_archive(chapter_id: str, force: bool = False, options: ExportOptions = ExportOptions()) -> Path:
    out_dir = OUTPUT_DIR / chapter_id
    out_dir.mkdir(parents=True, exist_ok=True)
    final_archive = out_dir / (f"chapter_{chapter_id}.zip" if options.default
                               else f"chapter_{chapter_id}.{options.key}.{options.archive}")
    tmp_archive = out_dir / f"chapter_{chapter_id}.export.{uuid.uuid4().hex[:12]}.tmp"

    snapshot = _snapshot_export_inputs(chapter_id, force)
    groups: dict[int, list[dict]] = {}
    for item in snapshot:
        path = item["path"]
        validate_image_size(path)
        groups.setdefault(int(item["source_page"]), []).append(item)

    try:
        with zipfile.ZipFile(tmp_archive, "w", compression=zipfile.ZIP_STORED) as archive:
            written = 0
            for output_index, source_page in enumerate(sorted(groups), start=1):
                items = sorted(
                    groups[source_page],
                    key=lambda item: (int(item["slice_index"]), int(item["page_index"])),
                )
                _validate_stitch_group(source_page, items)
                page_tmp = out_dir / (
                    f"page_{output_index:03d}.export.{uuid.uuid4().hex[:12]}.png"
                )
                try:
                    _stitch_png_to_file(
                        [item["path"] for item in items],
                        page_tmp,
                        [item["core_range"] for item in items],
                    )
                    if options.default:
                        archive.write(page_tmp, arcname=f"page_{output_index:03d}.png")
                    else:
                        with Image.open(page_tmp) as stitched:
                            files = finish_page(stitched, options)
                        for data in files:
                            written += 1
                            archive.writestr(f"{options.prefix}{written:03d}.{FORMATS[options.fmt][1]}", data)
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
            _editorial_gate_or_409(load_manifest_raw(chapter_id), require_final_approval=True, force=force)
            atomic_replace(tmp_archive, final_archive)
    finally:
        if tmp_archive.exists():
            try:
                tmp_archive.unlink()
            except OSError:
                pass
    return final_archive
