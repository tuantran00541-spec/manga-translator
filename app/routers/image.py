from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.config import OUTPUT_DIR, PROCESSED_DIR, RAW_DIR
from app.logging_config import logger
from app.manifest_utils import load_manifest_raw
from app.render.identity import render_artifact_is_current
from app.security import validate_chapter_id, validate_image_size, validate_managed_path

router = APIRouter(prefix="/api", tags=["images"])


def _page_file_path(chapter_id: str, page: dict, field: str) -> Path | None:
    value = page.get(field)
    if not value:
        return None
    if field == "original":
        root = RAW_DIR / chapter_id
    elif field == "clean":
        root = PROCESSED_DIR / chapter_id
    else:
        raise ValueError(f"Unsupported manifest image field: {field}")
    return validate_managed_path(value, root)


def _rendered_file_path(chapter_id: str, page_index: int) -> Path:
    return validate_managed_path(
        OUTPUT_DIR / chapter_id / f"page_{page_index:03d}.png",
        OUTPUT_DIR / chapter_id,
    )


def _fallback_page_path(chapter_id: str, page: dict) -> Path | None:
    clean_path = _page_file_path(chapter_id, page, "clean")
    if clean_path and clean_path.exists():
        return clean_path
    return _page_file_path(chapter_id, page, "original")


def _current_rendered_path(chapter_id: str, page_index: int, manifest: dict) -> Path | None:
    path = _rendered_file_path(chapter_id, page_index)
    if render_artifact_is_current(manifest, page_index, path):
        return path
    return None


@router.get(
    "/image/{chapter_id}/{page_index}/{kind}",
    responses={
        400: {"description": "Invalid image request"},
        404: {"description": "Image not found"},
        500: {"description": "Image could not be served"},
    },
)
def get_image(chapter_id: str, page_index: int, kind: str):
    validate_chapter_id(chapter_id)
    if page_index < 0:
        raise HTTPException(400, "page_index must be >= 0")
    if kind not in ("original", "clean", "rendered"):
        raise HTTPException(400, f"Invalid kind: {kind}")

    try:
        manifest = load_manifest_raw(chapter_id)
        pages = manifest.get("pages", [])
        if page_index >= len(pages):
            raise HTTPException(404, f"Page index {page_index} out of range")
        page = pages[page_index]

        if kind == "rendered":
            path = _current_rendered_path(chapter_id, page_index, manifest)
            if path is None:
                logger.warning(
                    "Chapter {} page {} rendered artifact is stale or unavailable; serving current base image",
                    chapter_id,
                    page_index,
                )
                path = _fallback_page_path(chapter_id, page)
        else:
            path = _page_file_path(chapter_id, page, kind)
            if path is None:
                raise HTTPException(404, f"{kind} image not available for page {page_index}")

        if path is None or not path.exists():
            raise HTTPException(404, "Image file not found")
        validate_image_size(path)
        return FileResponse(path)
    except HTTPException:
        raise
    except Exception as exc:
        logger.opt(exception=True).error(
            "Chapter {} page {} operation 'get_image' ({}) failed: {}",
            chapter_id,
            page_index,
            kind,
            exc,
        )
        raise HTTPException(500, "Cannot serve image") from exc


@router.get(
    "/download/{chapter_id}/{page_index}",
    responses={
        400: {"description": "Invalid download request"},
        404: {"description": "Chapter or page not found"},
        409: {"description": "Rendered output is stale or unavailable"},
        500: {"description": "Download failed"},
    },
)
def download_page(chapter_id: str, page_index: int):
    validate_chapter_id(chapter_id)
    if page_index < 0:
        raise HTTPException(400, "page_index must be >= 0")

    try:
        manifest = load_manifest_raw(chapter_id)
        pages = manifest.get("pages", [])
        if page_index >= len(pages):
            raise HTTPException(404, f"Page index {page_index} out of range")

        path = _current_rendered_path(chapter_id, page_index, manifest)
        if path is None:
            raise HTTPException(
                409,
                "Rendered output is stale or unavailable. Kết xuất lại trang trước khi tải.",
            )
        validate_image_size(path)
        return FileResponse(
            path,
            filename=f"page_{page_index + 1:03d}_rendered.png",
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.opt(exception=True).error(
            "Chapter {} page {} operation 'download_page' failed: {}",
            chapter_id,
            page_index,
            exc,
        )
        raise HTTPException(500, "Cannot download page image") from exc
