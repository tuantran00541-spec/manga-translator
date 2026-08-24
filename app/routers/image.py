from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.config import OUTPUT_DIR, PROCESSED_DIR, RAW_DIR
from app.logging_config import logger
from app.manifest_utils import load_manifest_raw
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


@router.get("/image/{chapter_id}/{page_index}/{kind}")
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
            if page.get("rendered"):
                path = _rendered_file_path(chapter_id, page_index)
                if not path.exists():
                    logger.warning(
                        "Chapter %s page %s: rendered=True but rendered file missing, using fallback",
                        chapter_id,
                        page_index,
                    )
                    clean_p = _page_file_path(chapter_id, page, "clean")
                    path = clean_p if (clean_p and clean_p.exists()) else _page_file_path(chapter_id, page, "original")
            else:
                clean_p = _page_file_path(chapter_id, page, "clean")
                path = clean_p if (clean_p and clean_p.exists()) else _page_file_path(chapter_id, page, "original")
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
        logger.error(
            "Chapter %s page %s operation 'get_image' (%s) failed: %s",
            chapter_id,
            page_index,
            kind,
            exc,
            exc_info=True,
        )
        raise HTTPException(500, "Cannot serve image") from exc


@router.get("/download/{chapter_id}/{page_index}")
def download_page(chapter_id: str, page_index: int):
    validate_chapter_id(chapter_id)
    if page_index < 0:
        raise HTTPException(400, "page_index must be >= 0")

    try:
        manifest = load_manifest_raw(chapter_id)
        pages = manifest.get("pages", [])
        if page_index >= len(pages):
            raise HTTPException(404, f"Page index {page_index} out of range")
        page = pages[page_index]

        if page.get("rendered"):
            path = _rendered_file_path(chapter_id, page_index)
            if not path.exists():
                logger.warning(
                    "Chapter %s page %s: rendered=True but rendered file missing for download, using fallback",
                    chapter_id,
                    page_index,
                )
                clean_p = _page_file_path(chapter_id, page, "clean")
                path = clean_p if (clean_p and clean_p.exists()) else _page_file_path(chapter_id, page, "original")
        else:
            clean_p = _page_file_path(chapter_id, page, "clean")
            path = clean_p if (clean_p and clean_p.exists()) else _page_file_path(chapter_id, page, "original")

        if path is None or not path.exists():
            raise HTTPException(404, "Output image file not found")
        validate_image_size(path)
        return FileResponse(path)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Chapter %s page %s operation 'download_page' failed: %s",
            chapter_id,
            page_index,
            exc,
            exc_info=True,
        )
        raise HTTPException(500, "Cannot download page image") from exc
