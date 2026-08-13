from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.config import OUTPUT_DIR
from app.logging_config import logger
from app.manifest_utils import load_manifest_raw
from app.security import validate_chapter_id, validate_image_size

router = APIRouter(prefix="/api", tags=["images"])


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
                path = OUTPUT_DIR / chapter_id / f"page_{page_index:03d}.png"
                if not path.exists():
                    logger.warning(
                        "Chapter %s page %s: rendered=True but rendered file missing, using fallback",
                        chapter_id,
                        page_index,
                    )
                    clean_p = Path(page["clean"]) if page.get("clean") else None
                    path = clean_p if (clean_p and clean_p.exists()) else Path(page["original"])
            else:
                clean_p = Path(page["clean"]) if page.get("clean") else None
                path = clean_p if (clean_p and clean_p.exists()) else Path(page["original"])
        else:
            field = page.get(kind) if kind != "original" else page.get("original")
            if not field:
                raise HTTPException(404, f"{kind} image not available for page {page_index}")
            path = Path(field)

        if not path.exists():
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
            path = OUTPUT_DIR / chapter_id / f"page_{page_index:03d}.png"
            if not path.exists():
                logger.warning(
                    "Chapter %s page %s: rendered=True but rendered file missing for download, using fallback",
                    chapter_id,
                    page_index,
                )
                clean_p = Path(page["clean"]) if page.get("clean") else None
                path = clean_p if (clean_p and clean_p.exists()) else Path(page["original"])
        else:
            clean_p = Path(page["clean"]) if page.get("clean") else None
            path = clean_p if (clean_p and clean_p.exists()) else Path(page["original"])

        if not path.exists():
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
