"""API Router for serving original, cleaned, rendered images and downloads."""

from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.config import OUTPUT_DIR
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

    if kind == "rendered":
        path = OUTPUT_DIR / chapter_id / f"page_{page_index:03d}.png"
    else:
        manifest = load_manifest_raw(chapter_id)
        if page_index >= len(manifest["pages"]):
            raise HTTPException(404, "Page not found")
        page = manifest["pages"][page_index]
        field = page.get(kind) if kind != "original" else page.get("original")
        if not field:
            raise HTTPException(404, f"{kind} not available for this page")
        path = Path(field)

    if not path.exists():
        raise HTTPException(404, "Image file not found")
    validate_image_size(path)
    return FileResponse(path)


@router.get("/download/{chapter_id}/{page_index}")
def download_page(chapter_id: str, page_index: int):
    validate_chapter_id(chapter_id)
    if page_index < 0:
        raise HTTPException(400, "page_index must be >= 0")
    path = OUTPUT_DIR / chapter_id / f"page_{page_index:03d}.png"
    if not path.exists():
        raise HTTPException(404, "Output image not found")
    return FileResponse(path)
