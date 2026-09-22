from fastapi import APIRouter, HTTPException
from PIL import Image

from app.render.font_matcher import match_fonts
from app.render.text_renderer import list_available_fonts
from app.schemas import FontMatchRequest
from app.security import validate_chapter_id

router = APIRouter(prefix="/api", tags=["render"])


@router.get("/fonts")
def get_available_fonts() -> list[dict]:
    return list_available_fonts()


@router.post("/fonts/match")
def match_available_font(req: FontMatchRequest) -> dict:
    """Suggest installed fonts by comparing the original lettering crop."""

    validate_chapter_id(req.chapter_id)
    from app.manifest_utils import get_manifest_lock, load_manifest_raw

    with get_manifest_lock(req.chapter_id):
        manifest = load_manifest_raw(req.chapter_id)
    pages = manifest.get("pages") or []
    if req.page_index >= len(pages):
        raise HTTPException(400, f"Invalid page_index: {req.page_index}")
    page = pages[req.page_index]
    obj = next(
        (item for item in page.get("text_objects", []) if isinstance(item, dict) and item.get("id") == req.object_id),
        None,
    )
    if obj is None:
        raise HTTPException(404, f"Unknown text object: {req.object_id}")
    region = obj.get("ocr_text_region") or obj.get("region")
    try:
        coords = tuple(int(region[key]) for key in ("x1", "y1", "x2", "y2"))
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(400, "Text object has no valid region") from exc
    source_value = page.get("original") or page.get("source") or page.get("raw")
    if not source_value:
        raise HTTPException(404, "Original page image is unavailable")
    try:
        image = Image.open(source_value).convert("RGB")
    except FileNotFoundError as exc:
        raise HTTPException(404, "Original page image is unavailable") from exc
    except Exception as exc:
        raise HTTPException(500, "Cannot open original page image") from exc
    source_text = req.source_text or obj.get("ocr_text") or obj.get("source_text") or ""
    matches = match_fonts(
        image,
        coords,
        source_text,
        category=req.category,
        top_k=req.top_k,
    )
    if not matches:
        raise HTTPException(400, "No fonts are available for the requested category")
    return {
        "chapter_id": req.chapter_id,
        "page_index": req.page_index,
        "object_id": req.object_id,
        "source_text": source_text,
        "matches": [item.as_dict() for item in matches],
    }
