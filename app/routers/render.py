from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.render.font_catalog import FontNotFoundError, load_font_catalog
from app.render.text_renderer import list_available_fonts

router = APIRouter(prefix="/api", tags=["render"])

_FONT_MEDIA_TYPES = {
    ".ttf": "font/ttf",
    ".otf": "font/otf",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
}


@router.get("/fonts")
def get_available_fonts() -> list[dict]:
    return list_available_fonts()


@router.get("/fonts/{font_id}/file")
def get_font_file(font_id: str) -> FileResponse:
    try:
        path: Path = load_font_catalog().resolve(font_id)
    except FontNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Font file not found: {font_id}")
    media_type = _FONT_MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")
    return FileResponse(
        path,
        media_type=media_type,
        headers={"Cache-Control": "public, max-age=86400"},
    )
