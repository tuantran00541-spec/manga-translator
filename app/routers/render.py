from fastapi import APIRouter

from app.render.text_renderer import list_available_fonts

router = APIRouter(prefix="/api", tags=["render"])


@router.get("/fonts")
def get_available_fonts() -> list[dict[str, str]]:
    return list_available_fonts()
