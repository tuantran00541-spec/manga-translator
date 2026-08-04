"""Pydantic schemas for request validation across API endpoints."""

from pydantic import BaseModel, field_validator
from app.security import MAX_RENDER_TEXT_LEN, MAX_RENDER_TRANSLATIONS


class ChapterRequest(BaseModel):
    url: str
    workers: int = 2


class OcrBoxRequest(BaseModel):
    chapter_id: str
    page_index: int
    box_index: int
    lang: str


class RenderRequest(BaseModel):
    chapter_id: str
    page_index: int
    translations: dict[int, str]
    colors: dict[str, str] = {}
    fonts: dict[str, str] = {}
    font_sizes: dict[str, int | str] = {}
    bolds: dict[str, bool] = {}
    stroke_widths: dict[str, int | str] = {}
    stroke_colors: dict[str, str] = {}
    bg_colors: dict[str, str] = {}
    corner_radii: dict[str, int] = {}

    @field_validator("translations")
    @classmethod
    def _check_translations(cls, v: dict[int, str]) -> dict[int, str]:
        if len(v) > MAX_RENDER_TRANSLATIONS:
            raise ValueError(f"Too many translations: {len(v)} > {MAX_RENDER_TRANSLATIONS}")
        for k, val in v.items():
            if len(val) > MAX_RENDER_TEXT_LEN:
                raise ValueError(f"Translation {k} too long: {len(val)} chars")
        return v


class SaveDraftRequest(BaseModel):
    chapter_id: str
    drafts: dict[str, dict] = {}


class ProcessPagesRequest(BaseModel):
    chapter_id: str
    page_indices: list[int]
    workers: int = 2


class SkipPagesRequest(BaseModel):
    chapter_id: str
    page_indices: list[int]
    skipped: bool


class AddBoxRequest(BaseModel):
    chapter_id: str
    page_index: int
    x1: int
    y1: int
    x2: int
    y2: int


class RemoveBoxRequest(BaseModel):
    chapter_id: str
    page_index: int
    box_index: int


class RegionModel(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class SaveExcludedRegionsRequest(BaseModel):
    chapter_id: str
    page_index: int
    excluded_regions: list[RegionModel]
