"""Pydantic schemas for request validation across API endpoints."""

from pydantic import BaseModel, ConfigDict, field_validator
from app.security import MAX_RENDER_TEXT_LEN, MAX_RENDER_TRANSLATIONS

ALLOWED_TEXT_OBJECT_SHAPES = ("rectangle", "ellipse")
TEXT_OBJECT_MIN_SIZE = 10


def _validate_region_coords(region: "TextObjectRegion") -> "TextObjectRegion":
    if region.x1 < 0 or region.y1 < 0:
        raise ValueError("Region coordinates must be non-negative")
    if region.x1 >= region.x2 or region.y1 >= region.y2:
        raise ValueError("Region must have x1 < x2 and y1 < y2")
    return region


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
    translations: dict[str, str]
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

    @field_validator("x1", "y1", "x2", "y2")
    @classmethod
    def _non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Box coordinates must be non-negative")
        return v


class UpdateBoxRequest(BaseModel):
    chapter_id: str
    page_index: int
    box_index: int
    x1: int
    y1: int
    x2: int
    y2: int

    @field_validator("x1", "y1", "x2", "y2")
    @classmethod
    def _non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Box coordinates must be non-negative")
        return v


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


class ResetManualMaskRequest(BaseModel):
    chapter_id: str
    page_index: int


class TextObjectRegion(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int

    @field_validator("x1", "y1", "x2", "y2")
    @classmethod
    def _finite(cls, v: int) -> int:
        if isinstance(v, float) and not (v != v):  # NaN guard
            raise ValueError("Region coordinates must be finite")
        return v


class TextObjectStyle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    color: str = "auto"
    font: str = "default"
    fontSize: str = "auto"
    bold: bool = False
    strokeWidth: str = "auto"
    strokeColor: str = "auto"
    bgColor: str = "transparent"
    cornerRadius: str = "0"


def _validate_text_object_shape(v: str) -> str:
    if v not in ALLOWED_TEXT_OBJECT_SHAPES:
        raise ValueError("shape must be 'rectangle' or 'ellipse'")
    return v


class CreateTextObjectRequest(BaseModel):
    chapter_id: str
    page_index: int
    shape: str = "rectangle"
    region: TextObjectRegion

    @field_validator("shape")
    @classmethod
    def _shape(cls, v: str) -> str:
        return _validate_text_object_shape(v)

    @field_validator("region")
    @classmethod
    def _region(cls, v: TextObjectRegion) -> TextObjectRegion:
        return _validate_region_coords(v)


class UpdateTextObjectRequest(BaseModel):
    chapter_id: str
    page_index: int
    id: str
    shape: str | None = None
    region: TextObjectRegion | None = None
    ocr_text: str | None = None
    translation: str | None = None
    style: TextObjectStyle | None = None

    @field_validator("id")
    @classmethod
    def _id_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("text_object id is required")
        return v

    @field_validator("shape")
    @classmethod
    def _shape(cls, v: str | None) -> str | None:
        if v is None:
            return v
        return _validate_text_object_shape(v)

    @field_validator("region")
    @classmethod
    def _region(cls, v: TextObjectRegion | None) -> TextObjectRegion | None:
        if v is None:
            return v
        return _validate_region_coords(v)


class DeleteTextObjectRequest(BaseModel):
    chapter_id: str
    page_index: int
    id: str

    @field_validator("id")
    @classmethod
    def _id_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("text_object id is required")
        return v


class OcrTextObjectRequest(BaseModel):
    chapter_id: str
    page_index: int
    id: str
    lang: str = "ja"

    @field_validator("id")
    @classmethod
    def _id_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("text_object id is required")
        return v
