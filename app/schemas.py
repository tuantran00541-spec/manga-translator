from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, field_validator
from app.security import MAX_RENDER_TEXT_LEN, MAX_RENDER_TRANSLATIONS

ALLOWED_TEXT_OBJECT_SHAPES = ("rectangle", "ellipse")
ALLOWED_HORIZONTAL_ALIGNS = ("left", "center", "right")
ALLOWED_VERTICAL_ALIGNS = ("top", "middle", "bottom")
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


class WorkflowCheckpointRequest(BaseModel):
    chapter_id: str
    stage: Literal["preview", "review", "script", "editor", "final_qc"]
    page_index: int


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
    horizontal_aligns: dict[str, str] = {}
    vertical_aligns: dict[str, str] = {}

    @field_validator("translations")
    @classmethod
    def _check_translations(cls, v: dict[int, str]) -> dict[int, str]:
        if len(v) > MAX_RENDER_TRANSLATIONS:
            raise ValueError(f"Too many translations: {len(v)} > {MAX_RENDER_TRANSLATIONS}")
        for k, val in v.items():
            if len(val) > MAX_RENDER_TEXT_LEN:
                raise ValueError(f"Translation {k} too long: {len(val)} chars")
        return v

    @field_validator("horizontal_aligns")
    @classmethod
    def _check_horizontal_aligns(cls, v: dict[str, str]) -> dict[str, str]:
        if len(v) > MAX_RENDER_TRANSLATIONS:
            raise ValueError(f"Too many horizontal_aligns: {len(v)} > {MAX_RENDER_TRANSLATIONS}")
        for k, val in v.items():
            if val not in ALLOWED_HORIZONTAL_ALIGNS:
                raise ValueError(f"horizontalAlign {val!r} for object {k} must be one of {ALLOWED_HORIZONTAL_ALIGNS}")
        return v

    @field_validator("vertical_aligns")
    @classmethod
    def _check_vertical_aligns(cls, v: dict[str, str]) -> dict[str, str]:
        if len(v) > MAX_RENDER_TRANSLATIONS:
            raise ValueError(f"Too many vertical_aligns: {len(v)} > {MAX_RENDER_TRANSLATIONS}")
        for k, val in v.items():
            if val not in ALLOWED_VERTICAL_ALIGNS:
                raise ValueError(f"verticalAlign {val!r} for object {k} must be one of {ALLOWED_VERTICAL_ALIGNS}")
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
        if isinstance(v, float) and not (v != v):
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
    horizontalAlign: str = "center"
    verticalAlign: str = "middle"

    @field_validator("horizontalAlign")
    @classmethod
    def _h_align(cls, v: str) -> str:
        if v not in ALLOWED_HORIZONTAL_ALIGNS:
            raise ValueError(f"horizontalAlign must be one of {ALLOWED_HORIZONTAL_ALIGNS}")
        return v

    @field_validator("verticalAlign")
    @classmethod
    def _v_align(cls, v: str) -> str:
        if v not in ALLOWED_VERTICAL_ALIGNS:
            raise ValueError(f"verticalAlign must be one of {ALLOWED_VERTICAL_ALIGNS}")
        return v


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


class VisualQCKeyRequest(BaseModel):
    api_key: str

    @field_validator("api_key")
    @classmethod
    def _api_key_not_empty(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("Gemini API key is required")
        if len(v) > 4096:
            raise ValueError("Gemini API key is unexpectedly long")
        return v


class VisualQCInspectRequest(BaseModel):
    chapter_id: str
    page_index: int

    @field_validator("page_index")
    @classmethod
    def _page_index_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("page_index must be non-negative")
        return v
