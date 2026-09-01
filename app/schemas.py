from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.parameters import PIPELINE_DEFAULT_WORKERS
from app.security import (
    MAX_REMOTE_URL_LENGTH,
    MAX_RENDER_TEXT_LEN,
    MAX_RENDER_TRANSLATIONS,
)

ALLOWED_TEXT_OBJECT_SHAPES = ("rectangle", "ellipse")
ALLOWED_HORIZONTAL_ALIGNS = ("left", "center", "right")
ALLOWED_VERTICAL_ALIGNS = ("top", "middle", "bottom")
ALLOWED_OCR_LANGS = {"ja", "japan", "ch", "zh", "en", "korean", "ko"}
TEXT_OBJECT_MIN_SIZE = 10
MAX_PAGE_SELECTION = 512
MAX_DRAFT_ENTRIES = 5000
MAX_TEXT_OBJECT_ID_LEN = 128


def _validate_region_coords(region: "TextObjectRegion") -> "TextObjectRegion":
    if region.x1 < 0 or region.y1 < 0:
        raise ValueError("Region coordinates must be non-negative")
    if region.x1 >= region.x2 or region.y1 >= region.y2:
        raise ValueError("Region must have x1 < x2 and y1 < y2")
    return region


def _validate_page_indices(values: list[int]) -> list[int]:
    if len(values) > MAX_PAGE_SELECTION:
        raise ValueError(
            f"Too many page indices: {len(values)} > {MAX_PAGE_SELECTION}"
        )
    if any(index < 0 for index in values):
        raise ValueError("page indices must be non-negative")
    return list(dict.fromkeys(values))


def _validate_ocr_lang(value: str) -> str:
    lang = str(value or "").strip().lower()
    if lang not in ALLOWED_OCR_LANGS:
        raise ValueError(f"Unsupported OCR language: {value!r}")
    return lang


def _validate_text_object_id(value: str) -> str:
    value = str(value or "").strip()
    if not value:
        raise ValueError("text_object id is required")
    if len(value) > MAX_TEXT_OBJECT_ID_LEN:
        raise ValueError("text_object id is too long")
    return value


class ChapterRequest(BaseModel):
    url: str
    workers: int = Field(default=PIPELINE_DEFAULT_WORKERS, ge=1, le=8)

    @field_validator("url")
    @classmethod
    def _url_length(cls, value: str) -> str:
        value = str(value or "").strip()
        if not value:
            raise ValueError("URL is required")
        if len(value) > MAX_REMOTE_URL_LENGTH:
            raise ValueError("URL is too long")
        return value


class OcrBoxRequest(BaseModel):
    chapter_id: str
    page_index: int = Field(ge=0)
    box_index: int = Field(ge=0)
    lang: str

    @field_validator("lang")
    @classmethod
    def _lang(cls, value: str) -> str:
        return _validate_ocr_lang(value)


class WorkflowCheckpointRequest(BaseModel):
    chapter_id: str
    stage: Literal["preview", "review", "editor"]
    page_index: int = Field(ge=0)


class RenderRequest(BaseModel):
    chapter_id: str
    page_index: int = Field(ge=0)
    translations: dict[str, str]
    colors: dict[str, str] = Field(default_factory=dict)
    fonts: dict[str, str] = Field(default_factory=dict)
    font_sizes: dict[str, int | str] = Field(default_factory=dict)
    bolds: dict[str, bool] = Field(default_factory=dict)
    stroke_widths: dict[str, int | str] = Field(default_factory=dict)
    stroke_colors: dict[str, str] = Field(default_factory=dict)
    bg_colors: dict[str, str] = Field(default_factory=dict)
    corner_radii: dict[str, int] = Field(default_factory=dict)
    horizontal_aligns: dict[str, str] = Field(default_factory=dict)
    vertical_aligns: dict[str, str] = Field(default_factory=dict)

    @field_validator("translations")
    @classmethod
    def _check_translations(cls, v: dict[str, str]) -> dict[str, str]:
        if len(v) > MAX_RENDER_TRANSLATIONS:
            raise ValueError(
                f"Too many translations: {len(v)} > {MAX_RENDER_TRANSLATIONS}"
            )
        for key, val in v.items():
            if len(str(key)) > MAX_TEXT_OBJECT_ID_LEN:
                raise ValueError("Translation key is too long")
            if len(val) > MAX_RENDER_TEXT_LEN:
                raise ValueError(
                    f"Translation {key} too long: {len(val)} chars"
                )
        return v

    @field_validator("horizontal_aligns")
    @classmethod
    def _check_horizontal_aligns(cls, v: dict[str, str]) -> dict[str, str]:
        if len(v) > MAX_RENDER_TRANSLATIONS:
            raise ValueError(
                f"Too many horizontal_aligns: {len(v)} > {MAX_RENDER_TRANSLATIONS}"
            )
        for key, val in v.items():
            if val not in ALLOWED_HORIZONTAL_ALIGNS:
                raise ValueError(
                    f"horizontalAlign {val!r} for object {key} must be one of {ALLOWED_HORIZONTAL_ALIGNS}"
                )
        return v

    @field_validator("vertical_aligns")
    @classmethod
    def _check_vertical_aligns(cls, v: dict[str, str]) -> dict[str, str]:
        if len(v) > MAX_RENDER_TRANSLATIONS:
            raise ValueError(
                f"Too many vertical_aligns: {len(v)} > {MAX_RENDER_TRANSLATIONS}"
            )
        for key, val in v.items():
            if val not in ALLOWED_VERTICAL_ALIGNS:
                raise ValueError(
                    f"verticalAlign {val!r} for object {key} must be one of {ALLOWED_VERTICAL_ALIGNS}"
                )
        return v


class SaveDraftRequest(BaseModel):
    chapter_id: str
    drafts: dict[str, dict] = Field(default_factory=dict)

    @field_validator("drafts")
    @classmethod
    def _draft_limits(cls, value: dict[str, dict]) -> dict[str, dict]:
        if len(value) > MAX_DRAFT_ENTRIES:
            raise ValueError(
                f"Too many draft entries: {len(value)} > {MAX_DRAFT_ENTRIES}"
            )
        for key, item in value.items():
            if len(str(key)) > 128:
                raise ValueError("Draft key is too long")
            if not isinstance(item, dict):
                raise ValueError("Draft entries must be objects")
            text = item.get("text")
            if text is not None and len(str(text)) > MAX_RENDER_TEXT_LEN:
                raise ValueError(f"Draft {key} text is too long")
        return value


class ProcessPagesRequest(BaseModel):
    chapter_id: str
    page_indices: list[int]
    workers: int = Field(default=PIPELINE_DEFAULT_WORKERS, ge=1, le=8)

    @field_validator("page_indices")
    @classmethod
    def _pages(cls, values: list[int]) -> list[int]:
        return _validate_page_indices(values)


class SkipPagesRequest(BaseModel):
    chapter_id: str
    page_indices: list[int]
    skipped: bool

    @field_validator("page_indices")
    @classmethod
    def _pages(cls, values: list[int]) -> list[int]:
        return _validate_page_indices(values)


class AddBoxRequest(BaseModel):
    chapter_id: str
    page_index: int = Field(ge=0)
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
    page_index: int = Field(ge=0)
    box_index: int = Field(ge=0)
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
    page_index: int = Field(ge=0)
    box_index: int = Field(ge=0)


class RegionModel(BaseModel):
    x1: int = Field(ge=0)
    y1: int = Field(ge=0)
    x2: int = Field(ge=0)
    y2: int = Field(ge=0)

    @field_validator("x2")
    @classmethod
    def _x_order(cls, value: int, info):
        x1 = info.data.get("x1")
        if x1 is not None and value <= x1:
            raise ValueError("x2 must be greater than x1")
        return value

    @field_validator("y2")
    @classmethod
    def _y_order(cls, value: int, info):
        y1 = info.data.get("y1")
        if y1 is not None and value <= y1:
            raise ValueError("y2 must be greater than y1")
        return value


class SaveExcludedRegionsRequest(BaseModel):
    chapter_id: str
    page_index: int = Field(ge=0)
    excluded_regions: list[RegionModel]

    @field_validator("excluded_regions")
    @classmethod
    def _region_count(cls, value: list[RegionModel]) -> list[RegionModel]:
        if len(value) > MAX_RENDER_TRANSLATIONS:
            raise ValueError("Too many excluded regions")
        return value


class ResetManualMaskRequest(BaseModel):
    chapter_id: str
    page_index: int = Field(ge=0)


class TextObjectRegion(BaseModel):
    x1: int = Field(ge=0)
    y1: int = Field(ge=0)
    x2: int = Field(ge=0)
    y2: int = Field(ge=0)


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

    @field_validator(
        "color",
        "font",
        "fontSize",
        "strokeWidth",
        "strokeColor",
        "bgColor",
        "cornerRadius",
    )
    @classmethod
    def _style_string_limit(cls, value: str) -> str:
        if len(str(value)) > 256:
            raise ValueError("Style value is too long")
        return value

    @field_validator("horizontalAlign")
    @classmethod
    def _h_align(cls, v: str) -> str:
        if v not in ALLOWED_HORIZONTAL_ALIGNS:
            raise ValueError(
                f"horizontalAlign must be one of {ALLOWED_HORIZONTAL_ALIGNS}"
            )
        return v

    @field_validator("verticalAlign")
    @classmethod
    def _v_align(cls, v: str) -> str:
        if v not in ALLOWED_VERTICAL_ALIGNS:
            raise ValueError(
                f"verticalAlign must be one of {ALLOWED_VERTICAL_ALIGNS}"
            )
        return v


def _validate_text_object_shape(v: str) -> str:
    if v not in ALLOWED_TEXT_OBJECT_SHAPES:
        raise ValueError("shape must be 'rectangle' or 'ellipse'")
    return v


class CreateTextObjectRequest(BaseModel):
    chapter_id: str
    page_index: int = Field(ge=0)
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
    page_index: int = Field(ge=0)
    id: str
    shape: str | None = None
    region: TextObjectRegion | None = None
    ocr_text: str | None = None
    translation: str | None = None
    style: TextObjectStyle | None = None

    @field_validator("id")
    @classmethod
    def _id_not_empty(cls, v: str) -> str:
        return _validate_text_object_id(v)

    @field_validator("ocr_text", "translation")
    @classmethod
    def _text_limit(cls, value: str | None) -> str | None:
        if value is not None and len(value) > MAX_RENDER_TEXT_LEN:
            raise ValueError("Text value is too long")
        return value

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
    page_index: int = Field(ge=0)
    id: str

    @field_validator("id")
    @classmethod
    def _id_not_empty(cls, v: str) -> str:
        return _validate_text_object_id(v)


class OcrTextObjectRequest(BaseModel):
    chapter_id: str
    page_index: int = Field(ge=0)
    id: str
    lang: str = "ja"

    @field_validator("id")
    @classmethod
    def _id_not_empty(cls, v: str) -> str:
        return _validate_text_object_id(v)

    @field_validator("lang")
    @classmethod
    def _lang(cls, value: str) -> str:
        return _validate_ocr_lang(value)


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
    page_index: int = Field(ge=0)
