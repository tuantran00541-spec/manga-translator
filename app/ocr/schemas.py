from __future__ import annotations

from pydantic import BaseModel, field_validator

from app.parameters import OCR_JOB_CONCURRENCY_LIMIT

_ALLOWED_OCR_LANGS = {"ja", "japan", "ch", "zh", "korean", "ko", "en"}


class ChapterOCRRequest(BaseModel):
    chapter_id: str
    lang: str = "ja"
    concurrency: int = 1
    force: bool = False

    @field_validator("lang")
    @classmethod
    def _valid_lang(cls, value: str) -> str:
        normalized = (value or "").strip().lower()
        if normalized not in _ALLOWED_OCR_LANGS:
            raise ValueError("Unsupported OCR language")
        return normalized

    @field_validator("concurrency")
    @classmethod
    def _bounded_concurrency(cls, value: int) -> int:
        if value < 1 or value > OCR_JOB_CONCURRENCY_LIMIT:
            raise ValueError(
                f"OCR concurrency must be between 1 and {OCR_JOB_CONCURRENCY_LIMIT}"
            )
        return value
