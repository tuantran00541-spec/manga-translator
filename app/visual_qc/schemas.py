from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, field_validator

from app.parameters import VISUAL_QC_JOB_CONCURRENCY, VISUAL_QC_JOB_CONCURRENCY_LIMIT


class VisualQCChapterRequest(BaseModel):
    chapter_id: str
    concurrency: int = VISUAL_QC_JOB_CONCURRENCY
    provider: Literal["gemini", "deepseek"] = "gemini"
    budget_usd: float = 0.08

    @field_validator("concurrency")
    @classmethod
    def _bounded_concurrency(cls, value: int) -> int:
        if value < 1 or value > VISUAL_QC_JOB_CONCURRENCY_LIMIT:
            raise ValueError(
                "concurrency must be between 1 and "
                f"{VISUAL_QC_JOB_CONCURRENCY_LIMIT}"
            )
        return value

    @field_validator("budget_usd")
    @classmethod
    def _bounded_budget(cls, value: float) -> float:
        value = float(value)
        if not math.isfinite(value) or value < 0.005 or value > 0.15:
            raise ValueError("budget_usd must be between 0.005 and 0.15")
        return value
