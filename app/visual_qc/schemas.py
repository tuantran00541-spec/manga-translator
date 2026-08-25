from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, field_validator


class VisualQCChapterRequest(BaseModel):
    chapter_id: str
    concurrency: int = 2
    provider: Literal["gemini", "deepseek"] = "gemini"
    budget_usd: float = 0.08

    @field_validator("concurrency")
    @classmethod
    def _bounded_concurrency(cls, value: int) -> int:
        if value < 1 or value > 4:
            raise ValueError("concurrency must be between 1 and 4")
        return value

    @field_validator("budget_usd")
    @classmethod
    def _bounded_budget(cls, value: float) -> float:
        value = float(value)
        if not math.isfinite(value) or value < 0.005 or value > 0.15:
            raise ValueError("budget_usd must be between 0.005 and 0.15")
        return value
