from __future__ import annotations

import os

from app.optimized_pipeline import OptimizedChapterPipeline


_PIPELINE_ENV = "MANGA_PROCESSING_PIPELINE"
_FAST_NAMES = {"", "fast", "optimized", "one-shot", "oneshot"}


def processing_pipeline_profile(value: str | None = None) -> str:
    raw = os.getenv(_PIPELINE_ENV, "") if value is None else value
    name = str(raw or "").strip().lower()
    if name in _FAST_NAMES:
        return "fast"
    raise ValueError(f"Unsupported {_PIPELINE_ENV}={raw!r}; the only pipeline is fast")


def build_processing_pipeline(profile: str | None = None) -> OptimizedChapterPipeline:
    processing_pipeline_profile(profile)
    return OptimizedChapterPipeline()
