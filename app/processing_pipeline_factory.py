from __future__ import annotations

import os

from app.mask_recall_envelope_v2_pipeline import CompleteFlatEnvelopeMaskRecallPipeline
from app.optimized_pipeline import OptimizedChapterPipeline


_PIPELINE_ENV = "MANGA_PROCESSING_PIPELINE"
_FAST_NAMES = {"", "fast", "optimized", "one-shot", "oneshot"}
_RECALL_NAMES = {"recall", "high-recall", "complete-recall"}


def processing_pipeline_profile(value: str | None = None) -> str:
    raw = os.getenv(_PIPELINE_ENV, "") if value is None else value
    name = str(raw or "").strip().lower()
    if name in _FAST_NAMES:
        return "fast"
    if name in _RECALL_NAMES:
        return "high-recall"
    raise ValueError(
        f"Unsupported {_PIPELINE_ENV}={raw!r}; expected fast or high-recall"
    )


def build_processing_pipeline(
    profile: str | None = None,
) -> OptimizedChapterPipeline:
    """Build the exact pipeline used by the web app and performance gates.

    Fast one-shot cleanup is the production default. The previous complete
    mask-recall stack remains available only as an explicit diagnostic/quality
    profile so benchmarks cannot silently measure a different runtime path.
    """
    selected = processing_pipeline_profile(profile)
    if selected == "high-recall":
        return CompleteFlatEnvelopeMaskRecallPipeline()
    return OptimizedChapterPipeline()
