import pytest

from app.mask_recall_envelope_v2_pipeline import CompleteFlatEnvelopeMaskRecallPipeline
from app.optimized_pipeline import OptimizedChapterPipeline
from app.processing_pipeline_factory import (
    build_processing_pipeline,
    processing_pipeline_profile,
)


def test_processing_pipeline_defaults_to_fast_one_shot(monkeypatch):
    monkeypatch.delenv("MANGA_PROCESSING_PIPELINE", raising=False)

    assert processing_pipeline_profile() == "fast"
    pipeline = build_processing_pipeline()

    assert type(pipeline) is OptimizedChapterPipeline


def test_high_recall_pipeline_is_explicit_opt_in(monkeypatch):
    monkeypatch.setenv("MANGA_PROCESSING_PIPELINE", "high-recall")

    assert processing_pipeline_profile() == "high-recall"
    pipeline = build_processing_pipeline()

    assert type(pipeline) is CompleteFlatEnvelopeMaskRecallPipeline


def test_unknown_pipeline_profile_fails_loudly(monkeypatch):
    monkeypatch.setenv("MANGA_PROCESSING_PIPELINE", "mystery")

    with pytest.raises(ValueError):
        build_processing_pipeline()
