import pytest

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


@pytest.mark.parametrize("name", ["high-recall", "mystery"])
def test_unknown_pipeline_profile_fails_loudly(monkeypatch, name):
    monkeypatch.setenv("MANGA_PROCESSING_PIPELINE", name)

    with pytest.raises(ValueError):
        build_processing_pipeline()
