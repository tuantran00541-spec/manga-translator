from app.inpaint.lama_inpainter import Inpainter
from app.optimized_pipeline import OptimizedChapterPipeline


def test_optimized_pipeline_uses_plain_lama_inpainter():
    pipeline = OptimizedChapterPipeline()
    assert isinstance(pipeline.inpainter, Inpainter)
