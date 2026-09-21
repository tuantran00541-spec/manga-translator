from app.optimized_pipeline import OptimizedChapterPipeline
from app.pipeline import ChapterPipeline


class _FakeInpainter:
    def __init__(self):
        self.prepared_for = None

    def prepare_for_page_workers(self, workers: int):
        self.prepared_for = workers
        return {"page_workers": workers}


def test_optimized_pipeline_preserves_requested_workers(monkeypatch):
    seen = {}

    def fake_process_pages(self, chapter_id, page_indices, workers=2, progress_callback=None):
        seen["chapter_id"] = chapter_id
        seen["page_indices"] = list(page_indices)
        seen["workers"] = workers
        return {"workers": workers}

    monkeypatch.setattr(ChapterPipeline, "process_pages", fake_process_pages)

    pipeline = OptimizedChapterPipeline()
    fake_inpainter = _FakeInpainter()
    pipeline._inpainter = fake_inpainter

    result = pipeline.process_pages("deadbeef", [0, 1, 2], workers=6)

    assert result["workers"] == 6
    assert seen["workers"] == 6
    assert fake_inpainter.prepared_for == 6


def test_optimized_pipeline_clamps_only_to_public_worker_contract(monkeypatch):
    seen = {}

    def fake_process_pages(self, chapter_id, page_indices, workers=2, progress_callback=None):
        seen["workers"] = workers
        return {"workers": workers}

    monkeypatch.setattr(ChapterPipeline, "process_pages", fake_process_pages)

    pipeline = OptimizedChapterPipeline()
    fake_inpainter = _FakeInpainter()
    pipeline._inpainter = fake_inpainter

    low = pipeline.process_pages("deadbeef", [0], workers=0)
    high = pipeline.process_pages("deadbeef", [0], workers=99)

    assert low["workers"] == 2
    assert high["workers"] == 8
    assert fake_inpainter.prepared_for == 8
