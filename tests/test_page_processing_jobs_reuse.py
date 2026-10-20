import asyncio

from app.page_processing_jobs import ChapterProcessingJobManager


def test_job_manager_reuses_an_active_job():
    manager = ChapterProcessingJobManager(lambda *args, **kwargs: None)

    async def run():
        first = manager.start("chapter-a", page_indices=[0, 1], workers=1)
        second = manager.start("chapter-a", page_indices=[2], workers=1)
        return first, second

    first, second = asyncio.run(run())

    assert first["job_id"] == second["job_id"]
    assert second["reused"] is True


def test_job_manager_marks_completed_after_plan_finishes():
    def plan(chapter_id, page_indices, workers, on_page_done, **kwargs):
        for page_index in page_indices:
            on_page_done(page_index)

    manager = ChapterProcessingJobManager(plan)

    async def run():
        started = manager.start("chapter-b", page_indices=[0, 1], workers=1)
        job_id = started["job_id"]
        while manager.snapshot(job_id)["status"] in {"pending", "running"}:
            await asyncio.sleep(0.01)
        return manager.snapshot(job_id)

    finished = asyncio.run(run())

    assert finished["status"] == "completed"
    assert finished["completed"] == 2
    assert finished["done_indices"] == [0, 1]
