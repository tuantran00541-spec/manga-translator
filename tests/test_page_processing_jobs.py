import asyncio
import time

from app.page_processing_jobs import ChapterProcessingJobManager


def _wait_for_terminal(manager, job_id, timeout=3.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        snapshot = manager.snapshot(job_id)
        if snapshot["status"] in {"completed", "failed"}:
            return snapshot
        time.sleep(0.01)
    raise AssertionError("processing job did not finish in time")


def test_server_dispatches_full_queue_once_after_client_disconnect(monkeypatch, tmp_path):
    import app.page_processing_jobs as jobs_module

    monkeypatch.setattr(jobs_module, "PROCESSED_DIR", tmp_path)
    chapter_id = "deadbeef"
    (tmp_path / chapter_id).mkdir()
    calls = []
    completed_chapters = []

    def process_plan(chapter, pages, workers, progress_callback):
        calls.append((chapter, list(pages), workers))
        for page_index in pages:
            progress_callback(page_index)
            time.sleep(0.002)
        return {"pages": pages}

    async def scenario():
        manager = ChapterProcessingJobManager(
            process_plan,
            on_completed=completed_chapters.append,
        )
        first = manager.start(
            chapter_id,
            page_indices=list(range(40)),
            workers=6,
        )
        job_id = first["job_id"]

        duplicate = manager.start(
            chapter_id,
            page_indices=list(range(40)),
            workers=6,
        )
        assert duplicate["job_id"] == job_id
        assert duplicate["reused"] is True

        await asyncio.sleep(0.02)
        reconnected = manager.latest_for_chapter(chapter_id)
        assert reconnected is not None
        assert reconnected["job_id"] == job_id
        assert reconnected["status"] in {"running", "completed"}
        if reconnected["status"] == "running":
            assert 0 < reconnected["completed"] < 40
            assert reconnected["current_batch"] == list(range(40))

        return await asyncio.to_thread(_wait_for_terminal, manager, job_id)

    terminal = asyncio.run(scenario())

    assert terminal["status"] == "completed"
    assert terminal["completed"] == 40
    assert terminal["remaining"] == 0
    assert terminal["workers"] == 6
    assert [pages for _, pages, _ in calls] == [list(range(40))]
    assert completed_chapters == [chapter_id]


def test_processing_job_keeps_partial_progress_when_full_queue_fails(monkeypatch, tmp_path):
    import app.page_processing_jobs as jobs_module

    monkeypatch.setattr(jobs_module, "PROCESSED_DIR", tmp_path)
    chapter_id = "cafebabe"
    (tmp_path / chapter_id).mkdir()
    calls = []

    def process_plan(_chapter, pages, _workers, progress_callback):
        calls.append(list(pages))
        for page_index in pages[:16]:
            progress_callback(page_index)
        raise RuntimeError("synthetic full-queue failure")

    async def scenario():
        manager = ChapterProcessingJobManager(process_plan)
        started = manager.start(
            chapter_id,
            page_indices=list(range(40)),
            workers=4,
        )
        return await asyncio.to_thread(
            _wait_for_terminal,
            manager,
            started["job_id"],
        )

    terminal = asyncio.run(scenario())

    assert terminal["status"] == "failed"
    assert terminal["completed"] == 16
    assert terminal["remaining"] == 24
    assert calls == [list(range(40))]
    assert terminal["errors"][0]["error_type"] == "RuntimeError"
