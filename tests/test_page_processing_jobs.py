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


def test_server_owns_all_batches_after_client_disconnect(monkeypatch, tmp_path):
    import app.page_processing_jobs as jobs_module

    monkeypatch.setattr(jobs_module, "PROCESSED_DIR", tmp_path)
    chapter_id = "deadbeef"
    (tmp_path / chapter_id).mkdir()
    calls = []
    completed_chapters = []

    def process_batch(chapter, pages, workers):
        calls.append((chapter, list(pages), workers))
        time.sleep(0.025)
        return {"pages": pages}

    async def scenario():
        manager = ChapterProcessingJobManager(
            process_batch,
            on_completed=completed_chapters.append,
            batch_size=16,
        )
        first = manager.start(
            chapter_id,
            page_indices=list(range(40)),
            workers=2,
        )
        job_id = first["job_id"]

        # A second browser/start during the active job must attach to the same
        # server-owned queue instead of creating duplicate processing.
        duplicate = manager.start(
            chapter_id,
            page_indices=list(range(40)),
            workers=2,
        )
        assert duplicate["job_id"] == job_id
        assert duplicate["reused"] is True

        # Simulate F5/client disappearance: after this point there are no more
        # client submissions. The background job itself must submit all batches.
        await asyncio.sleep(0.04)
        reconnected = manager.latest_for_chapter(chapter_id)
        assert reconnected is not None
        assert reconnected["job_id"] == job_id
        assert reconnected["status"] in {"running", "completed"}

        terminal = await asyncio.to_thread(_wait_for_terminal, manager, job_id)
        return terminal

    terminal = asyncio.run(scenario())

    assert terminal["status"] == "completed"
    assert terminal["completed"] == 40
    assert terminal["remaining"] == 0
    assert [pages for _, pages, _ in calls] == [
        list(range(16)),
        list(range(16, 32)),
        list(range(32, 40)),
    ]
    assert completed_chapters == [chapter_id]


def test_processing_job_stops_after_failed_batch(monkeypatch, tmp_path):
    import app.page_processing_jobs as jobs_module

    monkeypatch.setattr(jobs_module, "PROCESSED_DIR", tmp_path)
    chapter_id = "cafebabe"
    (tmp_path / chapter_id).mkdir()
    calls = []

    def process_batch(_chapter, pages, _workers):
        calls.append(list(pages))
        if pages[0] == 16:
            raise RuntimeError("synthetic batch failure")

    async def scenario():
        manager = ChapterProcessingJobManager(process_batch, batch_size=16)
        started = manager.start(
            chapter_id,
            page_indices=list(range(40)),
            workers=1,
        )
        return await asyncio.to_thread(
            _wait_for_terminal,
            manager,
            started["job_id"],
        )

    terminal = asyncio.run(scenario())

    assert terminal["status"] == "failed"
    assert terminal["completed"] == 16
    assert calls == [list(range(16)), list(range(16, 32))]
    assert terminal["errors"][0]["error_type"] == "RuntimeError"
