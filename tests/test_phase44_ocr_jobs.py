import asyncio
import threading
import time
import unittest

from app.ocr.jobs import ChapterOCRJobManager
from app.ocr.service import OCRCancelled


class FakeOCRService:
    def __init__(self, items):
        self.items = list(items)
        self.calls = []
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()
        self.started = threading.Event()
        self.release = threading.Event()
        self.block = False
        self.fail_once = set()

    def plan_chapter(self, chapter_id):
        return list(self.items)

    def inspect_box_id(
        self,
        chapter_id,
        page_index,
        box_id,
        lang,
        *,
        force=False,
        cancel_event=None,
    ):
        with self.lock:
            self.calls.append((page_index, box_id))
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            self.started.set()
            if self.block:
                self.release.wait(timeout=2)
            else:
                time.sleep(0.025)
            if cancel_event is not None and cancel_event.is_set():
                raise OCRCancelled("cancelled")
            key = (page_index, box_id)
            if key in self.fail_once:
                self.fail_once.remove(key)
                raise RuntimeError("synthetic OCR failure")
            return {
                "text": f"text-{box_id}",
                "cached": False,
                "committed": True,
            }
        finally:
            with self.lock:
                self.active -= 1


async def wait_terminal(manager, job_id, timeout=3):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        snapshot = manager.snapshot(job_id)
        if snapshot["status"] in {"completed", "cancelled"}:
            return snapshot
        await asyncio.sleep(0.01)
    raise AssertionError("OCR job did not reach a terminal state")


class ChapterOCRJobTests(unittest.IsolatedAsyncioTestCase):
    async def test_concurrency_is_bounded_to_two_workers(self):
        service = FakeOCRService([(0, "a"), (0, "b"), (1, "c"), (1, "d")])
        manager = ChapterOCRJobManager(service)
        started = manager.start("abc12345", lang="en", concurrency=2)
        done = await wait_terminal(manager, started["job_id"])

        self.assertEqual(done["status"], "completed")
        self.assertEqual(done["completed"], 4)
        self.assertLessEqual(service.max_active, 2)

    async def test_cancel_stops_scheduling_and_discards_inflight_result(self):
        service = FakeOCRService([(0, "a"), (0, "b"), (0, "c")])
        service.block = True
        manager = ChapterOCRJobManager(service)
        started = manager.start("abc12345", lang="en", concurrency=1)
        await asyncio.to_thread(service.started.wait, 1)

        manager.cancel(started["job_id"])
        service.release.set()
        done = await wait_terminal(manager, started["job_id"])

        self.assertEqual(done["status"], "cancelled")
        self.assertEqual(done["completed"], 0)
        self.assertEqual(service.calls, [(0, "a")])

    async def test_retry_runs_only_failed_regions(self):
        service = FakeOCRService([(0, "a"), (0, "b")])
        service.fail_once.add((0, "b"))
        manager = ChapterOCRJobManager(service)
        started = manager.start("abc12345", lang="en", concurrency=1)
        first = await wait_terminal(manager, started["job_id"])
        self.assertEqual(first["failed"], 1)

        retried = manager.retry(started["job_id"])
        second = await wait_terminal(manager, retried["job_id"])
        self.assertEqual(second["total"], 1)
        self.assertEqual(second["completed"], 1)
        self.assertEqual(service.calls.count((0, "b")), 2)

    async def test_same_chapter_cannot_start_twice(self):
        service = FakeOCRService([(0, "a")])
        service.block = True
        manager = ChapterOCRJobManager(service)
        started = manager.start("abc12345", lang="en", concurrency=1)
        await asyncio.to_thread(service.started.wait, 1)

        with self.assertRaisesRegex(RuntimeError, "already running"):
            manager.start("abc12345", lang="en", concurrency=1)

        manager.cancel(started["job_id"])
        service.release.set()
        await wait_terminal(manager, started["job_id"])

    async def test_manager_retains_background_task_until_completion(self):
        service = FakeOCRService([(0, "a")])
        service.block = True
        manager = ChapterOCRJobManager(service)
        started = manager.start("abc12345", lang="en", concurrency=1)
        await asyncio.to_thread(service.started.wait, 1)

        self.assertEqual(len(manager._tasks), 1)
        service.release.set()
        await wait_terminal(manager, started["job_id"])
        await asyncio.sleep(0)
        self.assertEqual(len(manager._tasks), 0)


if __name__ == "__main__":
    unittest.main()
