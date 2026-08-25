from __future__ import annotations

import asyncio
import threading
import uuid
from dataclasses import dataclass, field

from app.logging_config import logger
from app.ocr.service import OCRCancelled, OCRResultStale, OCRService

MAX_OCR_ITEMS = 5000
MAX_RETAINED_JOBS = 32
MAX_ACTIVE_JOBS = 2
MAX_JOB_ERRORS = 50
_TERMINAL_STATUSES = {"completed", "cancelled"}
_ACTIVE_STATUSES = {"pending", "running"}


@dataclass
class ChapterOCRJob:
    job_id: str
    chapter_id: str
    lang: str
    concurrency: int
    force: bool
    items: list[tuple[int, str]]
    status: str = "pending"
    completed: int = 0
    recognized: int = 0
    empty: int = 0
    cached: int = 0
    stale: int = 0
    failed: int = 0
    errors: list[dict] = field(default_factory=list)
    retry_items: list[tuple[int, str]] = field(default_factory=list)
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)


class ChapterOCRJobManager:
    def __init__(self, service: OCRService):
        self.service = service
        self._jobs: dict[str, ChapterOCRJob] = {}
        self._active_chapters: set[str] = set()
        self._tasks: set[asyncio.Task] = set()
        self._lock = threading.Lock()

    def start(
        self,
        chapter_id: str,
        *,
        lang: str,
        concurrency: int = 1,
        force: bool = False,
        items: list[tuple[int, str]] | None = None,
    ) -> dict:
        concurrency = max(1, min(2, int(concurrency)))
        planned = list(items) if items is not None else self.service.plan_chapter(chapter_id)
        if len(planned) > MAX_OCR_ITEMS:
            raise ValueError(f"Too many OCR regions: {len(planned)} > {MAX_OCR_ITEMS}")

        job = self._register_job(
            chapter_id,
            lang=lang,
            concurrency=concurrency,
            force=force,
            planned=planned,
        )
        if not planned:
            with self._lock:
                job.status = "completed"
                self._active_chapters.discard(chapter_id)
            return self.snapshot(job.job_id)

        task = asyncio.create_task(self._run(job))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return self.snapshot(job.job_id)

    def _register_job(
        self,
        chapter_id: str,
        *,
        lang: str,
        concurrency: int,
        force: bool,
        planned: list[tuple[int, str]],
    ) -> ChapterOCRJob:
        with self._lock:
            active_count = sum(
                1 for job in self._jobs.values() if job.status in _ACTIVE_STATUSES
            )
            if chapter_id in self._active_chapters:
                raise RuntimeError("OCR is already running for this chapter")
            if active_count >= MAX_ACTIVE_JOBS:
                raise RuntimeError("Too many active OCR jobs")
            self._prune_locked()
            job = ChapterOCRJob(
                job_id=uuid.uuid4().hex,
                chapter_id=chapter_id,
                lang=lang,
                concurrency=concurrency,
                force=bool(force),
                items=planned,
            )
            self._jobs[job.job_id] = job
            self._active_chapters.add(chapter_id)
            return job

    def snapshot(self, job_id: str) -> dict:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            return {
                "job_id": job.job_id,
                "chapter_id": job.chapter_id,
                "lang": job.lang,
                "concurrency": job.concurrency,
                "force": job.force,
                "status": job.status,
                "total": len(job.items),
                "completed": job.completed,
                "recognized": job.recognized,
                "empty": job.empty,
                "cached": job.cached,
                "stale": job.stale,
                "failed": job.failed,
                "errors": list(job.errors),
            }

    def cancel(self, job_id: str) -> dict:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            job.cancel_event.set()
        return self.snapshot(job_id)

    def retry(self, job_id: str) -> dict:
        with self._lock:
            previous = self._jobs.get(job_id)
            if previous is None:
                raise KeyError(job_id)
            if previous.status not in _TERMINAL_STATUSES:
                raise RuntimeError("OCR job is still running")
            retry_items = list(dict.fromkeys(previous.retry_items))
            chapter_id = previous.chapter_id
            lang = previous.lang
            concurrency = previous.concurrency
            force = previous.force
        if not retry_items:
            raise ValueError("OCR job has no failed or stale regions to retry")
        return self.start(
            chapter_id,
            lang=lang,
            concurrency=concurrency,
            force=force,
            items=retry_items,
        )

    async def _run(self, job: ChapterOCRJob) -> None:
        with self._lock:
            job.status = "running"

        queue: asyncio.Queue[tuple[int, str]] = asyncio.Queue()
        for item in job.items:
            queue.put_nowait(item)

        async def worker() -> None:
            while not job.cancel_event.is_set():
                try:
                    item = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    await self._process_item(job, item)
                finally:
                    queue.task_done()

        try:
            workers = [asyncio.create_task(worker()) for _ in range(job.concurrency)]
            await asyncio.gather(*workers)
        finally:
            with self._lock:
                job.status = "cancelled" if job.cancel_event.is_set() else "completed"
                self._active_chapters.discard(job.chapter_id)

    async def _process_item(self, job: ChapterOCRJob, item: tuple[int, str]) -> None:
        page_index, box_id = item
        try:
            result = await asyncio.to_thread(
                self.service.inspect_box_id,
                job.chapter_id,
                page_index,
                box_id,
                job.lang,
                force=job.force,
                cancel_event=job.cancel_event,
            )
        except OCRCancelled:
            return
        except OCRResultStale:
            self._record_retry(job, item, stale=True)
        except Exception as exc:
            self._record_failure(job, item, exc)
        else:
            self._record_success(job, result)

    def _record_retry(
        self, job: ChapterOCRJob, item: tuple[int, str], *, stale: bool
    ) -> None:
        with self._lock:
            if stale:
                job.stale += 1
            job.retry_items.append(item)

    def _record_failure(
        self, job: ChapterOCRJob, item: tuple[int, str], exc: Exception
    ) -> None:
        page_index, box_id = item
        logger.opt(exception=True).error(
            "Chapter {} page {} box {} OCR job failed: {}",
            job.chapter_id,
            page_index,
            box_id,
            exc,
        )
        with self._lock:
            job.failed += 1
            job.retry_items.append(item)
            if len(job.errors) < MAX_JOB_ERRORS:
                job.errors.append(
                    {
                        "page_index": page_index,
                        "box_id": box_id,
                        "error_type": type(exc).__name__,
                    }
                )

    def _record_success(self, job: ChapterOCRJob, result: dict) -> None:
        with self._lock:
            job.completed += 1
            if result.get("cached"):
                job.cached += 1
            if str(result.get("text") or "").strip():
                job.recognized += 1
            else:
                job.empty += 1

    def _prune_locked(self) -> None:
        if len(self._jobs) < MAX_RETAINED_JOBS:
            return
        terminal = [
            job_id
            for job_id, job in self._jobs.items()
            if job.status in _TERMINAL_STATUSES
        ]
        while len(self._jobs) >= MAX_RETAINED_JOBS and terminal:
            self._jobs.pop(terminal.pop(0), None)
