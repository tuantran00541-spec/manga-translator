from __future__ import annotations

import asyncio
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable

from filelock import FileLock, Timeout

from app.config import PROCESSED_DIR
from app.logging_config import logger

MAX_RETAINED_PROCESS_JOBS = 32
MAX_PROCESS_JOB_ERRORS = 10
_ACTIVE_STATUSES = {"pending", "running"}
_TERMINAL_STATUSES = {"completed", "failed"}


@dataclass
class ChapterProcessingJob:
    job_id: str
    chapter_id: str
    page_indices: list[int]
    workers: int
    status: str = "pending"
    completed: int = 0
    current_batch: list[int] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class ChapterProcessingJobManager:
    def __init__(
        self,
        process_plan: Callable[
            [str, list[int], int, Callable[[int], None]],
            object,
        ],
        *,
        on_completed: Callable[[str], None] | None = None,
    ) -> None:
        self._process_plan = process_plan
        self._on_completed = on_completed
        self._jobs: dict[str, ChapterProcessingJob] = {}
        self._latest_by_chapter: dict[str, str] = {}
        self._active_by_chapter: dict[str, str] = {}
        self._tasks: set[asyncio.Task] = set()
        self._lock = threading.RLock()

    def start(
        self,
        chapter_id: str,
        *,
        page_indices: list[int],
        workers: int,
    ) -> dict:
        planned = list(dict.fromkeys(int(index) for index in page_indices))
        if not planned:
            raise ValueError("page_indices cannot be empty")

        with self._lock:
            active_id = self._active_by_chapter.get(chapter_id)
            if active_id:
                active = self._jobs.get(active_id)
                if active is not None and active.status in _ACTIVE_STATUSES:
                    return self._snapshot_locked(active, reused=True)
                self._active_by_chapter.pop(chapter_id, None)

            self._prune_locked()
            job = ChapterProcessingJob(
                job_id=uuid.uuid4().hex,
                chapter_id=chapter_id,
                page_indices=planned,
                workers=max(1, min(8, int(workers))),
            )
            self._jobs[job.job_id] = job
            self._latest_by_chapter[chapter_id] = job.job_id
            self._active_by_chapter[chapter_id] = job.job_id

        task = asyncio.get_running_loop().create_task(
            asyncio.to_thread(self._run_sync, job)
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return self.snapshot(job.job_id)

    def snapshot(self, job_id: str) -> dict:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            return self._snapshot_locked(job)

    def latest_for_chapter(self, chapter_id: str) -> dict | None:
        with self._lock:
            job_id = self._active_by_chapter.get(chapter_id)
            if not job_id:
                job_id = self._latest_by_chapter.get(chapter_id)
            job = self._jobs.get(job_id) if job_id else None
            return self._snapshot_locked(job) if job is not None else None

    def _run_sync(self, job: ChapterProcessingJob) -> None:
        processing_lock = FileLock(
            PROCESSED_DIR / job.chapter_id / "processing.lock"
        )
        try:
            processing_lock.acquire(timeout=0)
        except Timeout:
            self._fail_job(
                job,
                "processing-lock-busy",
                "This chapter is already being processed.",
            )
            return

        try:
            self._set_running(job)
            total = len(job.page_indices)
            with self._lock:
                job.current_batch = list(job.page_indices)
                job.updated_at = time.time()

            completed_indices: set[int] = set()

            def on_page_done(page_index: int) -> None:
                with self._lock:
                    index = int(page_index)
                    if index in completed_indices:
                        return
                    completed_indices.add(index)
                    job.completed = min(len(completed_indices), total)
                    job.updated_at = time.time()

            self._process_plan(
                job.chapter_id,
                list(job.page_indices),
                job.workers,
                on_page_done,
            )
            with self._lock:
                job.completed = total
                job.current_batch = []
                job.updated_at = time.time()

            if self._on_completed is not None:
                self._on_completed(job.chapter_id)
            self._complete_job(job)
        except Exception as exc:
            logger.opt(exception=True).error(
                "Chapter {} background processing job {} failed: {}",
                job.chapter_id,
                job.job_id,
                exc,
            )
            self._fail_job(job, type(exc).__name__, str(exc) or "Processing failed")
        finally:
            try:
                processing_lock.release()
            except Exception:
                pass

    def _set_running(self, job: ChapterProcessingJob) -> None:
        with self._lock:
            job.status = "running"
            job.updated_at = time.time()

    def _complete_job(self, job: ChapterProcessingJob) -> None:
        with self._lock:
            job.status = "completed"
            job.current_batch = []
            job.updated_at = time.time()
            if self._active_by_chapter.get(job.chapter_id) == job.job_id:
                self._active_by_chapter.pop(job.chapter_id, None)

    def _fail_job(self, job: ChapterProcessingJob, error_type: str, message: str) -> None:
        with self._lock:
            job.status = "failed"
            job.current_batch = []
            job.updated_at = time.time()
            if len(job.errors) < MAX_PROCESS_JOB_ERRORS:
                job.errors.append(
                    {
                        "error_type": str(error_type),
                        "message": str(message),
                    }
                )
            if self._active_by_chapter.get(job.chapter_id) == job.job_id:
                self._active_by_chapter.pop(job.chapter_id, None)

    @staticmethod
    def _snapshot_locked(
        job: ChapterProcessingJob,
        *,
        reused: bool = False,
    ) -> dict:
        total = len(job.page_indices)
        return {
            "job_id": job.job_id,
            "chapter_id": job.chapter_id,
            "status": job.status,
            "active": job.status in _ACTIVE_STATUSES,
            "reused": bool(reused),
            "workers": job.workers,
            "total": total,
            "completed": min(job.completed, total),
            "remaining": max(0, total - job.completed),
            "current_batch": list(job.current_batch),
            "errors": list(job.errors),
            "created_at": job.created_at,
            "updated_at": job.updated_at,
        }

    def _prune_locked(self) -> None:
        if len(self._jobs) < MAX_RETAINED_PROCESS_JOBS:
            return
        terminal = [
            job_id
            for job_id, job in self._jobs.items()
            if job.status in _TERMINAL_STATUSES
        ]
        while len(self._jobs) >= MAX_RETAINED_PROCESS_JOBS and terminal:
            job_id = terminal.pop(0)
            removed = self._jobs.pop(job_id, None)
            if removed is None:
                continue
            if self._latest_by_chapter.get(removed.chapter_id) == job_id:
                self._latest_by_chapter.pop(removed.chapter_id, None)
