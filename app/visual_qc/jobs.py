from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Awaitable, Callable
import uuid

from app.parameters import VISUAL_QC_JOB_CONCURRENCY, VISUAL_QC_JOB_CONCURRENCY_LIMIT
from app.visual_qc.batch_protocol import RegionBatchDecision


@dataclass(frozen=True)
class QCWorkItem:
    work_id: str
    region_ids: tuple[str, ...]
    page_indices: tuple[int, ...]
    mode: str = "region-clean"


@dataclass
class VisualQCJob:
    job_id: str
    chapter_id: str
    total_regions: int
    concurrency: int = 1
    status: str = "pending"
    completed_regions: int = 0
    passed: int = 0
    flagged: int = 0
    failed: int = 0
    skipped_pages: int = 0
    errors: list[dict] = field(default_factory=list)
    results: list[dict] = field(default_factory=list)
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    task: asyncio.Task | None = field(default=None, repr=False)


Worker = Callable[[QCWorkItem], Awaitable[list[RegionBatchDecision]]]


class VisualQCJobManager:
    def __init__(self, *, max_jobs: int = 32, max_active_jobs: int = 2):
        self.max_jobs = max(1, int(max_jobs))
        self.max_active_jobs = max(1, min(int(max_active_jobs), self.max_jobs))
        self._jobs: dict[str, VisualQCJob] = {}

    async def start(
        self,
        chapter_id: str,
        items: list[QCWorkItem],
        worker: Worker,
        *,
        concurrency: int = VISUAL_QC_JOB_CONCURRENCY,
    ) -> VisualQCJob:
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        concurrency = min(int(concurrency), VISUAL_QC_JOB_CONCURRENCY_LIMIT)
        chapter_id = str(chapter_id)
        self._prune_completed()
        if len(self._jobs) >= self.max_jobs:
            raise RuntimeError("Too many visual QC jobs are retained")

        active = [job for job in self._jobs.values() if job.status in {"pending", "running"}]
        if any(job.chapter_id == chapter_id for job in active):
            raise RuntimeError("Visual QC is already running for this chapter")
        if len(active) >= self.max_active_jobs:
            raise RuntimeError("Too many active visual QC jobs")

        job = VisualQCJob(
            job_id=uuid.uuid4().hex,
            chapter_id=chapter_id,
            total_regions=sum(len(item.region_ids) for item in items),
            concurrency=concurrency,
        )
        self._jobs[job.job_id] = job
        job.task = asyncio.create_task(self._run(job, list(items), worker, concurrency))
        return job

    @staticmethod
    def _result_payload(decision: RegionBatchDecision) -> dict:
        return {
            "page_index": decision.page_index,
            "region_id": decision.region_id,
            "status": decision.status,
            "issues": [
                {
                    "issue_type": issue.issue_type,
                    "confidence": issue.confidence,
                    "bbox": list(issue.bbox),
                    "reason": issue.reason,
                    "recommended_action": issue.recommended_action,
                }
                for issue in decision.issues
            ],
        }

    @staticmethod
    def _record_batch_failure(job: VisualQCJob, item: QCWorkItem, exc: Exception) -> None:
        count = len(item.region_ids)
        job.completed_regions += count
        job.failed += count
        job.errors.append({
            "work_id": item.work_id,
            "region_ids": list(item.region_ids),
            "detail": "Visual QC batch failed",
            "error_type": type(exc).__name__[:120],
        })

    def _record_decisions(
        self,
        job: VisualQCJob,
        item: QCWorkItem,
        decisions: list[RegionBatchDecision],
    ) -> None:
        by_id = {
            decision.region_id: decision
            for decision in decisions
            if isinstance(decision, RegionBatchDecision)
        }
        for region_id in item.region_ids:
            decision = by_id.get(region_id)
            if decision is None:
                job.failed += 1
                continue
            job.results.append(self._result_payload(decision))
            if decision.status == "pass":
                job.passed += 1
            elif decision.status in {"flagged", "ambiguous"}:
                job.flagged += 1
            else:
                job.failed += 1
        job.completed_regions += len(item.region_ids)

    async def _run_worker(
        self,
        job: VisualQCJob,
        queue: asyncio.Queue[QCWorkItem],
        worker: Worker,
    ) -> None:
        while not job.cancel_event.is_set():
            try:
                item = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                if job.cancel_event.is_set():
                    return
                try:
                    decisions = await worker(item)
                except Exception as exc:
                    if not job.cancel_event.is_set():
                        self._record_batch_failure(job, item, exc)
                    continue
                if not job.cancel_event.is_set():
                    self._record_decisions(job, item, decisions)
            finally:
                queue.task_done()

    async def _run(self, job: VisualQCJob, items: list[QCWorkItem], worker: Worker, concurrency: int) -> None:
        job.status = "running"
        queue: asyncio.Queue[QCWorkItem] = asyncio.Queue()
        for item in items:
            queue.put_nowait(item)

        worker_count = min(concurrency, max(1, len(items))) if items else 0
        tasks = [
            asyncio.create_task(self._run_worker(job, queue, worker))
            for _ in range(worker_count)
        ]
        if tasks:
            await asyncio.gather(*tasks)
        job.status = "cancelled" if job.cancel_event.is_set() else "completed"

    def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job is None or job.status in {"completed", "cancelled"}:
            return False
        job.cancel_event.set()
        return True

    async def wait(self, job_id: str) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.task is not None:
            await job.task

    def snapshot(self, job_id: str) -> dict:
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return {
            "job_id": job.job_id,
            "chapter_id": job.chapter_id,
            "status": job.status,
            "concurrency": job.concurrency,
            "total_regions": job.total_regions,
            "completed_regions": job.completed_regions,
            "pending_regions": max(0, job.total_regions - job.completed_regions),
            "passed": job.passed,
            "flagged": job.flagged,
            "failed": job.failed,
            "skipped_pages": job.skipped_pages,
            "cancel_requested": job.cancel_event.is_set(),
            "errors": [dict(error) for error in job.errors],
            "results": [dict(result) for result in job.results],
        }

    def _prune_completed(self) -> None:
        removable = [
            job_id for job_id, job in self._jobs.items()
            if job.status in {"completed", "cancelled"}
        ]
        while len(self._jobs) >= self.max_jobs and removable:
            self._jobs.pop(removable.pop(0), None)
