"""Sequential multi-chapter queue runner built on top of AIModeJobManager."""

from __future__ import annotations

import asyncio
import copy
import time
import uuid
from dataclasses import dataclass, field, replace
from typing import Awaitable, Callable

from fastapi import HTTPException

from app.ai_mode.job import AIModeJobManager, AIModeSettings, _detail
from app.logging_config import logger

MAX_QUEUE_URLS = 500
MAX_RETAINED_QUEUES = 8
_FATAL_STATUS_CODES = frozenset({401, 402, 403})


@dataclass
class AIModeQueueItem:
    index: int
    url: str
    status: str = "pending"
    job_id: str | None = None
    chapter_id: str | None = None
    cost_usd: float | None = None
    error: str | None = None


@dataclass
class AIModeQueue:
    queue_id: str
    settings: AIModeSettings
    items: list[AIModeQueueItem]
    continue_on_error: bool = True
    status: str = "pending"
    current_index: int = -1
    cancel_requested: bool = False
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    task: asyncio.Task | None = field(default=None, repr=False)


ItemStarter = Callable[[AIModeSettings], Awaitable[dict]]


class AIModeQueueManager:
    """Coordinate sequential A.I mode execution across multiple chapter URLs."""

    def __init__(self, job_manager: AIModeJobManager):
        self._job_manager = job_manager
        self._queues: dict[str, AIModeQueue] = {}

    def active_queue(self) -> AIModeQueue | None:
        return next((q for q in self._queues.values() if q.status in {"pending", "running"}), None)

    def active_snapshot(self) -> dict | None:
        queue = self.active_queue()
        return self.snapshot(queue.queue_id) if queue else None

    def start(
        self,
        urls: list[str],
        base_settings: AIModeSettings,
        *,
        item_starter: ItemStarter,
        continue_on_error: bool = True,
    ) -> dict:
        if not urls or len(urls) > MAX_QUEUE_URLS:
            raise ValueError(f"Queue accepts between 1 and {MAX_QUEUE_URLS} chapter URLs")
        if self.active_queue() is not None or self._job_manager.active_job() is not None:
            raise RuntimeError("Một tác vụ hoặc hàng đợi Chế độ AI khác đang chạy.")
        self._prune()
        items = [AIModeQueueItem(index=i, url=url) for i, url in enumerate(urls)]
        queue = AIModeQueue(
            queue_id=uuid.uuid4().hex,
            settings=base_settings,
            items=items,
            continue_on_error=continue_on_error,
        )
        self._queues[queue.queue_id] = queue
        queue.task = asyncio.get_running_loop().create_task(self._run(queue, item_starter))
        return self.snapshot(queue.queue_id)

    async def _run(self, queue: AIModeQueue, item_starter: ItemStarter) -> None:
        queue.status = "running"
        queue.updated_at = time.time()
        stopped_early = False

        try:
            for item in queue.items:
                if queue.cancel_requested:
                    item.status = "cancelled"
                    continue
                if stopped_early:
                    item.status = "skipped"
                    continue

                queue.current_index = item.index
                item.status = "running"
                queue.updated_at = time.time()
                item_settings = replace(queue.settings, url=item.url, chapter_id=None)

                fatal_error = False
                try:
                    started = await item_starter(item_settings)
                    job_id = str(started["job_id"])
                    item.job_id = job_id
                    if queue.cancel_requested:
                        self._safe_cancel_job(job_id)
                    snap = await self._wait_for_job(job_id)
                    item.status = str(snap.get("status") or "failed")
                    item.chapter_id = snap.get("chapter_id")
                    item.cost_usd = snap.get("cost_usd")
                    item.error = snap.get("error")
                except Exception as exc:
                    logger.warning("Queue {} item {} failed to start: {}", queue.queue_id, item.index, type(exc).__name__)
                    item.status = "failed"
                    item.error = _detail(exc)[:500]
                    fatal_error = isinstance(exc, HTTPException) and exc.status_code in _FATAL_STATUS_CODES

                queue.updated_at = time.time()
                if item.status == "cancelled" or queue.cancel_requested:
                    queue.cancel_requested = True
                    continue
                if item.status == "failed" and (fatal_error or not queue.continue_on_error):
                    queue.error = item.error or "Tác vụ dừng do lỗi ở một chương."
                    stopped_early = True

            self._finalize_queue_status(queue)
        except asyncio.CancelledError:
            queue.cancel_requested = True
            for item in queue.items:
                if item.status in {"pending", "running"}:
                    item.status = "cancelled"
            queue.status = "cancelled"
            raise
        except Exception as exc:
            logger.opt(exception=True).error("Queue {} crashed: {}", queue.queue_id, type(exc).__name__)
            queue.status = "failed"
            queue.error = _detail(exc)[:500]
        finally:
            queue.current_index = -1
            queue.updated_at = time.time()

    def _safe_cancel_job(self, job_id: str) -> None:
        try:
            self._job_manager.cancel(job_id)
        except KeyError:
            pass

    async def _wait_for_job(self, job_id: str) -> dict:
        job = self._job_manager._jobs.get(job_id)
        if job is not None and job.task is not None:
            try:
                await job.task
            except asyncio.CancelledError:
                self._safe_cancel_job(job_id)
                if not job.task.cancelled():
                    raise
            except Exception:
                pass
        else:
            while True:
                snap = self._job_manager.snapshot(job_id)
                if snap.get("status") not in {"pending", "running"}:
                    return snap
                await asyncio.sleep(0.05)
        return self._job_manager.snapshot(job_id)

    def _finalize_queue_status(self, queue: AIModeQueue) -> None:
        if queue.cancel_requested:
            for item in queue.items:
                if item.status == "pending":
                    item.status = "cancelled"
            queue.status = "cancelled"
            return

        done_count = sum(1 for item in queue.items if item.status in {"completed", "completed_with_blockers"})
        failed = [item for item in queue.items if item.status == "failed"]
        if failed and done_count == 0:
            queue.status = "failed"
            queue.error = queue.error or failed[0].error
        elif failed:
            queue.status = "completed_with_errors"
        else:
            queue.status = "completed"

    def cancel(self, queue_id: str) -> dict:
        queue = self._get(queue_id)
        if queue.status in {"pending", "running"}:
            queue.cancel_requested = True
            if 0 <= queue.current_index < len(queue.items):
                job_id = queue.items[queue.current_index].job_id
                if job_id:
                    self._safe_cancel_job(job_id)
            queue.updated_at = time.time()
        return self.snapshot(queue_id)

    def snapshot(self, queue_id: str) -> dict:
        queue = self._get(queue_id)
        active_job = None
        if 0 <= queue.current_index < len(queue.items):
            current = queue.items[queue.current_index]
            if current.job_id:
                try:
                    active_job = self._job_manager.snapshot(current.job_id)
                except KeyError:
                    active_job = None

        costs = [item.cost_usd for item in queue.items if item.cost_usd is not None]
        if active_job and active_job.get("cost_usd") is not None and (
            0 <= queue.current_index < len(queue.items) and queue.items[queue.current_index].cost_usd is None
        ):
            costs.append(active_job["cost_usd"])

        return {
            "queue_id": queue.queue_id,
            "status": queue.status,
            "current_index": queue.current_index,
            "total": len(queue.items),
            "completed_count": sum(1 for i in queue.items if i.status in {"completed", "completed_with_blockers"}),
            "failed_count": sum(1 for i in queue.items if i.status == "failed"),
            "continue_on_error": queue.continue_on_error,
            "cancel_requested": queue.cancel_requested,
            "total_cost_usd": round(sum(costs), 6) if costs else None,
            "error": queue.error,
            "provider": queue.settings.provider,
            "model": queue.settings.model,
            "target_lang": queue.settings.target_lang,
            "items": [
                {
                    "index": item.index,
                    "url": item.url,
                    "status": item.status,
                    "job_id": item.job_id,
                    "chapter_id": item.chapter_id,
                    "cost_usd": None if item.cost_usd is None else round(item.cost_usd, 6),
                    "error": item.error,
                }
                for item in queue.items
            ],
            "active_job": copy.deepcopy(active_job),
            "created_at": queue.created_at,
            "updated_at": queue.updated_at,
        }

    def _get(self, queue_id: str) -> AIModeQueue:
        queue = self._queues.get(queue_id)
        if queue is None:
            raise KeyError(queue_id)
        return queue

    def _prune(self) -> None:
        finished = [qid for qid, q in self._queues.items() if q.status not in {"pending", "running"}]
        while len(self._queues) >= MAX_RETAINED_QUEUES and finished:
            self._queues.pop(finished.pop(0), None)
