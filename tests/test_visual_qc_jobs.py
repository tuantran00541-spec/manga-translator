import asyncio

from app.visual_qc.batch_protocol import RegionBatchDecision
from app.visual_qc.jobs import QCWorkItem, VisualQCJobManager


def test_job_manager_bounds_concurrency_and_counts_region_results():
    async def scenario():
        active = 0
        max_active = 0
        lock = asyncio.Lock()
        async def worker(item):
            nonlocal active, max_active
            async with lock:
                active += 1
                max_active = max(max_active, active)
            await asyncio.sleep(0.01)
            async with lock:
                active -= 1
            return [RegionBatchDecision(i, rid, "pass", ()) for i, rid in enumerate(item.region_ids)]
        manager = VisualQCJobManager()
        items = [QCWorkItem(f"b{i}", (f"P{i:04d}-R01", f"P{i:04d}-R02"), (i,)) for i in range(6)]
        job = await manager.start("chapter1", items, worker, concurrency=2)
        await manager.wait(job.job_id)
        snapshot = manager.snapshot(job.job_id)
        assert max_active <= 2
        assert snapshot["status"] == "completed"
        assert snapshot["total_regions"] == 12
        assert snapshot["completed_regions"] == 12
        assert snapshot["passed"] == 12
        assert snapshot["flagged"] == 0
        assert snapshot["failed"] == 0
    asyncio.run(scenario())


def test_job_manager_isolates_batch_failures_without_automatic_retry():
    async def scenario():
        calls = {}
        async def worker(item):
            calls[item.work_id] = calls.get(item.work_id, 0) + 1
            if item.work_id == "bad":
                raise RuntimeError("upstream 503")
            return [RegionBatchDecision(0, rid, "flagged", ()) for rid in item.region_ids]
        manager = VisualQCJobManager()
        items = [QCWorkItem("ok1", ("r1",), (0,)), QCWorkItem("bad", ("r2", "r3"), (1,)), QCWorkItem("ok2", ("r4",), (2,))]
        job = await manager.start("chapter1", items, worker, concurrency=2)
        await manager.wait(job.job_id)
        snapshot = manager.snapshot(job.job_id)
        assert snapshot["status"] == "completed"
        assert snapshot["flagged"] == 2
        assert snapshot["failed"] == 2
        assert calls == {"ok1": 1, "bad": 1, "ok2": 1}
        assert len(snapshot["errors"]) == 1
        assert snapshot["errors"][0]["work_id"] == "bad"
    asyncio.run(scenario())


def test_job_cancel_stops_scheduling_new_batches_and_discards_late_results():
    async def scenario():
        started = []
        gate = asyncio.Event()
        async def worker(item):
            started.append(item.work_id)
            await gate.wait()
            return [RegionBatchDecision(0, rid, "pass", ()) for rid in item.region_ids]
        manager = VisualQCJobManager()
        items = [QCWorkItem(f"b{i}", (f"r{i}",), (i,)) for i in range(8)]
        job = await manager.start("chapter1", items, worker, concurrency=2)
        await asyncio.sleep(0.01)
        assert len(started) == 2
        assert manager.cancel(job.job_id) is True
        gate.set()
        await manager.wait(job.job_id)
        snapshot = manager.snapshot(job.job_id)
        assert snapshot["status"] == "cancelled"
        assert len(started) == 2
        assert snapshot["completed_regions"] == 0
        assert snapshot["passed"] == 0
    asyncio.run(scenario())
