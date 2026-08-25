from __future__ import annotations

import asyncio
from unittest.mock import patch

import numpy as np
import pytest

from app.visual_qc.contact_sheet import build_contact_sheet
from app.visual_qc.jobs import QCWorkItem, VisualQCJobManager
from app.visual_qc.regions import QCRegion


def test_job_manager_rejects_duplicate_active_chapter_and_global_active_overflow():
    async def scenario():
        gate = asyncio.Event()

        async def worker(_item):
            await gate.wait()
            return []

        manager = VisualQCJobManager(max_jobs=8, max_active_jobs=1)
        first = await manager.start(
            "chapter-a",
            [QCWorkItem("a", ("r1",), (0,))],
            worker,
            concurrency=1,
        )

        with pytest.raises(RuntimeError, match="already running"):
            await manager.start(
                "chapter-a",
                [QCWorkItem("dup", ("r2",), (0,))],
                worker,
                concurrency=1,
            )

        with pytest.raises(RuntimeError, match="Too many active"):
            await manager.start(
                "chapter-b",
                [QCWorkItem("b", ("r3",), (0,))],
                worker,
                concurrency=1,
            )

        manager.cancel(first.job_id)
        gate.set()
        await manager.wait(first.job_id)

    asyncio.run(scenario())


def test_job_manager_public_error_does_not_expose_exception_detail():
    async def scenario():
        secret = "super-secret-gemini-key"

        async def worker(_item):
            raise RuntimeError(f"transport failure leaked {secret}")

        manager = VisualQCJobManager()
        job = await manager.start(
            "chapter-a",
            [QCWorkItem("bad", ("r1",), (0,))],
            worker,
            concurrency=1,
        )
        await manager.wait(job.job_id)
        snapshot = manager.snapshot(job.job_id)
        assert snapshot["failed"] == 1
        assert len(snapshot["errors"]) == 1
        assert secret not in str(snapshot["errors"])
        assert snapshot["errors"][0]["detail"] == "Visual QC batch failed"
        assert snapshot["errors"][0]["error_type"] == "RuntimeError"

    asyncio.run(scenario())


def test_contact_sheet_downscales_before_large_sheet_allocation():
    region = QCRegion(
        page_index=0,
        region_id="P0001-R01",
        bbox=(0, 0, 1600, 1200),
        source_box_ids=("box_a",),
        source_kinds=("box",),
        area_ratio=0.5,
        requires_deep_qc=True,
    )
    crop = np.full((1200, 1600, 3), 255, dtype=np.uint8)

    real_full = np.full
    requested_shapes = []

    def guarded_full(shape, *args, **kwargs):
        requested_shapes.append(tuple(shape))
        assert max(shape[:2]) <= 512
        return real_full(shape, *args, **kwargs)

    with patch("app.visual_qc.contact_sheet.np.full", side_effect=guarded_full):
        sheet = build_contact_sheet([(region, crop)], max_side=512)

    assert requested_shapes
    assert max(sheet.image.shape[:2]) <= 512
    assert 0 < sheet.scale < 1
