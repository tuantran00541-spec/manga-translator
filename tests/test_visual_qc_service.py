from __future__ import annotations

import asyncio
import copy
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from app.visual_qc.batch_protocol import RegionBatchDecision
from app.visual_qc.jobs import QCWorkItem, VisualQCJobManager
from app.visual_qc.regions import QCRegion
from app.visual_qc.service import ChapterQCService, StaleVisualQCResult


@contextmanager
def _lock(_chapter_id: str):
    yield


class _ManifestStore:
    def __init__(self, manifest: dict):
        self.value = copy.deepcopy(manifest)
        self.save_count = 0

    def load(self, _chapter_id: str) -> dict:
        return copy.deepcopy(self.value)

    def save(self, _chapter_id: str, manifest: dict) -> None:
        self.value = copy.deepcopy(manifest)
        self.save_count += 1


class _ScriptedRunner:
    def __init__(self, callback=None):
        self.calls: list[tuple[str, tuple[str, ...]]] = []
        self.callback = callback

    def inspect(self, item, manifest, regions_by_id, api_key):
        self.calls.append((item.mode, item.region_ids))
        if self.callback is not None:
            return self.callback(item, manifest, regions_by_id, api_key)
        status = "ambiguous" if item.mode in {"global-clean", "region-clean"} else "pass"
        return [
            RegionBatchDecision(
                page_index=regions_by_id[region_id].page_index,
                region_id=region_id,
                status=status,
                issues=(),
            )
            for region_id in item.region_ids
        ]


def _write_page_files(root: Path) -> tuple[Path, Path, Path, Path]:
    raw_root = root / "raw"
    processed_root = root / "processed"
    chapter_raw = raw_root / "deadbeef"
    chapter_processed = processed_root / "deadbeef"
    chapter_raw.mkdir(parents=True)
    chapter_processed.mkdir(parents=True)
    original = chapter_raw / "000.png"
    clean = chapter_processed / "clean_000.png"
    original.write_bytes(b"original-v1")
    clean.write_bytes(b"clean-v1")
    return raw_root, processed_root, original, clean


def _manifest(original: Path, clean: Path, boxes=None) -> dict:
    return {
        "chapter_id": "deadbeef",
        "pages": [{
            "original": str(original),
            "clean": str(clean),
            "width": 1000,
            "height": 1400,
            "boxes": boxes or [],
            "skipped": False,
            "source_revision": 1,
            "process_revision": 1,
            "clean_revision": 1,
        }],
    }


def _service(store: _ManifestStore, runner: _ScriptedRunner, manager=None) -> ChapterQCService:
    return ChapterQCService(
        runner,
        manager or VisualQCJobManager(),
        manifest_loader=store.load,
        manifest_saver=store.save,
        manifest_lock=_lock,
        api_key_provider=lambda: "test-key",
        model="test-model",
        global_batch_size=2,
        region_batch_size=4,
        pair_batch_size=2,
    )


def test_page_without_detector_boxes_still_runs_global_clean_qc():
    async def scenario():
        with tempfile.TemporaryDirectory() as tmp:
            raw_root, processed_root, original, clean = _write_page_files(Path(tmp))
            store = _ManifestStore(_manifest(original, clean))
            runner = _ScriptedRunner(
                lambda item, _manifest_value, regions_by_id, _api_key: [
                    RegionBatchDecision(
                        regions_by_id[region_id].page_index,
                        region_id,
                        "pass",
                        (),
                    )
                    for region_id in item.region_ids
                ]
            )
            manager = VisualQCJobManager()
            service = _service(store, runner, manager)
            with patch("app.visual_qc.service.RAW_DIR", raw_root), patch(
                "app.visual_qc.service.PROCESSED_DIR", processed_root
            ):
                job = await service.start("deadbeef", concurrency=1)
                await manager.wait(job.job_id)
                snapshot = manager.snapshot(job.job_id)

            assert snapshot["status"] == "completed"
            assert snapshot["total_regions"] == 1
            assert snapshot["passed"] == 1
            assert runner.calls == [("global-clean", ("P0001-GLOBAL",))]

    asyncio.run(scenario())


def test_ambiguous_clean_result_escalates_to_pair_and_pair_cache_is_reused():
    async def scenario():
        with tempfile.TemporaryDirectory() as tmp:
            raw_root, processed_root, original, clean = _write_page_files(Path(tmp))
            store = _ManifestStore(_manifest(original, clean))
            runner = _ScriptedRunner()
            service = _service(store, runner)
            region = QCRegion(0, "P0001-R01", (100, 100, 300, 300), ("box_a",), ("box",), 0.03, False)
            item = QCWorkItem("region-0001", (region.region_id,), (0,), "region-clean")
            planned = {0: (1, 1, 1)}

            with patch("app.visual_qc.service.RAW_DIR", raw_root), patch(
                "app.visual_qc.service.PROCESSED_DIR", processed_root
            ):
                first = await service._execute_item("deadbeef", item, {region.region_id: region}, planned, "test-key")
                calls_after_first = list(runner.calls)
                second = await service._execute_item("deadbeef", item, {region.region_id: region}, planned, "test-key")

            assert [decision.status for decision in first] == ["pass"]
            assert [decision.status for decision in second] == ["pass"]
            assert calls_after_first == [
                ("region-clean", (region.region_id,)),
                ("region-pair", (region.region_id,)),
            ]
            assert runner.calls == calls_after_first
            assert store.save_count == 1
            assert "region-clean" in store.value["pages"][0]["visual_qc_cache"]
            assert "region-pair" in store.value["pages"][0]["visual_qc_cache"]

    asyncio.run(scenario())


def test_changed_clean_file_discards_inflight_result_without_cache_commit():
    async def scenario():
        with tempfile.TemporaryDirectory() as tmp:
            raw_root, processed_root, original, clean = _write_page_files(Path(tmp))
            store = _ManifestStore(_manifest(original, clean))

            def mutate_clean(item, _manifest_value, regions_by_id, _api_key):
                clean.write_bytes(b"clean-v2-longer")
                return [
                    RegionBatchDecision(
                        regions_by_id[region_id].page_index,
                        region_id,
                        "pass",
                        (),
                    )
                    for region_id in item.region_ids
                ]

            runner = _ScriptedRunner(mutate_clean)
            service = _service(store, runner)
            region = QCRegion(0, "P0001-R01", (100, 100, 300, 300), ("box_a",), ("box",), 0.03, False)
            item = QCWorkItem("region-0001", (region.region_id,), (0,), "region-clean")

            with patch("app.visual_qc.service.RAW_DIR", raw_root), patch(
                "app.visual_qc.service.PROCESSED_DIR", processed_root
            ):
                try:
                    await service._execute_item(
                        "deadbeef", item, {region.region_id: region}, {0: (1, 1, 1)}, "test-key"
                    )
                except StaleVisualQCResult:
                    pass
                else:
                    raise AssertionError("Expected StaleVisualQCResult")

            assert store.save_count == 0
            assert "visual_qc_cache" not in store.value["pages"][0]

    asyncio.run(scenario())
