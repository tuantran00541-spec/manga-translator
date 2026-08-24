from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.config import PROCESSED_DIR, RAW_DIR
from app.security import validate_image_size, validate_managed_path
from app.visual_qc.batch_protocol import RegionBatchDecision, RegionBatchIssue
from app.visual_qc.cache import load_region_qc_cache, store_region_qc_cache
from app.visual_qc.jobs import QCWorkItem, VisualQCJobManager
from app.visual_qc.planner import build_chapter_qc_plan, chunk_regions
from app.visual_qc.region_client import DEFAULT_GEMINI_MODEL
from app.visual_qc.regions import QCRegion


def _env_batch_size(name: str, default: int) -> int:
    try:
        return max(1, min(8, int(os.getenv(name, str(default)))))
    except ValueError:
        return default


DEFAULT_GLOBAL_BATCH_SIZE = _env_batch_size("GEMINI_VISUAL_QC_GLOBAL_BATCH_SIZE", 2)
DEFAULT_REGION_BATCH_SIZE = _env_batch_size("GEMINI_VISUAL_QC_REGION_BATCH_SIZE", 4)
DEFAULT_PAIR_BATCH_SIZE = _env_batch_size("GEMINI_VISUAL_QC_PAIR_BATCH_SIZE", 2)
_IMAGES_UNAVAILABLE = "Page images are no longer available for visual QC"


class StaleVisualQCResult(RuntimeError):
    pass


@dataclass(frozen=True)
class _PageSnapshot:
    page_index: int
    source_revision: int
    process_revision: int
    clean_revision: int
    original_path: str
    clean_path: str
    original_file_revision: tuple[int, int, int]
    clean_file_revision: tuple[int, int, int]


def _file_revision(path_value: str | Path) -> tuple[int, int, int]:
    st = Path(path_value).stat()
    return st.st_size, st.st_mtime_ns, st.st_ctime_ns


def _page_revision_tuple(page: dict) -> tuple[int, int, int]:
    return (
        int(page.get("source_revision") or 0),
        int(page.get("process_revision") or 0),
        int(page.get("clean_revision") or 0),
    )


def _decision_to_dict(decision: RegionBatchDecision) -> dict:
    return {
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


def _cached_issue_from_dict(item: object, region: QCRegion) -> RegionBatchIssue | None:
    if not isinstance(item, dict):
        return None
    bbox = item.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        box = tuple(int(v) for v in bbox)
        confidence = float(item.get("confidence", 0.0))
    except (TypeError, ValueError, OverflowError):
        return None
    return RegionBatchIssue(
        page_index=region.page_index,
        region_id=region.region_id,
        issue_type=str(item.get("issue_type") or "unknown"),
        confidence=max(0.0, min(1.0, confidence)),
        bbox=box,
        reason=str(item.get("reason") or "")[:500],
        recommended_action=str(item.get("recommended_action") or "review"),
    )


def _decision_from_dict(raw: dict | None, region: QCRegion) -> RegionBatchDecision | None:
    if not isinstance(raw, dict) or raw.get("region_id") != region.region_id:
        return None
    status = str(raw.get("status") or "")
    if status not in {"pass", "flagged", "ambiguous"}:
        return None
    issues = tuple(
        issue
        for item in (raw.get("issues") or [])
        if (issue := _cached_issue_from_dict(item, region)) is not None
    )
    return RegionBatchDecision(region.page_index, region.region_id, status, issues)


def _needs_pair(decision: RegionBatchDecision) -> bool:
    return decision.status == "ambiguous" or any(
        issue.recommended_action in {"review_original", "deep_qc"}
        for issue in decision.issues
    )


class ChapterQCService:
    def __init__(
        self,
        runner,
        job_manager: VisualQCJobManager | None = None,
        *,
        manifest_loader=None,
        manifest_saver=None,
        manifest_lock=None,
        api_key_provider=None,
        model: str = DEFAULT_GEMINI_MODEL,
        global_batch_size: int = DEFAULT_GLOBAL_BATCH_SIZE,
        region_batch_size: int = DEFAULT_REGION_BATCH_SIZE,
        pair_batch_size: int = DEFAULT_PAIR_BATCH_SIZE,
    ):
        if manifest_loader is None or manifest_saver is None or manifest_lock is None:
            from app.manifest_utils import get_manifest_lock, load_manifest_raw, save_manifest_raw
            manifest_loader = manifest_loader or load_manifest_raw
            manifest_saver = manifest_saver or save_manifest_raw
            manifest_lock = manifest_lock or get_manifest_lock
        if api_key_provider is None:
            from app.secret_store import get_gemini_api_key
            api_key_provider = get_gemini_api_key
        self.runner = runner
        self.job_manager = job_manager or VisualQCJobManager()
        self.manifest_loader = manifest_loader
        self.manifest_saver = manifest_saver
        self.manifest_lock = manifest_lock
        self.api_key_provider = api_key_provider
        self.model = model
        self.global_batch_size = max(1, int(global_batch_size))
        self.region_batch_size = max(1, int(region_batch_size))
        self.pair_batch_size = max(1, int(pair_batch_size))

    def _load_manual_masks(self, manifest: dict, chapter_id: str) -> dict[int, np.ndarray]:
        masks: dict[int, np.ndarray] = {}
        for page_index, page in enumerate(manifest.get("pages") or []):
            path_value = page.get("manual_mask") if isinstance(page, dict) else None
            if not path_value:
                continue
            path = validate_managed_path(path_value, PROCESSED_DIR / chapter_id)
            if not path.is_file():
                continue
            validate_image_size(path)
            try:
                data = np.fromfile(str(path), dtype=np.uint8)
                mask = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    masks[page_index] = mask
            except OSError:
                continue
        return masks

    async def start(self, chapter_id: str, *, concurrency: int = 2):
        with self.manifest_lock(chapter_id):
            manifest = self.manifest_loader(chapter_id)
        api_key = self.api_key_provider()
        if not api_key:
            raise ValueError("Gemini API key is not configured")

        plan = build_chapter_qc_plan(
            manifest,
            manual_masks=self._load_manual_masks(manifest, chapter_id),
        )
        all_regions = tuple(plan.global_regions) + tuple(plan.candidate_regions)
        regions_by_id = {region.region_id: region for region in all_regions}
        planned_revisions = {
            i: _page_revision_tuple(page)
            for i, page in enumerate(manifest.get("pages") or [])
            if isinstance(page, dict)
        }

        normal_regions = [region for region in plan.candidate_regions if not region.requires_deep_qc]
        deep_regions = [region for region in plan.candidate_regions if region.requires_deep_qc]
        items: list[QCWorkItem] = []
        items.extend(chunk_regions(plan.global_regions, batch_size=self.global_batch_size, work_prefix="global", mode="global-clean"))
        items.extend(chunk_regions(normal_regions, batch_size=self.region_batch_size, work_prefix="region", mode="region-clean"))
        items.extend(chunk_regions(deep_regions, batch_size=self.pair_batch_size, work_prefix="deep", mode="region-pair"))

        async def worker(item: QCWorkItem):
            return await self._execute_item(chapter_id, item, regions_by_id, planned_revisions, api_key)

        job = await self.job_manager.start(chapter_id, items, worker, concurrency=concurrency)
        job.skipped_pages = len(plan.skipped_pages)
        return job

    @staticmethod
    def _managed_page_paths(chapter_id: str, page: dict) -> tuple[Path, Path]:
        original_value = str(page.get("original") or "")
        clean_value = str(page.get("clean") or "")
        if not original_value or not clean_value:
            raise StaleVisualQCResult(_IMAGES_UNAVAILABLE)
        original_path = validate_managed_path(original_value, RAW_DIR / chapter_id)
        clean_path = validate_managed_path(clean_value, PROCESSED_DIR / chapter_id)
        if not original_path.is_file() or not clean_path.is_file():
            raise StaleVisualQCResult(_IMAGES_UNAVAILABLE)
        return original_path, clean_path

    def _snapshot_page(
        self,
        chapter_id: str,
        page_index: int,
        page: dict,
        planned_revisions: dict[int, tuple[int, int, int]],
    ) -> _PageSnapshot:
        if _page_revision_tuple(page) != planned_revisions.get(page_index):
            raise StaleVisualQCResult("Page revision changed before visual QC could start")
        original_path, clean_path = self._managed_page_paths(chapter_id, page)
        page["original"] = str(original_path)
        page["clean"] = str(clean_path)
        return _PageSnapshot(
            page_index,
            *_page_revision_tuple(page),
            str(original_path),
            str(clean_path),
            _file_revision(original_path),
            _file_revision(clean_path),
        )

    def _snapshot(self, chapter_id: str, page_indices: tuple[int, ...], planned_revisions: dict[int, tuple[int, int, int]]):
        with self.manifest_lock(chapter_id):
            manifest = self.manifest_loader(chapter_id)
        pages = manifest.get("pages") or []
        snapshots: dict[int, _PageSnapshot] = {}
        for page_index in page_indices:
            if page_index < 0 or page_index >= len(pages):
                raise StaleVisualQCResult("Page changed before visual QC could start")
            snapshots[page_index] = self._snapshot_page(
                chapter_id,
                page_index,
                pages[page_index],
                planned_revisions,
            )
        return manifest, snapshots

    def _assert_snapshot_page_current(
        self,
        chapter_id: str,
        page: dict,
        snapshot: _PageSnapshot,
        *,
        during_commit: bool = False,
    ) -> None:
        expected_revision = (
            snapshot.source_revision,
            snapshot.process_revision,
            snapshot.clean_revision,
        )
        if _page_revision_tuple(page) != expected_revision:
            suffix = "before visual QC cache commit" if during_commit else "while visual QC was running"
            raise StaleVisualQCResult(f"Page revision changed {suffix}")
        original_path, clean_path = self._managed_page_paths(chapter_id, page)
        if str(original_path) != snapshot.original_path or str(clean_path) != snapshot.clean_path:
            suffix = "before visual QC cache commit" if during_commit else "while visual QC was running"
            raise StaleVisualQCResult(f"Page image path changed {suffix}")
        if _file_revision(original_path) != snapshot.original_file_revision:
            suffix = "before visual QC cache commit" if during_commit else "while visual QC was running"
            raise StaleVisualQCResult(f"Original image changed {suffix}")
        if _file_revision(clean_path) != snapshot.clean_file_revision:
            suffix = "before visual QC cache commit" if during_commit else "while visual QC was running"
            raise StaleVisualQCResult(f"Clean image changed {suffix}")

    def _assert_current(self, chapter_id: str, snapshots: dict[int, _PageSnapshot]) -> dict:
        with self.manifest_lock(chapter_id):
            manifest = self.manifest_loader(chapter_id)
            pages = manifest.get("pages") or []
            for page_index, snapshot in snapshots.items():
                if page_index >= len(pages):
                    raise StaleVisualQCResult("Page changed while visual QC was running")
                self._assert_snapshot_page_current(
                    chapter_id,
                    pages[page_index],
                    snapshot,
                )
            return manifest

    def _cached_decision(self, manifest: dict, snapshots: dict[int, _PageSnapshot], region: QCRegion, mode: str):
        page = manifest["pages"][region.page_index]
        snapshot = snapshots[region.page_index]
        raw = load_region_qc_cache(
            page,
            region_id=region.region_id,
            model=self.model,
            mode=mode,
            clean_file_revision=snapshot.clean_file_revision,
            source_file_revision=(snapshot.original_file_revision if mode == "region-pair" else None),
        )
        return _decision_from_dict(raw, region)

    def _partition_cached(
        self,
        manifest: dict,
        snapshots: dict[int, _PageSnapshot],
        regions: list[QCRegion],
        mode: str,
    ) -> tuple[dict[str, RegionBatchDecision], list[str]]:
        decisions: dict[str, RegionBatchDecision] = {}
        fresh_ids: list[str] = []
        for region in regions:
            cached = self._cached_decision(manifest, snapshots, region, mode)
            if cached is None:
                fresh_ids.append(region.region_id)
            else:
                decisions[region.region_id] = cached
        return decisions, fresh_ids

    async def _inspect_ids(
        self,
        chapter_id: str,
        item: QCWorkItem,
        region_ids: list[str],
        mode: str,
        manifest: dict,
        regions_by_id: dict[str, QCRegion],
        snapshots: dict[int, _PageSnapshot],
        api_key: str,
    ) -> list[RegionBatchDecision]:
        if not region_ids:
            return []
        subitem = QCWorkItem(item.work_id, tuple(region_ids), item.page_indices, mode)
        decisions = await asyncio.to_thread(
            self.runner.inspect,
            subitem,
            manifest,
            regions_by_id,
            api_key,
        )
        self._assert_current(chapter_id, snapshots)
        allowed = set(region_ids)
        return [decision for decision in decisions if decision.region_id in allowed]

    @staticmethod
    def _merge_fresh(
        decisions: dict[str, RegionBatchDecision],
        pending_cache: list[tuple[str, RegionBatchDecision]],
        mode: str,
        fresh: list[RegionBatchDecision],
    ) -> None:
        for decision in fresh:
            decisions[decision.region_id] = decision
            pending_cache.append((mode, decision))

    def _pair_fallback_ids(
        self,
        manifest: dict,
        snapshots: dict[int, _PageSnapshot],
        regions: list[QCRegion],
        decisions: dict[str, RegionBatchDecision],
    ) -> list[str]:
        fallback_ids: list[str] = []
        for region in regions:
            decision = decisions.get(region.region_id)
            if decision is None or not _needs_pair(decision):
                continue
            pair_cached = self._cached_decision(
                manifest,
                snapshots,
                region,
                "region-pair",
            )
            if pair_cached is None:
                fallback_ids.append(region.region_id)
            else:
                decisions[region.region_id] = pair_cached
        return fallback_ids

    def _commit_cache(
        self,
        chapter_id: str,
        snapshots: dict[int, _PageSnapshot],
        pending_cache: list[tuple[str, RegionBatchDecision]],
    ) -> None:
        if not pending_cache:
            return
        with self.manifest_lock(chapter_id):
            latest = self.manifest_loader(chapter_id)
            pages = latest.get("pages") or []
            for page_index, snapshot in snapshots.items():
                if page_index >= len(pages):
                    raise StaleVisualQCResult("Page changed before visual QC cache commit")
                self._assert_snapshot_page_current(
                    chapter_id,
                    pages[page_index],
                    snapshot,
                    during_commit=True,
                )
            for mode, decision in pending_cache:
                page = pages[decision.page_index]
                snapshot = snapshots[decision.page_index]
                store_region_qc_cache(
                    page,
                    region_id=decision.region_id,
                    model=self.model,
                    mode=mode,
                    clean_file_revision=snapshot.clean_file_revision,
                    source_file_revision=(snapshot.original_file_revision if mode == "region-pair" else None),
                    result=_decision_to_dict(decision),
                )
            self.manifest_saver(chapter_id, latest)

    async def _execute_item(self, chapter_id: str, item: QCWorkItem, regions_by_id: dict[str, QCRegion], planned_revisions: dict[int, tuple[int, int, int]], api_key: str) -> list[RegionBatchDecision]:
        manifest, snapshots = self._snapshot(
            chapter_id,
            item.page_indices,
            planned_revisions,
        )
        ordered_regions = [regions_by_id[region_id] for region_id in item.region_ids]
        decisions, fresh_ids = self._partition_cached(
            manifest,
            snapshots,
            ordered_regions,
            item.mode,
        )
        pending_cache: list[tuple[str, RegionBatchDecision]] = []

        fresh = await self._inspect_ids(
            chapter_id,
            item,
            fresh_ids,
            item.mode,
            manifest,
            regions_by_id,
            snapshots,
            api_key,
        )
        self._merge_fresh(decisions, pending_cache, item.mode, fresh)

        if item.mode in {"global-clean", "region-clean"}:
            fallback_ids = self._pair_fallback_ids(
                manifest,
                snapshots,
                ordered_regions,
                decisions,
            )
            pair_fresh = await self._inspect_ids(
                chapter_id,
                item,
                fallback_ids,
                "region-pair",
                manifest,
                regions_by_id,
                snapshots,
                api_key,
            )
            self._merge_fresh(
                decisions,
                pending_cache,
                "region-pair",
                pair_fresh,
            )

        self._commit_cache(chapter_id, snapshots, pending_cache)
        return [
            decisions.get(region.region_id)
            or RegionBatchDecision(region.page_index, region.region_id, "ambiguous", ())
            for region in ordered_regions
        ]
