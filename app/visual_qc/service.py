from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

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


def _file_revision(path_value: str) -> tuple[int, int, int]:
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


def _decision_from_dict(raw: dict | None, region: QCRegion) -> RegionBatchDecision | None:
    if not isinstance(raw, dict) or raw.get("region_id") != region.region_id:
        return None
    status = str(raw.get("status") or "")
    if status not in {"pass", "flagged", "ambiguous"}:
        return None
    issues: list[RegionBatchIssue] = []
    for item in raw.get("issues") or []:
        if not isinstance(item, dict):
            continue
        bbox = item.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            box = tuple(int(v) for v in bbox)
            confidence = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            continue
        issues.append(
            RegionBatchIssue(
                page_index=region.page_index,
                region_id=region.region_id,
                issue_type=str(item.get("issue_type") or "unknown"),
                confidence=max(0.0, min(1.0, confidence)),
                bbox=box,
                reason=str(item.get("reason") or "")[:500],
                recommended_action=str(item.get("recommended_action") or "review"),
            )
        )
    return RegionBatchDecision(region.page_index, region.region_id, status, tuple(issues))


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

    def _load_manual_masks(self, manifest: dict) -> dict[int, np.ndarray]:
        masks: dict[int, np.ndarray] = {}
        for page_index, page in enumerate(manifest.get("pages") or []):
            path_value = page.get("manual_mask") if isinstance(page, dict) else None
            if not path_value:
                continue
            path = Path(path_value)
            if not path.is_file():
                continue
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

        plan = build_chapter_qc_plan(manifest, manual_masks=self._load_manual_masks(manifest))
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

    def _snapshot(self, chapter_id: str, page_indices: tuple[int, ...], planned_revisions: dict[int, tuple[int, int, int]]):
        with self.manifest_lock(chapter_id):
            manifest = self.manifest_loader(chapter_id)
        pages = manifest.get("pages") or []
        snapshots: dict[int, _PageSnapshot] = {}
        for page_index in page_indices:
            if page_index < 0 or page_index >= len(pages):
                raise StaleVisualQCResult("Page changed before visual QC could start")
            page = pages[page_index]
            if _page_revision_tuple(page) != planned_revisions.get(page_index):
                raise StaleVisualQCResult("Page revision changed before visual QC could start")
            original_path = str(page.get("original") or "")
            clean_path = str(page.get("clean") or "")
            if not original_path or not clean_path:
                raise StaleVisualQCResult("Page images are no longer available for visual QC")
            snapshots[page_index] = _PageSnapshot(
                page_index,
                *_page_revision_tuple(page),
                original_path,
                clean_path,
                _file_revision(original_path),
                _file_revision(clean_path),
            )
        return manifest, snapshots

    def _assert_current(self, chapter_id: str, snapshots: dict[int, _PageSnapshot]) -> dict:
        with self.manifest_lock(chapter_id):
            manifest = self.manifest_loader(chapter_id)
            pages = manifest.get("pages") or []
            for page_index, snapshot in snapshots.items():
                if page_index >= len(pages):
                    raise StaleVisualQCResult("Page changed while visual QC was running")
                page = pages[page_index]
                if _page_revision_tuple(page) != (snapshot.source_revision, snapshot.process_revision, snapshot.clean_revision):
                    raise StaleVisualQCResult("Page revision changed while visual QC was running")
                if str(page.get("original") or "") != snapshot.original_path or str(page.get("clean") or "") != snapshot.clean_path:
                    raise StaleVisualQCResult("Page image path changed while visual QC was running")
                if _file_revision(snapshot.original_path) != snapshot.original_file_revision or _file_revision(snapshot.clean_path) != snapshot.clean_file_revision:
                    raise StaleVisualQCResult("Page image changed while visual QC was running")
            return manifest

    def _cached_decision(self, manifest: dict, snapshots: dict[int, _PageSnapshot], region: QCRegion, mode: str):
        page = manifest["pages"][region.page_index]
        raw = load_region_qc_cache(
            page,
            region_id=region.region_id,
            model=self.model,
            mode=mode,
            clean_file_revision=snapshots[region.page_index].clean_file_revision,
        )
        return _decision_from_dict(raw, region)

    async def _execute_item(self, chapter_id: str, item: QCWorkItem, regions_by_id: dict[str, QCRegion], planned_revisions: dict[int, tuple[int, int, int]], api_key: str) -> list[RegionBatchDecision]:
        manifest, snapshots = self._snapshot(chapter_id, item.page_indices, planned_revisions)
        ordered_regions = [regions_by_id[region_id] for region_id in item.region_ids]
        decisions: dict[str, RegionBatchDecision] = {}
        fresh_ids: list[str] = []
        pending_cache: list[tuple[str, RegionBatchDecision]] = []

        for region in ordered_regions:
            cached = self._cached_decision(manifest, snapshots, region, item.mode)
            if cached is None:
                fresh_ids.append(region.region_id)
            else:
                decisions[region.region_id] = cached

        if fresh_ids:
            subitem = QCWorkItem(item.work_id, tuple(fresh_ids), item.page_indices, item.mode)
            fresh = await asyncio.to_thread(self.runner.inspect, subitem, manifest, regions_by_id, api_key)
            self._assert_current(chapter_id, snapshots)
            for decision in fresh:
                decisions[decision.region_id] = decision
                pending_cache.append((item.mode, decision))

        if item.mode in {"global-clean", "region-clean"}:
            fallback_ids = []
            for region in ordered_regions:
                decision = decisions.get(region.region_id)
                if decision is None:
                    continue
                needs_pair = decision.status == "ambiguous" or any(
                    issue.recommended_action in {"review_original", "deep_qc"}
                    for issue in decision.issues
                )
                if not needs_pair:
                    continue
                pair_cached = self._cached_decision(manifest, snapshots, region, "region-pair")
                if pair_cached is not None:
                    decisions[region.region_id] = pair_cached
                else:
                    fallback_ids.append(region.region_id)

            if fallback_ids:
                pair_item = QCWorkItem(item.work_id + "-pair", tuple(fallback_ids), item.page_indices, "region-pair")
                pair_decisions = await asyncio.to_thread(self.runner.inspect, pair_item, manifest, regions_by_id, api_key)
                self._assert_current(chapter_id, snapshots)
                for decision in pair_decisions:
                    decisions[decision.region_id] = decision
                    pending_cache.append(("region-pair", decision))

        if pending_cache:
            with self.manifest_lock(chapter_id):
                latest = self.manifest_loader(chapter_id)
                pages = latest.get("pages") or []
                for page_index, snapshot in snapshots.items():
                    if page_index >= len(pages) or _page_revision_tuple(pages[page_index]) != (snapshot.source_revision, snapshot.process_revision, snapshot.clean_revision):
                        raise StaleVisualQCResult("Page changed before visual QC cache commit")
                    if _file_revision(snapshot.clean_path) != snapshot.clean_file_revision:
                        raise StaleVisualQCResult("Clean image changed before visual QC cache commit")
                for mode, decision in pending_cache:
                    page = pages[decision.page_index]
                    store_region_qc_cache(
                        page,
                        region_id=decision.region_id,
                        model=self.model,
                        mode=mode,
                        clean_file_revision=snapshots[decision.page_index].clean_file_revision,
                        result=_decision_to_dict(decision),
                    )
                self.manifest_saver(chapter_id, latest)

        return [
            decisions.get(region.region_id)
            or RegionBatchDecision(region.page_index, region.region_id, "ambiguous", ())
            for region in ordered_regions
        ]
