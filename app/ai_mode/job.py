"""A.I mode job: chapter URL -> scan -> clean -> QC/repaint -> translate -> repair -> render -> ZIP.

Every stage reuses the same endpoint functions the editor calls, so locking,
revision checks and manifest bookkeeping are identical to manual work. The
chapter stays a normal chapter afterwards: it can be opened, fixed by hand
and exported again.
"""
from __future__ import annotations

import asyncio
import copy
import json
import re
import time
import uuid
from dataclasses import dataclass, field

from fastapi import HTTPException

from app.ai_mode.page_scan import scan_slices
from app.config import OUTPUT_DIR, RAW_DIR
from app.image_io import read_image
from app.logging_config import logger
from app.manifest_utils import get_manifest_lock, load_manifest_raw
from app.parameters import VISUAL_QC_JOB_CONCURRENCY_LIMIT
from app.security import validate_managed_path

STAGES: tuple[tuple[str, str], ...] = (
    ("download", "Tải chương"),
    ("scan", "AI quét credit, lát trống và logo"),
    ("clean", "Clean ảnh"),
    ("qc", "AI kiểm tra và repaint"),
    ("translate", "Dịch và chọn font"),
    ("repair", "AI tự sửa lỗi"),
    ("render", "Render"),
    ("export", "Đóng gói zip"),
)
SCAN_BATCH_SIZE = 4
SCAN_CONCURRENCY = 3
# More "credit" slices than this is a misread of the chapter, not credits.
CREDIT_MAX_SHARE = 0.25
CREDIT_MAX_ABSOLUTE = 3
# More than this share of textless slices means the scan misread the chapter.
TEXTLESS_MAX_SHARE = 0.5
REPAINT_ISSUE_TYPES = frozenset({"residual_text", "partial_text", "partial_erase", "smear", "inpaint_artifact"})
REPAINT_MIN_CONFIDENCE = 0.5
# QC often marks clear leftover text "review" instead of "repaint"; act on the confident ones.
REPAINT_REVIEW_MIN_CONFIDENCE = 0.7
REPAINT_PAD_PX = 6
TRANSLATE_CONCURRENCY = 3
QC_CONCURRENCY = VISUAL_QC_JOB_CONCURRENCY_LIMIT
# Repair: objects per retry request, and retry batches without progress before restoring the original.
RETRY_BATCH = 4
RETRY_ROUNDS = 3
KEEP_PAD_PX = 4
# Objects per slice given back their original pixels when their translation cannot be lettered.
RENDER_RESTORE_ATTEMPTS = 3
RENDER_FAILED_OBJECT = re.compile(r"\(vùng ([\w-]+)\)")
POLL_SECONDS = 1.0
MAX_RETAINED_JOBS = 16
MAX_REPORT_ITEMS = 50


class AIModeCancelled(Exception):
    pass


class AIModeFailed(Exception):
    pass


@dataclass(frozen=True)
class AIModeSettings:
    url: str
    provider: str
    model: str
    target_lang: str = "vi"
    budget_usd: float = 0.30
    workers: int = 2
    story_notes: str = ""


@dataclass
class AIModeJob:
    job_id: str
    settings: AIModeSettings
    status: str = "pending"
    stage: str | None = None
    stages: dict[str, dict] = field(default_factory=dict)
    chapter_id: str | None = None
    error: str | None = None
    cost_usd: float | None = 0.0
    report: dict = field(default_factory=dict)
    archive_path: str | None = None
    cancel_requested: bool = False
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    task: asyncio.Task | None = field(default=None, repr=False)


def _detail(exc: Exception) -> str:
    if isinstance(exc, HTTPException):
        return str(exc.detail)
    return str(exc) or type(exc).__name__


def _append(items: list, value) -> None:
    if len(items) < MAX_REPORT_ITEMS:
        items.append(value)


class AIModeRunner:
    """Owns the stage logic. Kept separate from the manager so tests can stub stages."""

    def __init__(self, job: AIModeJob, provider, api_key: str):
        self.job = job
        self.provider = provider
        self.api_key = api_key
        self.settings = job.settings
        self.report = job.report
        self.report.update({
            "credit_pages": [], "credit_rejected": [], "logo_regions": 0, "scan_errors": [],
            "repainted_regions": 0, "repaint_pages": [], "qc_flagged": 0, "qc_errors": [],
            "translated": 0, "unreadable": 0, "review_flags": 0, "translate_errors": [], "render_errors": [],
            "editorial_blockers": 0, "blocker_samples": [], "source_lang": None,
            "textless_pages": [], "textless_rejected": [], "kept_regions": 0, "missed_added": 0,
            "retried_pages": [], "restored_regions": 0, "review_list": [],
        })
        # page index -> what translation asked the repair stage to do
        self._repair: dict[int, dict] = {}
        self._memory = None
        self._slice_total: int | None = None

    # -- bookkeeping ---------------------------------------------------------
    def _check_cancel(self) -> None:
        if self.job.cancel_requested:
            raise AIModeCancelled()

    def _progress(self, done: int, total: int, detail: str | None = None) -> None:
        state = self.job.stages[self.job.stage]
        state["done"], state["total"] = int(done), int(total)
        if detail is not None:
            state["detail"] = detail
        self.job.updated_at = time.time()

    def _add_cost(self, cost: float | None) -> None:
        if cost is None:
            self.job.cost_usd = None
        elif self.job.cost_usd is not None:
            self.job.cost_usd += float(cost)

    def _remaining_budget(self) -> float | None:
        if not self.provider.tracks_cost or self.job.cost_usd is None:
            return None
        return max(0.0, self.settings.budget_usd - self.job.cost_usd)

    def _manifest(self) -> dict:
        with get_manifest_lock(self.job.chapter_id):
            return load_manifest_raw(self.job.chapter_id)

    def _active_pages(self) -> list[int]:
        return [
            index for index, page in enumerate(self._manifest().get("pages", []))
            if isinstance(page, dict) and not page.get("skipped")
        ]

    # -- stages --------------------------------------------------------------
    async def download(self) -> None:
        from app.routers.chapters import create_chapter
        from app.schemas import ChapterRequest

        self._progress(0, 1, "Đang tải ảnh từ nguồn")
        manifest = await asyncio.to_thread(
            create_chapter, ChapterRequest(url=self.settings.url, workers=self.settings.workers),
        )
        self.job.chapter_id = str(manifest["chapter_id"])
        self._progress(1, 1, f"{len(manifest.get('pages') or [])} lát")

    async def scan(self) -> None:
        from app.dependencies import pipeline
        from app.routers.chapters import _set_page_preserve_regions
        from app.schemas import RegionModel

        chapter_id = self.job.chapter_id
        pages = self._manifest().get("pages", [])
        active = [index for index, page in enumerate(pages) if not page.get("skipped")]
        batches = [active[i:i + SCAN_BATCH_SIZE] for i in range(0, len(active), SCAN_BATCH_SIZE)]
        scans = []
        async def scan_images(batch: list[int]) -> None:
            images = [
                (index, read_image(validate_managed_path(pages[index]["original"], RAW_DIR / chapter_id)))
                for index in batch
            ]
            found, cost = await asyncio.to_thread(
                scan_slices, self.provider, self.settings.model, self.api_key, images,
            )
            self._add_cost(cost)
            scans.extend(found)

        gate = asyncio.Semaphore(SCAN_CONCURRENCY)
        finished = 0

        async def scan_one(index: int) -> None:
            nonlocal finished
            async with gate:
                if self.job.cancel_requested:
                    return
                try:
                    await scan_images([index])
                except (RuntimeError, ValueError, OSError) as exc:
                    _append(self.report["scan_errors"], {"pages": [index], "error": _detail(exc)[:300]})
            finished += 1
            self._progress(finished, len(active))

        async def scan_batch(batch: list[int]) -> bool:
            nonlocal finished
            async with gate:
                if self.job.cancel_requested:
                    return True
                try:
                    await scan_images(batch)
                except (RuntimeError, ValueError, OSError) as exc:
                    if len(batch) > 1:
                        return False
                    _append(self.report["scan_errors"], {"pages": batch, "error": _detail(exc)[:300]})
            finished += len(batch)
            self._progress(finished, len(active))
            return True

        # Probe with one batch; if it fails, scan slice by slice.
        self._progress(0, len(active))
        if batches and not await scan_batch(batches[0]):
            await asyncio.gather(*(scan_one(index) for index in active))
        elif batches:
            results = await asyncio.gather(*(scan_batch(batch) for batch in batches[1:]))
            rejected = [index for batch, ok in zip(batches[1:], results) if not ok for index in batch]
            await asyncio.gather(*(scan_one(index) for index in rejected))
        self._check_cancel()

        credits = sorted(scan.page_index for scan in scans if scan.is_credit)
        limit = max(CREDIT_MAX_ABSOLUTE, int(CREDIT_MAX_SHARE * len(active)))
        if len(credits) > limit:
            self.report["credit_rejected"] = credits
            credits = []
        if credits:
            await asyncio.to_thread(pipeline.mark_skipped, chapter_id, credits, True)
        self.report["credit_pages"] = credits

        # Textless slices are exported as they are; skipping them saves cleanup and translation.
        textless = sorted(scan.page_index for scan in scans if scan.no_text and scan.page_index not in credits)
        if len(textless) > TEXTLESS_MAX_SHARE * len(active):
            self.report["textless_rejected"] = textless
            textless = []
        if textless:
            await asyncio.to_thread(pipeline.mark_skipped, chapter_id, textless, True)
        self.report["textless_pages"] = textless

        logos = 0
        for scan in scans:
            if not scan.logos or scan.page_index in credits:
                continue
            page = self._manifest()["pages"][scan.page_index]
            regions = [RegionModel(**region) for region in page.get("preserve_regions") or []]
            regions += [RegionModel(x1=x1, y1=y1, x2=x2, y2=y2) for x1, y1, x2, y2 in scan.logos]
            await asyncio.to_thread(_set_page_preserve_regions, chapter_id, scan.page_index, regions)
            logos += len(scan.logos)
        self.report["logo_regions"] = logos
        self._progress(len(active), len(active),
                       f"{len(credits)} lát credit, {len(textless)} lát không chữ, {logos} logo")

    async def clean(self) -> None:
        from app.routers.chapters import chapter_processing_jobs

        indices = self._active_pages()
        if not indices:
            self._progress(0, 0, "Không có lát nào cần clean")
            return
        snapshot = chapter_processing_jobs.start(
            self.job.chapter_id, page_indices=indices, workers=self.settings.workers,
        )
        job_id = snapshot["job_id"]
        while True:
            snapshot = chapter_processing_jobs.snapshot(job_id)
            self._progress(snapshot["completed"], snapshot["total"])
            if snapshot["status"] in {"completed", "failed"}:
                break
            await asyncio.sleep(POLL_SECONDS)
        if snapshot["status"] == "failed":
            errors = "; ".join(str(item.get("message")) for item in snapshot.get("errors") or [])
            raise AIModeFailed(f"Clean thất bại: {errors or 'lỗi không rõ'}")
        self._check_cancel()

    async def qc(self) -> None:
        from app.routers.editor import repaint_regions
        from app.routers.visual_qc import chapter_qc_jobs, chapter_visual_qc_status, start_chapter_visual_qc
        from app.schemas import RegionModel, RepaintRegionsRequest
        from app.visual_qc.schemas import VisualQCChapterRequest

        if not self._active_pages():
            self._progress(0, 0, "Không có lát nào")
            return
        remaining = self._remaining_budget()
        qc_budget = 0.15 if remaining is None else max(0.005, min(0.15, 0.25 * remaining))
        try:
            snapshot = await start_chapter_visual_qc(VisualQCChapterRequest(
                chapter_id=self.job.chapter_id, provider=self.provider.id,
                model=self.settings.model, budget_usd=qc_budget, concurrency=QC_CONCURRENCY,
            ))
        except HTTPException as exc:
            _append(self.report["qc_errors"], _detail(exc)[:300])
            self._progress(0, 0, "Bỏ qua: không chạy được AI kiểm tra")
            return
        qc_job_id = snapshot["job_id"]
        while snapshot["status"] in {"pending", "running"}:
            if self.job.cancel_requested:
                chapter_qc_jobs.cancel(qc_job_id)
            self._progress(snapshot["completed_regions"], snapshot["total_regions"], "AI đang kiểm tra")
            await asyncio.sleep(POLL_SECONDS)
            snapshot = chapter_visual_qc_status(qc_job_id)
        self._check_cancel()
        usage = snapshot.get("usage") or {}
        if self.provider.tracks_cost:
            self._add_cost(float(usage.get("estimated_cost_usd") or 0.0))
        self.report["qc_flagged"] = int(snapshot.get("flagged") or 0)
        if snapshot.get("failed"):
            _append(self.report["qc_errors"], f"{snapshot['failed']} vùng AI không kiểm tra được")

        manifest = self._manifest()
        by_page: dict[int, list] = {}
        for result in snapshot.get("results") or []:
            for issue in result.get("issues") or []:
                action, confidence = issue.get("recommended_action"), float(issue.get("confidence") or 0.0)
                if issue.get("issue_type") not in REPAINT_ISSUE_TYPES or not (
                    (action == "repaint" and confidence >= REPAINT_MIN_CONFIDENCE)
                    or (action == "review" and confidence >= REPAINT_REVIEW_MIN_CONFIDENCE)
                ):
                    continue
                page_index = int(result["page_index"])
                page = manifest["pages"][page_index]
                width, height = int(page.get("width") or 0), int(page.get("height") or 0)
                x1, y1, x2, y2 = (int(v) for v in issue["bbox"])
                x1, y1 = max(0, x1 - REPAINT_PAD_PX), max(0, y1 - REPAINT_PAD_PX)
                x2 = min(width, x2 + REPAINT_PAD_PX) if width else x2 + REPAINT_PAD_PX
                y2 = min(height, y2 + REPAINT_PAD_PX) if height else y2 + REPAINT_PAD_PX
                if x2 > x1 and y2 > y1:
                    by_page.setdefault(page_index, []).append(RegionModel(x1=x1, y1=y1, x2=x2, y2=y2))

        done = 0
        for page_index, regions in sorted(by_page.items()):
            self._check_cancel()
            self._progress(done, len(by_page), "Đang repaint")
            try:
                await repaint_regions(RepaintRegionsRequest(
                    chapter_id=self.job.chapter_id, page_index=page_index, regions=regions, mode="standard",
                ))
            except HTTPException as exc:
                _append(self.report["qc_errors"], f"Lát {page_index + 1}: {_detail(exc)[:200]}")
                continue
            self.report["repainted_regions"] += len(regions)
            _append(self.report["repaint_pages"], page_index)
            done += 1
        self._progress(len(by_page), len(by_page), f"Repaint {self.report['repainted_regions']} vùng")

    async def translate(self) -> None:
        from app.routers.ocr import detect_chapter_language
        from app.routers.translation import TranslateVisionPageRequest, translate_page_in_context
        from app.translation.context import ChapterMemory

        try:
            language = await detect_chapter_language(self.job.chapter_id)
            source_lang = language.get("source_lang") or "auto"
        except HTTPException:
            source_lang = "auto"
        self.report["source_lang"] = source_lang
        indices = self._active_pages()
        memory = self._memory = ChapterMemory(self.settings.story_notes)
        slice_total = self._slice_total = len(self._manifest().get("pages", []))
        # Admit slices in reading order so each sees the memory of earlier ones.
        gate = asyncio.Semaphore(TRANSLATE_CONCURRENCY)
        finished = 0
        out_of_budget = False

        async def translate_one(page_index: int) -> None:
            nonlocal finished, out_of_budget
            async with gate:
                if self.job.cancel_requested or out_of_budget:
                    return
                remaining = self._remaining_budget()
                if remaining is not None and remaining < 0.001:
                    out_of_budget = True
                    return
                try:
                    data = await translate_page_in_context(TranslateVisionPageRequest(
                        chapter_id=self.job.chapter_id, page_index=page_index,
                        source_lang=source_lang, target_lang=self.settings.target_lang,
                        budget_usd=0.25 if remaining is None else max(0.001, min(0.25, remaining)),
                        provider=self.provider.id, model=self.settings.model,
                    ), memory=memory, slice_total=slice_total)
                except HTTPException as exc:
                    _append(self.report["translate_errors"], f"Lát {page_index + 1}: {_detail(exc)[:200]}")
                    self._repair.setdefault(page_index, {})["failed"] = True
                    return
                finally:
                    finished += 1
                    self._progress(finished, len(indices))
                run = data.get("translation_run") or {}
                self._note_repair(page_index, run)
                self.report["translated"] += int(run.get("translated") or 0)
                self.report["unreadable"] += int(run.get("unreadable") or 0)
                self.report["review_flags"] += int(run.get("review") or 0)
                self._add_cost(run.get("estimated_cost_usd") if self.provider.tracks_cost else None)
                if run.get("render_error"):
                    _append(self.report["render_errors"], f"Lát {page_index + 1}: {run['render_error']}")

        self._progress(0, len(indices))
        await asyncio.gather(*(translate_one(page_index) for page_index in indices))
        self._check_cancel()
        if out_of_budget:
            _append(self.report["translate_errors"], "Hết ngân sách, các lát còn lại chưa dịch")
        sheet = memory.snapshot()
        self.report["characters"] = sheet["characters"][:MAX_REPORT_ITEMS]
        self.report["address"] = sheet["address"][:MAX_REPORT_ITEMS]
        self._progress(len(indices), len(indices), f"Dịch {self.report['translated']} vùng")

    def _note_repair(self, page_index: int, run: dict) -> None:
        keep, missed, missing = run.get("keep_regions") or [], run.get("missed_boxes") or [], run.get("missing_ids") or []
        if keep or missed or missing or run.get("remaining"):
            todo = self._repair.setdefault(page_index, {})
            todo.setdefault("keep", []).extend(keep)
            todo.setdefault("missed", []).extend(missed)
            todo.setdefault("missing", []).extend(missing)
            if run.get("remaining"):
                todo["failed"] = True

    async def _translate_retry(self, page_index: int, source_lang: str, object_ids: list[str] | None) -> dict:
        from app.routers.translation import TranslateVisionPageRequest, translate_page_in_context

        remaining = self._remaining_budget()
        data = await translate_page_in_context(TranslateVisionPageRequest(
            chapter_id=self.job.chapter_id, page_index=page_index,
            source_lang=source_lang, target_lang=self.settings.target_lang,
            budget_usd=0.25 if remaining is None else max(0.001, min(0.25, remaining)),
            provider=self.provider.id, model=self.settings.model,
            max_objects=RETRY_BATCH, object_ids=object_ids,
        ), memory=self._memory, slice_total=self._slice_total)
        run = data.get("translation_run") or {}
        self.report["translated"] += int(run.get("translated") or 0)
        self._add_cost(run.get("estimated_cost_usd") if self.provider.tracks_cost else None)
        return run

    def _ensure_objects(self, page_index: int) -> None:
        from app.manifest_utils import save_manifest_raw
        from app.text_objects import ensure_page_text_objects

        with get_manifest_lock(self.job.chapter_id):
            manifest = load_manifest_raw(self.job.chapter_id)
            _, changed = ensure_page_text_objects(manifest["pages"][page_index])
            if changed:
                save_manifest_raw(self.job.chapter_id, manifest)

    def _untranslated(self, page_index: int, only: set[str] | None, blank: set[str]) -> list[dict]:
        from app.region_policy import text_object_in_preserve_region

        page = self._manifest()["pages"][page_index]
        return [
            obj for obj in page.get("text_objects") or []
            if isinstance(obj, dict) and obj.get("id") and not obj.get("source_missing")
            and not str(obj.get("translation") or "").strip()
            and (only is None or str(obj["id"]) in only) and str(obj["id"]) not in blank
            and isinstance(obj.get("region"), dict) and not text_object_in_preserve_region(page, obj)
        ]

    async def repair(self) -> None:
        """Use the editor's own tools on what translation reported.

        keep -> preserve region (original pixels back, object left out);
        missed text -> add_box (erased) and translated in small batches;
        a slice whose reply was cut off or skipped objects -> retried in small batches;
        whatever is still untranslated -> original restored and listed for review.
        """
        from app.dependencies import pipeline

        chapter_id = self.job.chapter_id
        pages = sorted(self._repair)
        source_lang = self.report.get("source_lang") or "auto"
        for done, page_index in enumerate(pages):
            self._check_cancel()
            self._progress(done, len(pages), f"Lát {page_index + 1}")
            todo = self._repair[page_index]
            page = self._manifest()["pages"][page_index]
            if page.get("skipped"):
                continue
            height, width = int(page.get("height") or 0), int(page.get("width") or 0)

            keep = [
                {"x1": max(0, x1 - KEEP_PAD_PX), "y1": max(0, y1 - KEEP_PAD_PX),
                 "x2": (min(width, x2 + KEEP_PAD_PX) if width else x2 + KEEP_PAD_PX),
                 "y2": (min(height, y2 + KEEP_PAD_PX) if height else y2 + KEEP_PAD_PX)}
                for x1, y1, x2, y2 in (tuple(int(v) for v in region[:4]) for region in todo.get("keep") or [])
            ]
            if keep:
                try:
                    await asyncio.to_thread(pipeline.preserve_and_reinpaint, chapter_id, page_index, keep)
                    self.report["kept_regions"] += len(keep)
                except (ValueError, RuntimeError, OSError) as exc:
                    _append(self.report["qc_errors"], f"Lát {page_index + 1}: giữ nguyên thất bại: {_detail(exc)[:150]}")

            rects = []
            for box in todo.get("missed") or []:
                x1, y1, x2, y2 = (int(v) for v in box[:4])
                if width and height:
                    x1, x2 = max(0, x1), min(width, x2)
                    y1, y2 = max(0, y1), min(height, y2)
                if x2 - x1 >= 10 and y2 - y1 >= 10:
                    rects.append((x1, y1, x2, y2))
            added = 0
            if rects:
                # One re-inpaint for all of a slice's missed text, not one per box.
                try:
                    await asyncio.to_thread(pipeline.add_manual_boxes, chapter_id, page_index, rects)
                    added = len(rects)
                except (ValueError, RuntimeError, OSError) as exc:
                    _append(self.report["qc_errors"], f"Lát {page_index + 1}: thêm vùng chữ thất bại: {_detail(exc)[:150]}")
            self.report["missed_added"] += added
            if added:
                # Create text objects for the new boxes before retrying.
                await asyncio.to_thread(self._ensure_objects, page_index)

            # New boxes and failed slices: every untranslated object; otherwise only the ones skipped.
            only = None if (added or todo.get("failed")) else set(todo.get("missing") or [])
            if only == set():
                continue
            blank: set[str] = set()
            _append(self.report["retried_pages"], page_index + 1)
            # Keep going while batches make progress; give up after RETRY_ROUNDS fruitless ones.
            fruitless = 0
            while fruitless < RETRY_ROUNDS:
                self._check_cancel()
                targets = self._untranslated(page_index, only, blank)
                if not targets:
                    break
                remaining = self._remaining_budget()
                if remaining is not None and remaining < 0.001:
                    break
                batch = [str(obj["id"]) for obj in targets[:RETRY_BATCH]]
                try:
                    run = await self._translate_retry(page_index, source_lang, batch)
                except HTTPException as exc:
                    _append(self.report["translate_errors"], f"Lát {page_index + 1} (thử lại): {_detail(exc)[:200]}")
                    fruitless += 1
                    continue
                blank.update(run.get("blank_ids") or [])
                left = {str(obj["id"]) for obj in self._untranslated(page_index, only, blank)}
                if set(batch) <= left:
                    fruitless += 1

            unresolved = self._untranslated(page_index, only, blank)
            if unresolved:
                regions = [dict(obj["region"]) for obj in unresolved]
                try:
                    await asyncio.to_thread(pipeline.preserve_and_reinpaint, chapter_id, page_index, regions)
                    self.report["restored_regions"] += len(regions)
                except (ValueError, RuntimeError, OSError) as exc:
                    _append(self.report["qc_errors"], f"Lát {page_index + 1}: khôi phục thất bại: {_detail(exc)[:150]}")
                for obj in unresolved:
                    _append(self.report["review_list"], {
                        "page": page_index + 1, "id": obj["id"],
                        "reason": "AI không dịch được sau khi thử lại; đã giữ ảnh gốc",
                    })
        self._progress(len(pages), len(pages), (
            f"Giữ {self.report['kept_regions']}, thêm {self.report['missed_added']} vùng sót, "
            f"khôi phục {self.report['restored_regions']}"
        ))

    async def render(self) -> None:
        from app.routers.image import _current_rendered_path

        chapter_id = self.job.chapter_id
        indices = self._active_pages()
        for done, page_index in enumerate(indices):
            self._check_cancel()
            self._progress(done, len(indices))
            manifest = self._manifest()
            page = manifest["pages"][page_index]
            if page.get("skipped") or _current_rendered_path(chapter_id, page_index, manifest) is not None:
                continue
            await self._render_one(page_index)
        self._progress(len(indices), len(indices))

    async def _render_one(self, page_index: int) -> None:
        """Render a slice; an object that cannot be lettered gets its original pixels back.

        Export needs every active slice rendered, so one translation too long for
        its region must not cost the whole chapter.
        """
        from app.dependencies import pipeline
        from app.routers.export import _render_request_from_page
        from app.routers.render_commit import render_page

        chapter_id = self.job.chapter_id
        # Sync text objects first so export cannot make the render stale.
        await asyncio.to_thread(self._ensure_objects, page_index)
        for attempt in range(RENDER_RESTORE_ATTEMPTS + 1):
            page = self._manifest()["pages"][page_index]
            try:
                await asyncio.to_thread(render_page, _render_request_from_page(chapter_id, page_index, page))
                return
            except HTTPException as exc:
                detail = _detail(exc)
                _append(self.report["render_errors"], f"Lát {page_index + 1}: {detail[:200]}")
                failed = RENDER_FAILED_OBJECT.search(detail)
                obj = next((
                    item for item in page.get("text_objects") or []
                    if failed and isinstance(item, dict) and str(item.get("id")) == failed.group(1)
                    and isinstance(item.get("region"), dict)
                ), None)
                if obj is None or attempt == RENDER_RESTORE_ATTEMPTS:
                    return
            try:
                await asyncio.to_thread(pipeline.preserve_and_reinpaint, chapter_id, page_index, [dict(obj["region"])])
            except (ValueError, RuntimeError, OSError) as exc:
                _append(self.report["render_errors"], f"Lát {page_index + 1}: khôi phục thất bại: {_detail(exc)[:150]}")
                return
            self.report["restored_regions"] += 1
            _append(self.report["review_list"], {
                "page": page_index + 1, "id": obj["id"],
                "reason": "Bản dịch không vừa vùng chữ; đã giữ ảnh gốc",
            })

    async def export(self) -> None:
        from app.routers.export import _preflight_copy, write_chapter_archive

        chapter_id = self.job.chapter_id
        # The gate normally blocks export until a human reviews every story
        # object. A.I mode exports anyway and reports what a reviewer would see.
        preflight = _preflight_copy(self._manifest(), require_final_approval=False)
        self.report["editorial_blockers"] = int(preflight.get("blocker_count") or 0)
        self.report["blocker_samples"] = [
            {"kind": item.get("kind"), "page": int(item.get("page_index", 0)) + 1, "reason": item.get("reason")}
            for item in (preflight.get("blockers") or [])[:10]
        ]
        self._progress(0, 1, "Đang ghép trang và nén")
        try:
            archive = await asyncio.to_thread(
                write_chapter_archive, chapter_id,
                enforce_editorial_gate=False, archive_name=f"ai_mode_{chapter_id}.zip",
            )
        except HTTPException as exc:
            raise AIModeFailed(f"Không đóng gói được zip: {_detail(exc)}") from exc
        self.job.archive_path = str(archive)
        report_path = OUTPUT_DIR / chapter_id / "ai_mode_report.json"
        report_path.write_text(json.dumps(self.report, ensure_ascii=False, indent=1), encoding="utf-8")
        self._progress(1, 1, "Xong")


class AIModeJobManager:
    def __init__(self, runner_factory=AIModeRunner):
        self._jobs: dict[str, AIModeJob] = {}
        self._runner_factory = runner_factory

    def active_job(self) -> AIModeJob | None:
        return next((job for job in self._jobs.values() if job.status in {"pending", "running"}), None)

    def start(self, settings: AIModeSettings, *, provider, api_key: str, on_finish=None) -> dict:
        if self.active_job() is not None:
            raise RuntimeError("An A.I mode job is already running")
        self._prune()
        job = AIModeJob(job_id=uuid.uuid4().hex, settings=settings)
        job.stages = {key: {"label": label, "status": "pending", "done": 0, "total": 0, "detail": "", "elapsed_s": None}
                      for key, label in STAGES}
        if not provider.tracks_cost:
            job.cost_usd = None
        self._jobs[job.job_id] = job
        runner = self._runner_factory(job, provider, api_key)
        job.task = asyncio.get_running_loop().create_task(self._run(job, runner, on_finish))
        return self.snapshot(job.job_id)

    async def _run(self, job: AIModeJob, runner, on_finish=None) -> None:
        job.status = "running"
        try:
            for key, _label in STAGES:
                if job.cancel_requested:
                    raise AIModeCancelled()
                job.stage = key
                job.stages[key]["status"] = "running"
                started = job.updated_at = time.time()
                try:
                    await getattr(runner, key)()
                finally:
                    job.stages[key]["elapsed_s"] = round(time.time() - started, 1)
                job.stages[key]["status"] = "done"
            job.status = "completed"
        except AIModeCancelled:
            job.status = "cancelled"
            if job.stage:
                job.stages[job.stage]["status"] = "cancelled"
        except Exception as exc:
            logger.opt(exception=True).error("A.I mode job {} failed at {}: {}", job.job_id, job.stage, type(exc).__name__)
            job.status = "failed"
            job.error = _detail(exc)[:500]
            if job.stage:
                job.stages[job.stage]["status"] = "failed"
        finally:
            job.updated_at = time.time()
            if on_finish is not None:
                try:
                    await asyncio.to_thread(on_finish, job)
                except Exception as exc:
                    logger.warning("A.I mode finish hook failed: {}", type(exc).__name__)

    def cancel(self, job_id: str) -> dict:
        job = self._get(job_id)
        if job.status in {"pending", "running"}:
            job.cancel_requested = True
        return self.snapshot(job_id)

    def archive_path(self, job_id: str) -> str | None:
        job = self._get(job_id)
        return job.archive_path if job.status == "completed" else None

    def snapshot(self, job_id: str) -> dict:
        job = self._get(job_id)
        return {
            "job_id": job.job_id,
            "status": job.status,
            "stage": job.stage,
            "stages": [{"key": key, **copy.deepcopy(job.stages[key])} for key, _ in STAGES],
            "chapter_id": job.chapter_id,
            "url": job.settings.url,
            "provider": job.settings.provider,
            "model": job.settings.model,
            "target_lang": job.settings.target_lang,
            "cost_usd": None if job.cost_usd is None else round(job.cost_usd, 6),
            "budget_usd": job.settings.budget_usd,
            "cancel_requested": job.cancel_requested,
            "error": job.error,
            "report": copy.deepcopy(job.report),
            "download_url": f"/api/ai_mode/jobs/{job.job_id}/download" if job.archive_path and job.status == "completed" else None,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
        }

    def _get(self, job_id: str) -> AIModeJob:
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job

    def _prune(self) -> None:
        finished = [job_id for job_id, job in self._jobs.items() if job.status not in {"pending", "running"}]
        while len(self._jobs) >= MAX_RETAINED_JOBS and finished:
            self._jobs.pop(finished.pop(0), None)
