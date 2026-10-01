"""A.I mode job: URL -> four A.I checkpoints (scan, clean, review, translate) -> a lettered chapter.

Every stage reuses the same endpoint functions the editor calls, so locking,
revision checks and manifest bookkeeping are identical to manual work. The
chapter stays a normal chapter afterwards: it opens in the editor, where it
can be fixed by hand and exported.
"""
from __future__ import annotations

import asyncio
import copy
import functools
import json
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace

import requests
from fastapi import HTTPException

from app.ai_mode.checkpoints import _covered, review_clean, settle_clean_review
from app.ai_mode.page_scan import scan_slices
from app.config import OUTPUT_DIR, PROCESSED_DIR, RAW_DIR
from app.image_io import read_image
from app.logging_config import logger
from app.manifest_utils import get_manifest_lock, load_manifest_raw, save_manifest_raw
from app.security import validate_managed_path

STAGES: tuple[tuple[str, str], ...] = (
    ("download", "Tải chương"),
    ("scan", "Checkpoint 1: AI bỏ lát credit, lát trống và giữ logo"),
    ("clean", "Checkpoint 2: Clean ảnh"),
    ("review", "Checkpoint 3: AI so ảnh gốc và ảnh clean"),
    ("translate", "Checkpoint 4: Dịch và chọn font"),
    ("polish", "Nâng cao: Jev soát câu dịch"),
    ("render", "Render"),
    ("finish", "Hoàn tất"),
)
SCAN_BATCH_SIZE = 4
SCAN_CONCURRENCY = 3
GLOSSARY_BATCH_SIZE = 6
# More "credit" slices than this is a misread of the chapter, not credits.
CREDIT_MAX_SHARE = 0.25
CREDIT_MAX_ABSOLUTE = 3
# More than this share of textless slices means the scan misread the chapter.
TEXTLESS_MAX_SHARE = 0.5
REVIEW_CONCURRENCY = 8  # review (checkpoint 3) has no reading-order dependency
TRANSLATE_CONCURRENCY = 3
# Objects per retry request, and retry batches without progress before restoring the original.
RETRY_BATCH = 4
RETRY_ROUNDS = 3
# Objects per slice given back their original pixels when their translation cannot be lettered.
RENDER_RESTORE_ATTEMPTS = 3
POLISH_ROUNDS = 2  # rewrites of one line before its best version stays
POLISH_CONCURRENCY = 8  # judge calls in flight; each is one short text
MAX_GRADED_KEPT = 400  # first grades kept in the report
RENDER_FAILED_OBJECT = re.compile(r"\(vùng ([\w-]+)\)")
POLL_SECONDS = 1.0
# A.I calls wait on the network, so they get their own threads instead of queueing behind re-inpaints.
_AI_CALLS = ThreadPoolExecutor(max_workers=16, thread_name_prefix="ai-mode-call")
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
    polish: bool = False  # advanced: a judge model grades each line and weak ones are rewritten


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
    cancel_requested: bool = False
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    task: asyncio.Task | None = field(default=None, repr=False)


def _detail(exc: Exception) -> str:
    if isinstance(exc, HTTPException):
        return str(exc.detail)
    return str(exc) or type(exc).__name__


async def _ai_call(fn, *args):
    """Run a blocking A.I call on the A.I thread pool."""
    return await asyncio.get_running_loop().run_in_executor(_AI_CALLS, functools.partial(fn, *args))


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
            "retried_pages": [], "restored_regions": 0, "sfx_kept": 0, "review_list": [],
            "polish": {"judged": 0, "flagged": 0, "rewritten": 0, "still_flagged": 0, "errors": [], "samples": []},
        })
        self._memory = None
        self._glossary_task: asyncio.Task | None = None
        self._review_queue: asyncio.Queue | None = None
        self._review_task: asyncio.Task | None = None
        self._review_spill: dict[int, dict[str, list]] = {}
        # Logo regions the scan added, per slice; the review may find leftover text on one and free it.
        self._scan_logos: dict[int, list[tuple[int, int, int, int]]] = {}
        self._slice_total: int | None = None

    # -- bookkeeping ---------------------------------------------------------
    def _check_cancel(self) -> None:
        if self.job.cancel_requested:
            raise AIModeCancelled()

    def _progress(self, done: int, total: int, detail: str | None = None, stage: str | None = None) -> None:
        state = self.job.stages[stage or self.job.stage]
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
        def scan_call(batch: list[int]):
            images = [
                (index, read_image(validate_managed_path(pages[index]["original"], RAW_DIR / chapter_id)))
                for index in batch
            ]
            return scan_slices(self.provider, self.settings.model, self.api_key, images)

        async def scan_images(batch: list[int]) -> None:
            found, cost = await _ai_call(scan_call, batch)
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
            self._scan_logos[scan.page_index] = list(scan.logos)
            logos += len(scan.logos)
        self.report["logo_regions"] = logos
        self._progress(len(active), len(active),
                       f"{len(credits)} lát credit, {len(textless)} lát không chữ, {logos} logo")

    async def clean(self) -> None:
        from app.routers.chapters import chapter_processing_jobs

        indices = self._active_pages()
        # The glossary only needs the original slices, so it is read while cleanup runs.
        self._glossary_task = asyncio.create_task(self._read_glossary(indices))
        if not indices:
            self._progress(0, 0, "Không có lát nào cần clean")
            return
        snapshot = chapter_processing_jobs.start(
            self.job.chapter_id, page_indices=indices, workers=self.settings.workers,
        )
        job_id = snapshot["job_id"]
        # Checkpoint 3 starts on each slice as soon as it is clean, so its A.I calls overlap the cleanup.
        self._start_review(len(indices))
        wanted, fed = set(indices), 0
        try:
            while True:
                snapshot = chapter_processing_jobs.snapshot(job_id)
                self._progress(snapshot["completed"], snapshot["total"])
                done = snapshot.get("done_indices") or []
                for page_index in done[fed:]:
                    if page_index in wanted:
                        self._review_queue.put_nowait(page_index)
                fed = len(done)
                if snapshot["status"] in {"completed", "failed"}:
                    break
                await asyncio.sleep(POLL_SECONDS)
            if snapshot["status"] == "failed":
                errors = "; ".join(str(item.get("message")) for item in snapshot.get("errors") or [])
                raise AIModeFailed(f"Clean thất bại: {errors or 'lỗi không rõ'}")
            self._check_cancel()
        except BaseException:
            self._review_task.cancel()
            raise
        self._review_queue.put_nowait(None)

    async def _read_glossary(self, indices: list[int]) -> dict:
        """Names, terms and forms of address for the whole chapter; empty when the read fails."""
        from app.ai_mode.glossary import merge_glossaries, read_glossary
        from app.translation.deepseek import _language_name

        pages = self._manifest().get("pages", [])
        gate = asyncio.Semaphore(SCAN_CONCURRENCY)

        def read_batch(batch: list[int]):
            images = [(index, read_image(validate_managed_path(pages[index]["original"], RAW_DIR / self.job.chapter_id)))
                      for index in batch]
            return read_glossary(self.provider, self.settings.model, self.api_key,
                                 _language_name(self.settings.target_lang), self.settings.target_lang, images)

        async def read(batch: list[int]) -> dict:
            async with gate:
                if self.job.cancel_requested:
                    return {}
                try:
                    data, cost = await _ai_call(read_batch, batch)
                except (RuntimeError, ValueError, OSError) as exc:
                    _append(self.report["translate_errors"], f"Bảng thuật ngữ lát {batch[0] + 1}: {_detail(exc)[:150]}")
                    return {}
                self._add_cost(cost)
                return data

        batches = [indices[i:i + GLOSSARY_BATCH_SIZE] for i in range(0, len(indices), GLOSSARY_BATCH_SIZE)]
        glossary = merge_glossaries(list(await asyncio.gather(*(read(batch) for batch in batches))))
        self.report["glossary"] = glossary
        return glossary

    def _images(self, page_index: int, second_key: str, second_root) -> tuple:
        """The raw slice and one derived image (clean or rendered) of a page."""
        page = self._manifest()["pages"][page_index]
        chapter_id = self.job.chapter_id
        original = read_image(validate_managed_path(page["original"], RAW_DIR / chapter_id))
        other = read_image(validate_managed_path(page[second_key], second_root / chapter_id))
        return original, other

    def _in_core(self, page_index: int, boxes, spill: dict | None = None, kind: str = "") -> tuple:
        """Boxes whose centre is in the part of the slice that is exported; a box crossing the cut is also given to the neighbour."""
        pages = self._manifest()["pages"]
        core = pages[page_index].get("stitch_core")
        if not isinstance(core, dict):
            return tuple(boxes)
        try:
            top, bottom, offset = int(core["core_y1"]), int(core["core_y2"]), int(core["source_y1"])
        except (KeyError, TypeError, ValueError):
            return tuple(boxes)
        kept = []
        for box in boxes:
            if not top <= (box[1] + box[3]) / 2 < bottom:
                continue
            kept.append(box)
            if spill is None or (box[1] >= top and box[3] <= bottom):
                continue
            for other_index in (page_index - 1,) * (box[1] < top) + (page_index + 1,) * (box[3] > bottom):
                other = pages[other_index] if 0 <= other_index < len(pages) else {}
                other_core = other.get("stitch_core")
                if (other.get("skipped") or not isinstance(other_core, dict)
                        or other.get("source_page") != pages[page_index].get("source_page")):
                    continue
                shift = offset - int(other_core["source_y1"])
                spill.setdefault(other_index, {}).setdefault(kind, []).append(
                    (box[0], box[1] + shift, box[2], box[3] + shift))
        return tuple(kept)

    def _page_boxes(self, page_index: int) -> list[tuple[int, int, int, int]]:
        """Pixel rectangles of the page's active text boxes."""
        boxes = self._manifest()["pages"][page_index].get("boxes") or []
        return [(int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])) for b in boxes
                if isinstance(b, dict) and not b.get("removed") and all(k in b for k in ("x1", "y1", "x2", "y2"))]

    def _start_review(self, total: int) -> None:
        """Start checkpoint 3 as a queue: each slice put on it is checked; None ends it."""
        self._review_queue = asyncio.Queue()
        self._review_spill = {}
        self._review_task = asyncio.create_task(self._review_stream(total))

    async def _review_stream(self, total: int) -> None:
        from app.dependencies import pipeline

        chapter_id = self.job.chapter_id
        gate = asyncio.Semaphore(REVIEW_CONCURRENCY)
        finished = 0
        spill = self._review_spill  # fixes crossing a slice cut, for the neighbouring slice

        def check(page_index: int):
            """Ask the model about one slice and keep what applies to it; manifest reads stay off the event loop."""
            original, clean = self._images(page_index, "clean", PROCESSED_DIR)
            found, cost = review_clean(self.provider, self.settings.model, self.api_key, page_index, original, clean)
            found = settle_clean_review(found, self._page_boxes(page_index))
            crossing: dict[int, dict[str, list]] = {}
            found = replace(found, missed=self._in_core(page_index, found.missed, crossing, "missed"),
                            residue=self._in_core(page_index, found.residue, crossing, "residue"),
                            restore=self._in_core(page_index, found.restore))
            return found, cost, crossing

        async def review_one(page_index: int) -> None:
            nonlocal finished
            try:
                # Only the model call holds the gate; the local fix runs while other slices are checked.
                async with gate:
                    remaining = self._remaining_budget()
                    if self.job.cancel_requested or (remaining is not None and remaining < 0.001):
                        return
                    found, cost, crossing = await _ai_call(check, page_index)
                    self._add_cost(cost)
                for other, kinds in crossing.items():
                    for kind, boxes in kinds.items():
                        spill.setdefault(other, {}).setdefault(kind, []).extend(boxes)
                if found.residue and self._scan_logos.get(page_index):
                    await asyncio.to_thread(self._free_logos_with_text, page_index, found.residue)
                if found.restore or found.missed or found.residue:
                    await asyncio.to_thread(
                        pipeline.apply_review_fixes, chapter_id, page_index,
                        preserve=list(found.restore), boxes=list(found.missed), repaint=list(found.residue))
                    self.report["kept_regions"] += len(found.restore)
                    self.report["missed_added"] += len(found.missed)
                    self.report["repainted_regions"] += len(found.residue)
                    if found.residue:
                        _append(self.report["repaint_pages"], page_index)
            except (HTTPException, RuntimeError, ValueError, OSError, KeyError) as exc:
                _append(self.report["qc_errors"], f"Lát {page_index + 1}: {_detail(exc)[:200]}")
            finally:
                finished += 1
                self._progress(finished, total, stage="review")

        self._progress(0, total, stage="review")
        tasks = []
        try:
            while (page_index := await self._review_queue.get()) is not None:
                tasks.append(asyncio.create_task(review_one(page_index)))
            await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            raise

    def _free_logos_with_text(self, page_index: int, residue) -> None:
        """A scan logo region the review finds leftover text on was a watermark; it stops protecting that text."""
        freed = [logo for logo in self._scan_logos[page_index] if _covered(logo, list(residue))]
        if not freed:
            return
        with get_manifest_lock(self.job.chapter_id):
            manifest = load_manifest_raw(self.job.chapter_id)
            page = manifest["pages"][page_index]
            keep = [region for region in page.get("preserve_regions") or []
                    if (region.get("x1"), region.get("y1"), region.get("x2"), region.get("y2")) not in freed]
            if len(keep) != len(page.get("preserve_regions") or []):
                page["preserve_regions"] = keep
                save_manifest_raw(self.job.chapter_id, manifest)
        self._scan_logos[page_index] = [logo for logo in self._scan_logos[page_index] if logo not in freed]
        self.report["logos_freed"] = self.report.get("logos_freed", 0) + len(freed)

    async def review(self) -> None:
        """Checkpoint 3: the model compares each raw and clean slice; the system applies what it reports."""
        from app.dependencies import pipeline

        chapter_id = self.job.chapter_id
        indices = self._active_pages()
        if self._review_task is None:  # the cleanup did not feed it, so every slice is checked now
            self._start_review(len(indices))
            for page_index in indices:
                self._review_queue.put_nowait(page_index)
            self._review_queue.put_nowait(None)
        await self._review_task
        for page_index, fixes in self._review_spill.items():
            # The neighbour may already hold that text as its own box.
            existing = self._page_boxes(page_index)
            missed = [box for box in fixes.get("missed", []) if not _covered(box, existing)]
            try:
                await asyncio.to_thread(pipeline.apply_review_fixes, chapter_id, page_index,
                                        boxes=missed, repaint=fixes.get("residue", []))
            except (HTTPException, RuntimeError, ValueError, OSError, KeyError) as exc:
                _append(self.report["qc_errors"], f"Lát {page_index + 1}: {_detail(exc)[:200]}")
        self._check_cancel()
        self._progress(len(indices), len(indices), (
            f"Xoá thêm {self.report['missed_added']} vùng sót, repaint {self.report['repainted_regions']}, "
            f"giữ nguyên {self.report['kept_regions']}"
        ))

    async def translate(self) -> None:
        from app.ocr.language_detect import site_language_hint
        from app.routers.translation import TranslateVisionPageRequest, translate_page_in_context
        from app.translation.context import ChapterMemory

        # The vision model reads the source language off the image, so no OCR probe runs here.
        manifest = self._manifest()
        source_lang = manifest.get("source_lang") or site_language_hint(manifest.get("source_url")) or "auto"
        self.report["source_lang"] = source_lang
        indices = self._active_pages()
        try:
            glossary = await self._glossary_task if self._glossary_task else {}
        except Exception as exc:  # the glossary only helps; translation goes on without it
            _append(self.report["translate_errors"], f"Bảng thuật ngữ: {_detail(exc)[:150]}")
            glossary = {}
        memory = self._memory = ChapterMemory(self.settings.story_notes, glossary)
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
                    ), memory=memory, slice_total=slice_total, skip_seam_mirrors=True)
                    run = data.get("translation_run") or {}
                except HTTPException as exc:
                    _append(self.report["translate_errors"], f"Lát {page_index + 1}: {_detail(exc)[:200]}")
                    run = None
                if run is not None:
                    await self._keep_art(page_index, run)
                    self.report["translated"] += int(run.get("translated") or 0)
                    self.report["unreadable"] += int(run.get("unreadable") or 0)
                    self.report["review_flags"] += int(run.get("review") or 0)
                    self._add_cost(run.get("estimated_cost_usd") if self.provider.tracks_cost else None)
                    if run.get("render_error"):
                        _append(self.report["render_errors"], f"Lát {page_index + 1}: {run['render_error']}")
                # A failed or cut-off slice retries everything; otherwise only what the model skipped.
                if run is None or run.get("remaining") or run.get("missing_ids"):
                    only = None if run is None or run.get("remaining") else set(run.get("missing_ids") or [])
                    await self._retry_untranslated(page_index, source_lang, only)
                finished += 1
                self._progress(finished, len(indices))

        await asyncio.to_thread(self._join_stacked_lines, indices)
        self._progress(0, len(indices))
        await asyncio.gather(*(translate_one(page_index) for page_index in indices))
        self._check_cancel()
        if out_of_budget:
            _append(self.report["translate_errors"], "Hết ngân sách, các lát còn lại chưa dịch")
        await self._sync_seams()
        sheet = memory.snapshot()
        self.report["characters"] = sheet["characters"][:MAX_REPORT_ITEMS]
        self.report["address"] = sheet["address"][:MAX_REPORT_ITEMS]
        self._progress(len(indices), len(indices), f"Dịch {self.report['translated']} vùng")

    async def polish(self) -> None:
        """Advanced: the judge grades each translated line; weak lines are rewritten and kept only when graded better."""
        from app.ai_mode import polish

        stats = self.report["polish"]
        if not self.settings.polish:
            self._progress(0, 0, "Tắt")
            return
        from app.ai_mode.seams import seam_mirror_ids

        manifest = self._manifest()
        indices = self._active_pages()
        # A text crossing a slice cut is graded on the slice that owns it; the copy follows.
        mirrors = {(index, str(obj_id)) for index in indices for obj_id in seam_mirror_ids(manifest, index)}
        lines = [line for line in polish.collect_lines(manifest.get("pages", []), indices)
                 if (line.page_index, line.id) not in mirrors]
        memory = self._memory.snapshot() if self._memory is not None else {}
        gate = asyncio.Semaphore(POLISH_CONCURRENCY)
        unavailable = False

        async def grade(line, text=None):
            nonlocal unavailable
            async with gate:
                if unavailable or self.job.cancel_requested:
                    return None
                try:
                    return await _ai_call(polish.judge, self.provider, self.api_key, polish.line_state(line, text))
                except (RuntimeError, ValueError, OSError, requests.RequestException) as exc:
                    _append(stats["errors"], _detail(exc)[:200])
                    # No judge behind this provider: stop asking.
                    unavailable = unavailable or " 404" in _detail(exc)
                    return None

        self._progress(0, len(lines))
        pending = lines
        for _round in range(POLISH_ROUNDS):
            scores = await asyncio.gather(*(grade(line) for line in pending))
            judged = [(line, found) for line, found in zip(pending, scores) if found]
            for line, found in judged:
                line.scores = found
            if _round == 0:
                # Every first grade is kept, so the score limits can be calibrated on real chapters.
                stats["scores"] = [{"page": line.page_index + 1, "text": line.text, "scores": found}
                                   for line, found in judged[:MAX_GRADED_KEPT]]
                stats["judged"] = len(judged)
                stats["flagged"] = sum(bool(polish.problems(line.scores)) for line, _ in judged)
            flagged = [line for line, _ in judged if polish.problems(line.scores)]
            if not flagged or self.job.cancel_requested:
                break
            batches = [flagged[i:i + polish.REWRITE_BATCH] for i in range(0, len(flagged), polish.REWRITE_BATCH)]
            kept = []
            for batch in batches:
                try:
                    texts, cost = await _ai_call(polish.rewrite, self.provider, self.settings.model, self.api_key,
                                                 batch, memory)
                except (RuntimeError, ValueError, OSError) as exc:
                    _append(stats["errors"], _detail(exc)[:200])
                    continue
                self._add_cost(cost if self.provider.tracks_cost else None)
                candidates = [(line, texts[line.id]) for line in batch if texts.get(line.id, line.text) != line.text]
                regraded = await asyncio.gather(*(grade(line, text) for line, text in candidates))
                for (line, text), found in zip(candidates, regraded):
                    if found and polish.quality(found) > polish.quality(line.scores):
                        _append(stats["samples"], {"page": line.page_index + 1, "source": line.source,
                                                   "before": line.text, "after": text,
                                                   "scores_before": line.scores, "scores_after": found})
                        line.text, line.scores = text, found
                        kept.append(line)
            await asyncio.to_thread(self._save_polished, kept)
            stats["rewritten"] += len(kept)
            # A line still flagged, rewritten or not, gets one more try with its current text.
            pending = [line for line in flagged if polish.problems(line.scores)]
            self._progress(len(lines), len(lines))
        stats["still_flagged"] = sum(bool(polish.problems(line.scores)) for line in lines if line.scores)
        if stats["rewritten"]:
            await self._sync_seams()
        self._progress(len(lines), len(lines), f"Chấm {stats['judged']} câu, sửa {stats['rewritten']}")

    def _save_polished(self, lines) -> None:
        from app.manifest_utils import invalidate_page_render

        if not lines:
            return
        with get_manifest_lock(self.job.chapter_id):
            manifest = load_manifest_raw(self.job.chapter_id)
            for line in lines:
                page = manifest["pages"][line.page_index]
                obj = next((o for o in page.get("text_objects") or []
                            if isinstance(o, dict) and str(o.get("id")) == line.id), None)
                if obj is None:
                    continue
                obj.setdefault("polished_from", obj.get("translation"))
                obj["translation"] = obj["auto_translation"] = line.text
                invalidate_page_render(manifest, line.page_index)
            save_manifest_raw(self.job.chapter_id, manifest)

    async def _sync_seams(self) -> list[int]:
        """Letter every text crossing a slice cut exactly as its owning slice does."""
        from app.ai_mode.seams import drop_overlapping_letters, sync_seam_mirrors
        from app.manifest_utils import invalidate_page_render, save_manifest_raw

        def sync() -> list[int]:
            with get_manifest_lock(self.job.chapter_id):
                manifest = load_manifest_raw(self.job.chapter_id)
                dropped = drop_overlapping_letters(manifest)
                self.report["overlaps_dropped"] = self.report.get("overlaps_dropped", 0) + len(dropped)
                changed = sorted(set(sync_seam_mirrors(manifest)) | {page_index for page_index, _ in dropped})
                for page_index in changed:
                    invalidate_page_render(manifest, page_index)
                if changed:
                    save_manifest_raw(self.job.chapter_id, manifest)
                return changed

        changed = await asyncio.to_thread(sync)
        self.report["seam_copies"] = self.report.get("seam_copies", 0) + len(changed)
        return changed

    async def _translate_retry(self, page_index: int, source_lang: str, object_ids: list[str] | None) -> dict:
        from app.routers.translation import TranslateVisionPageRequest, translate_page_in_context

        remaining = self._remaining_budget()
        data = await translate_page_in_context(TranslateVisionPageRequest(
            chapter_id=self.job.chapter_id, page_index=page_index,
            source_lang=source_lang, target_lang=self.settings.target_lang,
            budget_usd=0.25 if remaining is None else max(0.001, min(0.25, remaining)),
            provider=self.provider.id, model=self.settings.model,
            max_objects=RETRY_BATCH, object_ids=object_ids,
        ), memory=self._memory, slice_total=self._slice_total, skip_seam_mirrors=True)
        run = data.get("translation_run") or {}
        self.report["translated"] += int(run.get("translated") or 0)
        self._add_cost(run.get("estimated_cost_usd") if self.provider.tracks_cost else None)
        await self._keep_art(page_index, run)
        return run

    async def _keep_art(self, page_index: int, run: dict) -> None:
        """Sound effects the model left as art get their original pixels back."""
        from app.dependencies import pipeline

        regions = run.get("art_regions") or []
        if not regions:
            return
        try:
            await asyncio.to_thread(pipeline.preserve_and_reinpaint, self.job.chapter_id, page_index, regions)
            self.report["sfx_kept"] += len(regions)
        except (ValueError, RuntimeError, OSError) as exc:
            _append(self.report["translate_errors"], f"Lát {page_index + 1}: giữ SFX thất bại: {_detail(exc)[:150]}")

    def _join_stacked_lines(self, indices: list[int]) -> None:
        from app.ai_mode.seams import join_stacked_lines, set_letter_bounds
        from app.manifest_utils import save_manifest_raw
        from app.text_objects import ensure_page_text_objects

        with get_manifest_lock(self.job.chapter_id):
            manifest = load_manifest_raw(self.job.chapter_id)
            for page_index in indices:
                ensure_page_text_objects(manifest["pages"][page_index])
                set_letter_bounds(manifest["pages"][page_index])
            self.report["lines_joined"] = join_stacked_lines(manifest, indices)
            save_manifest_raw(self.job.chapter_id, manifest)

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

        from app.ai_mode.seams import seam_mirror_ids

        manifest = self._manifest()
        page = manifest["pages"][page_index]
        blank = blank | seam_mirror_ids(manifest, page_index)
        return [
            obj for obj in page.get("text_objects") or []
            if isinstance(obj, dict) and obj.get("id") and not obj.get("source_missing")
            and not obj.get("overlap_dropped") and not obj.get("joined_into")
            and not str(obj.get("translation") or "").strip()
            and (only is None or str(obj["id"]) in only) and str(obj["id"]) not in blank
            and isinstance(obj.get("region"), dict) and not text_object_in_preserve_region(page, obj)
        ]

    async def _retry_untranslated(self, page_index: int, source_lang: str, only: set[str] | None) -> None:
        """Retry untranslated objects in small batches; what never translates keeps its original pixels."""
        from app.dependencies import pipeline

        await asyncio.to_thread(self._ensure_objects, page_index)
        blank: set[str] = set()
        _append(self.report["retried_pages"], page_index + 1)
        fruitless = 0
        while fruitless < RETRY_ROUNDS and not self.job.cancel_requested:
            targets = self._untranslated(page_index, only, blank)
            remaining = self._remaining_budget()
            if not targets or (remaining is not None and remaining < 0.001):
                break
            batch = [str(obj["id"]) for obj in targets[:RETRY_BATCH]]
            try:
                run = await self._translate_retry(page_index, source_lang, batch)
            except HTTPException as exc:
                _append(self.report["translate_errors"], f"Lát {page_index + 1} (thử lại): {_detail(exc)[:200]}")
                fruitless += 1
                continue
            blank.update(run.get("blank_ids") or [])
            if set(batch) <= {str(obj["id"]) for obj in self._untranslated(page_index, only, blank)}:
                fruitless += 1
        unresolved = self._untranslated(page_index, only, blank)
        if not unresolved:
            return
        regions = [dict(obj["region"]) for obj in unresolved]
        try:
            await asyncio.to_thread(pipeline.preserve_and_reinpaint, self.job.chapter_id, page_index, regions)
            self.report["restored_regions"] += len(regions)
        except (ValueError, RuntimeError, OSError) as exc:
            _append(self.report["translate_errors"], f"Lát {page_index + 1}: khôi phục thất bại: {_detail(exc)[:150]}")
        for obj in unresolved:
            _append(self.report["review_list"], {
                "page": page_index + 1, "id": obj["id"],
                "reason": "AI không dịch được sau khi thử lại; đã giữ ảnh gốc",
            })

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

    async def finish(self) -> None:
        """Report what a reviewer would still see and leave the chapter open in the editor."""
        from app.routers.export import _preflight_copy

        chapter_id = self.job.chapter_id
        with get_manifest_lock(chapter_id):
            manifest = load_manifest_raw(chapter_id)
            manifest["workflow"] = {"stage": "review", "page_index": 0}
            save_manifest_raw(chapter_id, manifest)
        preflight = _preflight_copy(manifest, require_final_approval=False)
        self.report["editorial_blockers"] = int(preflight.get("blocker_count") or 0)
        self.report["blocker_samples"] = [
            {"kind": item.get("kind"), "page": int(item.get("page_index", 0)) + 1, "reason": item.get("reason")}
            for item in (preflight.get("blockers") or [])[:10]
        ]
        report_path = OUTPUT_DIR / chapter_id / "ai_mode_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
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
