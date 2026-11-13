from __future__ import annotations

import asyncio
import copy
import functools
import json
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import requests
from fastapi import HTTPException

from app.ai_mode import decider
from app.ai_mode.checkpoints import _covered
from app.ai_mode.job_rendering import AIModeRenderingMixin
from app.ai_mode.job_seams import AIModeSeamsMixin
from app import series_memory
from app.ai_mode.page_scan import scan_slices
from app.config import OUTPUT_DIR, PROCESSED_DIR, RAW_DIR
from app.image_io import read_image
from app.logging_config import logger
from app.manifest_utils import get_manifest_lock, load_manifest_raw, save_manifest_raw
from app.parameters import PIPELINE_DEFAULT_WORKERS
from app.security import validate_managed_path

STAGES: tuple[tuple[str, str], ...] = (
    ("download", "Tải chương"),
    ("scan", "Checkpoint 1: AI bỏ lát credit, lát trống và giữ logo"),
    ("clean", "Checkpoint 2: Clean ảnh"),
    ("review", "Checkpoint 3: AI so ảnh gốc và ảnh clean"),
    ("translate", "Checkpoint 4: Dịch và chọn font"),
    ("polish", "Nâng cao: soát câu dịch"),
    ("render", "Render"),
    ("finish", "Hoàn tất"),
)
SCAN_CONCURRENCY = 4
GLOSSARY_TEXT_SLICES = 40
CREDIT_MAX_SHARE = 0.25
CREDIT_MAX_ABSOLUTE = 3
TEXTLESS_MAX_SHARE = 0.5
REVIEW_CONCURRENCY = 8
TRANSLATE_CONCURRENCY = 3
RETRY_BATCH = 4
RETRY_ROUNDS = 3
RENDER_RESTORE_ATTEMPTS = 3
POLISH_ROUNDS = 2
POLISH_CONCURRENCY = 8
MAX_GRADED_KEPT = 400
RENDER_FAILED_OBJECT = re.compile(r"\(vùng ([^\s)]+)\)")
POLL_SECONDS = 1.0
_AI_CALLS = ThreadPoolExecutor(max_workers=16, thread_name_prefix="ai-mode-call")
MAX_RETAINED_JOBS = 16
MAX_REPORT_ITEMS = 50


def decide_skips(scan_credits, scan_textless, storyless, chances, active_count):
    credits = sorted(scan_credits)
    textless = sorted(scan_textless)
    skip_kept, skip_unjudged, skip_vetoed = [], [], []
    if storyless is not None:
        storyless = set(storyless)
        skip_kept = sorted((set(credits) | set(textless)) - storyless)
        skip_unjudged = sorted(index for index, chance in (chances or {}).items() if chance is None)
        skip_vetoed = sorted(storyless - set(credits) - set(textless))
        credits = [index for index in credits if index in storyless]
        textless = [index for index in textless if index in storyless]
    limit = max(CREDIT_MAX_ABSOLUTE, int(CREDIT_MAX_SHARE * active_count))
    credit_rejected = credits if len(credits) > limit else []
    if credit_rejected:
        credits = []
    textless = [index for index in textless if index not in credits]
    textless_rejected = textless if len(textless) > TEXTLESS_MAX_SHARE * active_count else []
    if textless_rejected:
        textless = []
    return {"credits": credits, "textless": textless, "credit_rejected": credit_rejected,
            "textless_rejected": textless_rejected, "skip_kept": skip_kept,
            "skip_unjudged": skip_unjudged, "skip_vetoed": skip_vetoed,
            "judge_available": storyless is not None,
            "single_judge_skips": sorted(set(credits) | set(textless)) if storyless is None else []}


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
    workers: int = PIPELINE_DEFAULT_WORKERS
    story_notes: str = ""
    polish: bool = False
    images_per_request: int = 2
    translate_provider: str | None = None
    translate_model: str | None = None
    chapter_id: str | None = None


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
        detail = exc.detail
        return str(detail.get("message", detail)) if isinstance(detail, dict) else str(detail)
    return str(exc) or type(exc).__name__


async def _ai_call(fn, *args):
    return await asyncio.get_running_loop().run_in_executor(_AI_CALLS, functools.partial(fn, *args))


async def _cancel_and_wait(task: asyncio.Task | None) -> None:
    if task is None:
        return
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass


def _object_has_text_evidence(obj: dict) -> bool:
    if str(obj.get("ocr_text") or "").strip():
        return True
    if str(obj.get("source_read") or "").strip():
        return True
    return bool(str(obj.get("typography_role") or "").strip()
                or str(obj.get("role") or "").strip()
                or str(obj.get("container") or "").strip())


def _append(items: list, value) -> None:
    if len(items) < MAX_REPORT_ITEMS:
        items.append(value)


def _one_based(indices) -> list[int]:
    return [int(index) + 1 for index in indices]


def pipeline_fix(chapter_id: str, page_index: int, rects: list) -> None:
    from app.dependencies import pipeline

    pipeline.apply_review_fixes(chapter_id, page_index, repaint=list(rects))


class AIModeRunner(AIModeSeamsMixin, AIModeRenderingMixin):

    def __init__(self, job: AIModeJob, provider, api_key: str):
        self.job = job
        self.provider = provider
        self.api_key = api_key
        self.settings = job.settings
        self.report = job.report
        self.report.update({
            "page_numbering": "1-based",
            "credit_pages": [], "credit_rejected": [], "logo_regions": 0, "scan_errors": [],
            "repainted_regions": 0, "repaint_pages": [], "qc_flagged": 0, "qc_errors": [],
            "translated": 0, "unreadable": 0, "review_flags": 0, "translate_errors": [], "render_errors": [],
            "editorial_blockers": 0, "blocker_samples": [], "source_lang": None,
            "textless_pages": [], "textless_rejected": [], "kept_regions": 0, "missed_added": 0,
            "retried_pages": [], "restored_regions": 0, "sfx_kept": 0, "review_list": [], "unboxed_pages": [],
            "repaint_retried": 0, "residue_left": [], "art_kept": 0, "skip_kept": [], "skip_unjudged": [],
            "skip_vetoed": [], "judge": {}, "single_judge_skips": [],
            "polish": {"judged": 0, "flagged": 0, "rewritten": 0, "still_flagged": 0, "gate_dropped": 0,
                       "errors": [], "samples": []},
        })
        self._art_restore_reviewed: set[tuple[int, str]] = set()
        self._report_page_counts: dict[tuple[int, int], int] = {}
        self._memory = None
        self._series: dict | None = None
        self._glossary_task: asyncio.Task | None = None
        self._review_queue: asyncio.Queue | None = None
        self._review_task: asyncio.Task | None = None
        self._scan_logos: dict[int, list[tuple[int, int, int, int]]] = {}
        self._transcripts: dict[int, tuple[tuple[str, str], ...]] = {}
        self._slice_total: int | None = None

    def _append_page_capped(self, items: list, value: dict) -> None:
        page = int(value.get("page") or 0)
        key = (id(items), page)
        if self._report_page_counts.get(key, 0) >= MAX_REPORT_ITEMS:
            return
        self._report_page_counts[key] = self._report_page_counts.get(key, 0) + 1
        items.append(value)

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

    async def download(self) -> None:
        if self.settings.chapter_id:
            self.job.chapter_id = str(self.settings.chapter_id)
            manifest = await asyncio.to_thread(self._manifest)
            self._progress(1, 1, f"{len(manifest.get('pages') or [])} lát")
            return
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
        size = max(1, self.settings.images_per_request)
        batches = [active[i:i + size] for i in range(0, len(active), size)]
        scans = []
        def scan_call(batch: list[int]):
            images = [
                (index, read_image(validate_managed_path(pages[index]["original"], RAW_DIR / chapter_id)))
                for index in batch
            ]
            return scan_slices(self.provider, self.settings.model, self.api_key, images)

        async def scan_images(batch: list[int]) -> None:
            found, cost, malformed = await _ai_call(scan_call, batch)
            self._add_cost(cost)
            scans.extend(found)
            for bad in malformed:
                slice_value = bad.get("slice")
                _append(self.report["scan_errors"], {
                    "pages": ([int(slice_value) + 1] if isinstance(slice_value, int)
                              and not isinstance(slice_value, bool) else []),
                    "error": f"Malformed scan entry (slice {slice_value!r}): {bad.get('reason')}",
                })

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
                    _append(self.report["scan_errors"], {"pages": [index + 1], "error": _detail(exc)[:300]})
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
                    _append(self.report["scan_errors"], {"pages": _one_based(batch), "error": _detail(exc)[:300]})
            finished += len(batch)
            self._progress(finished, len(active))
            return True

        def load(index: int):
            return read_image(validate_managed_path(pages[index]["original"], RAW_DIR / chapter_id))

        judge_status = decider.judge_status(self.provider, self.api_key)
        judge = None
        if judge_status["available"]:
            judge = asyncio.ensure_future(asyncio.to_thread(decider.storyless_slices, self.provider, self.api_key, active, load))
        else:
            logger.warning("Decider judge unavailable (provider {}: {}); the scan alone decides skips",
                           judge_status["provider"], judge_status["reason"])

        self._progress(0, len(active))
        try:
            if batches and not await scan_batch(batches[0]):
                await asyncio.gather(*(scan_one(index) for index in active))
            elif batches:
                results = await asyncio.gather(*(scan_batch(batch) for batch in batches[1:]))
                rejected = [index for batch, ok in zip(batches[1:], results) if not ok for index in batch]
                await asyncio.gather(*(scan_one(index) for index in rejected))
            storyless = await judge if judge is not None else None
        except BaseException:
            await _cancel_and_wait(judge)
            raise
        self._check_cancel()

        storyless_set, chances = storyless if storyless is not None else (None, None)
        decision = decide_skips(
            {scan.page_index for scan in scans if scan.is_credit},
            {scan.page_index for scan in scans if scan.no_text},
            storyless_set, chances, len(active),
        )
        self.report["judge"] = {**judge_status, "single_judge_skips": _one_based(decision["single_judge_skips"])}
        self.report["skip_kept"] = _one_based(decision["skip_kept"])
        self.report["skip_unjudged"] = _one_based(decision["skip_unjudged"])
        self.report["skip_vetoed"] = _one_based(decision["skip_vetoed"])
        self.report["single_judge_skips"] = _one_based(decision["single_judge_skips"])
        self.report["credit_rejected"] = _one_based(decision["credit_rejected"])
        credits = decision["credits"]
        if credits:
            await asyncio.to_thread(pipeline.mark_skipped, chapter_id, credits, True)
        self.report["credit_pages"] = _one_based(credits)

        textless = decision["textless"]
        self.report["textless_rejected"] = _one_based(decision["textless_rejected"])
        if textless:
            await asyncio.to_thread(pipeline.mark_skipped, chapter_id, textless, True)
        self.report["textless_pages"] = _one_based(textless)

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
        self._transcripts = {scan.page_index: scan.lines for scan in scans if scan.page_index not in credits}
        self._progress(len(active), len(active),
                       f"{len(credits)} lát credit, {len(textless)} lát không chữ, {logos} logo")

    async def clean(self) -> None:
        from app.routers.chapters import chapter_processing_jobs

        indices = self._active_pages()
        self._glossary_task = asyncio.create_task(self._read_glossary(indices))
        try:
            if not indices:
                self._progress(0, 0, "Không có lát nào cần clean")
                return
            screen = self._art_screen()
            snapshot = chapter_processing_jobs.start(
                self.job.chapter_id, page_indices=indices, workers=self.settings.workers,
                **({"screen": screen} if screen is not None else {}),
            )
            job_id = snapshot["job_id"]
            self._start_review(len(indices))
            wanted, fed = set(indices), 0
            while True:
                snapshot = chapter_processing_jobs.snapshot(job_id)
                self._progress(snapshot["completed"], snapshot["total"])
                done = snapshot.get("done_indices") or []
                for page_index in done[fed:]:
                    if page_index in wanted:
                        self._review_queue.put_nowait(page_index)
                fed = len(done)
                self._check_cancel()
                if snapshot["status"] in {"completed", "failed"}:
                    break
                await asyncio.sleep(POLL_SECONDS)
            if snapshot["status"] == "failed":
                errors = "; ".join(str(item.get("message")) for item in snapshot.get("errors") or [])
                raise AIModeFailed(f"Clean thất bại: {errors or 'lỗi không rõ'}")
            self._check_cancel()
        except BaseException:
            await _cancel_and_wait(self._glossary_task)
            await _cancel_and_wait(self._review_task)
            raise
        self._review_queue.put_nowait(None)

    def _art_screen(self):
        from app.ai_mode.decider import screen_for

        screen = screen_for(self.provider, self.api_key)
        if screen is None:
            status = decider.judge_status(self.provider, self.api_key)
            logger.warning("Decider art screen unavailable (provider {}: {}); "
                           "detector boxes are erased without the lettering check",
                           status["provider"], status["reason"])
            return None
        lock = threading.Lock()

        def counted(image, boxes) -> set[int]:
            art = screen(image, boxes)
            with lock:
                self.report["art_kept"] += len(art)
            return art
        return counted

    async def _read_glossary(self, indices: list[int]) -> dict:
        from app.ai_mode.glossary import merge_glossaries, read_glossary, read_glossary_text
        from app.translation.cost import language_name

        pages = self._manifest().get("pages", [])
        gate = asyncio.Semaphore(SCAN_CONCURRENCY)

        target = language_name(self.settings.target_lang)

        def read_batch(batch: list[int]):
            if all(index in self._transcripts for index in batch):
                return read_glossary_text(self.provider, self.settings.model, self.api_key, target,
                                          self.settings.target_lang, {i: self._transcripts[i] for i in batch})
            images = [(index, read_image(validate_managed_path(pages[index]["original"], RAW_DIR / self.job.chapter_id)))
                      for index in batch]
            return read_glossary(self.provider, self.settings.model, self.api_key, target, self.settings.target_lang,
                                 images)

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

        known = [index for index in indices if index in self._transcripts]
        unknown = [index for index in indices if index not in self._transcripts]
        batches = [known[i:i + GLOSSARY_TEXT_SLICES] for i in range(0, len(known), GLOSSARY_TEXT_SLICES)]
        batches = [batch for batch in batches if any(self._transcripts[index] for index in batch)]
        size = max(1, self.settings.images_per_request)
        batches += [unknown[i:i + size] for i in range(0, len(unknown), size)]
        self.report["glossary_text_slices"] = len(known)
        self.report["glossary_image_slices"] = len(unknown)
        glossary = merge_glossaries(list(await asyncio.gather(*(read(batch) for batch in batches))))
        self.report["glossary"] = glossary
        return glossary

    def _images(self, page_index: int, second_key: str, second_root) -> tuple:
        page = self._manifest()["pages"][page_index]
        chapter_id = self.job.chapter_id
        original = read_image(validate_managed_path(page["original"], RAW_DIR / chapter_id))
        other = read_image(validate_managed_path(page[second_key], second_root / chapter_id))
        return original, other

    def _in_core(self, page_index: int, boxes) -> tuple:
        core = self._manifest()["pages"][page_index].get("stitch_core")
        if not isinstance(core, dict):
            return tuple(boxes)
        try:
            top, bottom = int(core["core_y1"]), int(core["core_y2"])
        except (KeyError, TypeError, ValueError):
            return tuple(boxes)
        return tuple(box for box in boxes if top <= (box[1] + box[3]) / 2 < bottom)

    def _page_boxes(self, page_index: int) -> list[tuple[int, int, int, int]]:
        boxes = self._manifest()["pages"][page_index].get("boxes") or []
        return [(int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])) for b in boxes
                if isinstance(b, dict) and not b.get("removed") and all(k in b for k in ("x1", "y1", "x2", "y2"))]

    def _start_review(self, total: int) -> None:
        self._review_queue = asyncio.Queue()
        self._review_task = asyncio.create_task(self._review_stream(total))

    async def _review_stream(self, total: int) -> None:
        from app.ai_mode import checkpoints

        crops: list = []
        finished = 0

        def cut(page_index: int) -> list:
            original, clean = self._images(page_index, "clean", PROCESSED_DIR)
            return checkpoints.crops_for(page_index, original, clean, self._in_core(page_index, self._page_boxes(page_index)))

        self._progress(0, total, stage="review")
        while (page_index := await self._review_queue.get()) is not None:
            try:
                crops.extend(await asyncio.to_thread(cut, page_index))
            except (HTTPException, RuntimeError, ValueError, OSError, KeyError) as exc:
                _append(self.report["qc_errors"], f"Lát {page_index + 1}: {_detail(exc)[:200]}")
            finished += 1
            self._progress(finished, total, stage="review")
        sheets = await asyncio.to_thread(checkpoints.pack, crops)
        gate = asyncio.Semaphore(REVIEW_CONCURRENCY)

        async def check(sheet, placed) -> list:
            async with gate:
                remaining = self._remaining_budget()
                if self.job.cancel_requested or (remaining is not None and remaining < 0.001):
                    return []
                try:
                    found, cost = await _ai_call(checkpoints.review_sheet, self.provider, self.settings.model,
                                                 self.api_key, sheet, placed)
                except (RuntimeError, ValueError, OSError) as exc:
                    _append(self.report["qc_errors"], f"Tờ crop: {_detail(exc)[:200]}")
                    return []
                self._add_cost(cost)
                return found

        flagged = [crop for found in await asyncio.gather(*(check(s, p) for s, p in sheets)) for crop in found]
        self.report["review_sheets"] = len(sheets)
        self.report["review_crops"] = len(crops)
        by_page: dict[int, list] = {}
        for crop in flagged:
            by_page.setdefault(crop.page_index, []).append(crop.rect)
        workers = asyncio.Semaphore(max(1, int(self.settings.workers or 1)))

        async def repaint(page_index: int, rects: list) -> None:
            async with workers:
                try:
                    if self._scan_logos.get(page_index):
                        await asyncio.to_thread(self._free_logos_with_text, page_index, rects)
                    await asyncio.to_thread(pipeline_fix, self.job.chapter_id, page_index, rects)
                    self.report["repainted_regions"] += len(rects)
                    _append(self.report["repaint_pages"], page_index + 1)
                    await self._verify_repaint(page_index, rects)
                except (HTTPException, RuntimeError, ValueError, OSError, KeyError) as exc:
                    _append(self.report["qc_errors"], f"Lát {page_index + 1}: {_detail(exc)[:200]}")

        await asyncio.gather(*(repaint(page_index, rects) for page_index, rects in sorted(by_page.items())))

    def _still_reading(self, page_index: int, rects: list) -> list:
        from app.detector.ctd_mask import still_reads

        clean = self._images(page_index, "clean", PROCESSED_DIR)[1]
        return [rect for rect in rects if still_reads(clean, tuple(int(v) for v in rect))]

    async def _verify_repaint(self, page_index: int, rects: list) -> None:
        left = await asyncio.to_thread(self._still_reading, page_index, rects)
        if left:
            self.report["repaint_retried"] += len(left)
            await asyncio.to_thread(pipeline_fix, self.job.chapter_id, page_index, left)
            left = await asyncio.to_thread(self._still_reading, page_index, left)
        for rect in left:
            self._append_page_capped(self.report["residue_left"], {"page": page_index + 1, "rect": [int(v) for v in rect]})

    def _free_logos_with_text(self, page_index: int, residue) -> None:
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
        indices = self._active_pages()
        if self._review_task is None:
            self._start_review(len(indices))
            for page_index in indices:
                self._review_queue.put_nowait(page_index)
            self._review_queue.put_nowait(None)
        await self._review_task
        self._check_cancel()
        self._progress(len(indices), len(indices),
                       f"{self.report.get('review_sheets', 0)} tờ crop, repaint {self.report['repainted_regions']}")

    async def translate(self) -> None:
        from app.ocr.language_detect import site_language_hint
        from app.routers.translation import TranslateVisionPageRequest, translate_page_in_context
        from app.translation.context import ChapterMemory

        manifest = self._manifest()
        source_lang = manifest.get("source_lang") or site_language_hint(manifest.get("source_url")) or "auto"
        self.report["source_lang"] = source_lang
        indices = self._active_pages()
        if indices and not manifest.get("setting"):
            try:
                from app.translation.vision import VisionPageTranslator
                original = validate_managed_path(manifest["pages"][indices[0]]["original"],
                                                 RAW_DIR / self.job.chapter_id)
                probe = VisionPageTranslator(self.provider, self.settings.model)
                setting = await asyncio.to_thread(probe.detect_setting, original, self.api_key)
                if setting:
                    with get_manifest_lock(self.job.chapter_id):
                        stored = load_manifest_raw(self.job.chapter_id)
                        stored["setting"] = setting
                        save_manifest_raw(self.job.chapter_id, stored)
                    manifest["setting"] = setting
                    self.report["setting"] = setting
            except Exception as exc:
                logger.warning("Setting detection failed; using the general register: {}", _detail(exc)[:150])
        try:
            glossary = await self._glossary_task if self._glossary_task else {}
        except Exception as exc:
            _append(self.report["translate_errors"], f"Bảng thuật ngữ: {_detail(exc)[:150]}")
            glossary = {}
        key = series_memory.series_key(self.settings.url)
        series = self._series = series_memory.load(key)
        glossary = series_memory.settled_glossary(series, glossary or {})
        memory = self._memory = ChapterMemory(self.settings.story_notes or series["notes"], glossary)
        for character in series["characters"]:
            memory.characters[character["name"]] = character["note"]
        self.report["series"] = {"key": key, **{k: len(series[k]) for k in ("names", "terms", "address", "characters", "chapters")}}
        slice_total = self._slice_total = len(self._manifest().get("pages", []))
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
                        translate_provider=self.settings.translate_provider,
                        translate_model=self.settings.translate_model,
                    ), memory=memory, slice_total=slice_total, skip_seam_mirrors=True)
                    run = data.get("translation_run") or {}
                except HTTPException as exc:
                    _append(self.report["translate_errors"], f"Lát {page_index + 1}: {_detail(exc)[:200]}")
                    run = None
                if run is not None:
                    await self._keep_art(page_index, run)
                    self.report["translated"] += int(run.get("translated") or 0)
                    self.report["unreadable"] += max(
                        0, int(run.get("unreadable") or 0) - len(run.get("blank_ids") or []))
                    self.report["review_flags"] += int(run.get("review") or 0)
                    self._add_cost(run.get("estimated_cost_usd") if self.provider.tracks_cost else None)
                    if run.get("render_error"):
                        _append(self.report["render_errors"], f"Lát {page_index + 1}: {run['render_error']}")
                if run is not None and run.get("unboxed"):
                    await self._add_unboxed(page_index, run["unboxed"], source_lang)
                if run is None or run.get("remaining") or run.get("missing_ids") or run.get("story_blank_ids") or run.get("stale"):
                    only = None if run is None or run.get("remaining") or run.get("stale") else (
                        set(run.get("missing_ids") or []) | set(run.get("story_blank_ids") or []))
                    await self._retry_untranslated(page_index, source_lang, only)
                finished += 1
                self._progress(finished, len(indices))

        await asyncio.to_thread(self._join_stacked_lines, indices)
        await self._create_seam_mirrors()
        self._progress(0, len(indices))
        await asyncio.gather(*(translate_one(page_index) for page_index in indices))
        self._check_cancel()
        if out_of_budget:
            self.report["translate_out_of_budget"] = True
            _append(self.report["translate_errors"], "Hết ngân sách, các lát còn lại chưa dịch")
        await self._sync_seams()
        sheet = memory.snapshot()
        self.report["characters"] = sheet["characters"][:MAX_REPORT_ITEMS]
        self.report["address"] = sheet["address"][:MAX_REPORT_ITEMS]
        self._progress(len(indices), len(indices), f"Dịch {self.report['translated']} vùng")


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
            report = runner.__dict__.get("report") or {}
            job.status = ("completed_with_blockers"
                          if report.get("editorial_blockers") or report.get("translate_out_of_budget")
                          else "completed")
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
