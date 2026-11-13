from __future__ import annotations

import asyncio
import sys

import requests
from fastapi import HTTPException

from app.ai_mode.checkpoints import _covered
from app.manifest_utils import (
    get_manifest_lock,
    invalidate_page_render,
    load_manifest_raw,
    save_manifest_raw,
)


def _j(name: str, fallback=None):
    job_mod = sys.modules.get("app.ai_mode.job")
    if job_mod is not None and hasattr(job_mod, name):
        return getattr(job_mod, name)
    return fallback


class AIModeSeamsMixin:
    """Polish, seam synchronization, unboxed recovery, and retry helpers for AIModeRunner."""

    async def polish(self) -> None:
        from app.ai_mode import polish

        stats = self.report["polish"]
        if not self.settings.polish:
            self._progress(0, 0, "Tắt")
            return
        from app.ai_mode.seams import seam_mirror_ids

        manifest = self._manifest()
        indices = self._active_pages()
        mirrors = {(index, str(obj_id)) for index in indices for obj_id in seam_mirror_ids(manifest, index)}
        lines = [
            line for line in polish.collect_lines(manifest.get("pages", []), indices)
            if (line.page_index, line.id) not in mirrors
        ]
        memory = self._memory.snapshot() if self._memory is not None else {}
        from app.translation.cost import language_name

        language = language_name(self.settings.target_lang)
        gate = asyncio.Semaphore(_j("POLISH_CONCURRENCY", 8))
        unavailable = False
        ai_call = _j("_ai_call")
        append_fn = _j("_append")
        detail_fn = _j("_detail")

        async def grade(line, text=None):
            nonlocal unavailable
            async with gate:
                if unavailable or self.job.cancel_requested:
                    return None
                try:
                    return await ai_call(
                        polish.judge, self.provider, self.api_key,
                        polish.line_state(line, text, memory, language),
                        polish.questions(line, memory, language),
                    )
                except (RuntimeError, ValueError, OSError, requests.RequestException) as exc:
                    append_fn(stats["errors"], detail_fn(exc)[:200])
                    unavailable = unavailable or " 404" in detail_fn(exc)
                    return None

        self._progress(0, len(lines))
        pending = lines
        max_kept = _j("MAX_GRADED_KEPT", 400)
        for _round in range(_j("POLISH_ROUNDS", 2)):
            if not pending or self.job.cancel_requested:
                break
            scores = await asyncio.gather(*(grade(line) for line in pending))
            judged = [(line, found) for line, found in zip(pending, scores) if found]
            for line, found in judged:
                line.scores = found
            if _round == 0:
                stats["scores"] = [
                    {"page": line.page_index + 1, "text": line.text, "scores": found}
                    for line, found in judged[:max_kept]
                ]
                stats["judged"] = len(judged)
                stats["flagged"] = sum(bool(polish.problems(line.scores)) for line, _ in judged)
            flagged = [line for line, _ in judged if polish.problems(line.scores)]
            if not flagged or self.job.cancel_requested:
                break
            batches = [flagged[i:i + polish.REWRITE_BATCH] for i in range(0, len(flagged), polish.REWRITE_BATCH)]
            kept, approved = [], []
            for batch in batches:
                try:
                    texts, cost = await ai_call(
                        polish.rewrite, self.provider, self.settings.model, self.api_key,
                        batch, memory, language,
                    )
                except (RuntimeError, ValueError, OSError) as exc:
                    append_fn(stats["errors"], detail_fn(exc)[:200])
                    continue
                self._add_cost(cost if self.provider.tracks_cost else None)
                candidates = [(line, texts[line.id]) for line in batch if texts.get(line.id, line.text) != line.text]
                regraded = await asyncio.gather(*(grade(line, text) for line, text in candidates))
                for (line, text), found in zip(candidates, regraded):
                    if found and polish.better(found, line.scores):
                        approved.append((line, text, line.text, line.scores, found))
                        line.text, line.scores = text, found
                        kept.append(line)
            saved = await asyncio.to_thread(self._save_polished, kept)
            if isinstance(saved, tuple):
                committed_ids = saved[0]
                stats["rewritten"] += len(committed_ids)
            else:
                committed_ids = {line.id for line in kept}
                stats["rewritten"] += len(kept)
            for line, text, before, scores_before, found in approved:
                if line.id in committed_ids:
                    append_fn(
                        stats["samples"],
                        {
                            "page": line.page_index + 1, "source": line.source,
                            "before": before, "after": text,
                            "scores_before": scores_before, "scores_after": found,
                        },
                    )
            pending = [line for line in flagged if polish.problems(line.scores)]
            self._progress(len(lines), len(lines))
        stats["still_flagged"] = sum(bool(polish.problems(line.scores)) for line in lines if line.scores)
        if stats["rewritten"]:
            await self._sync_seams()
        self._progress(len(lines), len(lines), f"Chấm {stats['judged']} câu, sửa {stats['rewritten']}")

    def _save_polished(self, lines) -> tuple[set[str], int]:
        from app.dependencies import pipeline
        from app.translation.vision import VisionTranslationResult, _drop_foreign_script, collapse_repeated_lines

        if not lines:
            return (set(), 0)
        stats = self.report.get("polish") or {}
        committed: set[str] = set()
        dropped = 0
        touched: set[int] = set()
        append_fn = _j("_append")
        with _j("get_manifest_lock", get_manifest_lock)(self.job.chapter_id):
            manifest = _j("load_manifest_raw", load_manifest_raw)(self.job.chapter_id)
            for line in lines:
                page = manifest["pages"][line.page_index]
                obj = next(
                    (o for o in page.get("text_objects") or []
                     if isinstance(o, dict) and str(o.get("id")) == line.id),
                    None,
                )
                if obj is None:
                    continue
                text = collapse_repeated_lines(line.text).strip()
                probe = VisionTranslationResult(
                    translations={line.id: text}, model="",
                    usage={}, estimated_cost_usd=None,
                )
                probe, _ = _drop_foreign_script(
                    probe, {"translations": [{"id": line.id, "source": line.source}]}
                )
                filtered = (probe.translations.get(line.id) or "").strip()
                if not filtered:
                    line.text = str(obj.get("translation") or "")
                    dropped += 1
                    stats["gate_dropped"] = stats.get("gate_dropped", 0) + 1
                    append_fn(
                        stats.setdefault("gate_dropped_samples", []),
                        {
                            "page": line.page_index + 1, "id": line.id,
                            "before": obj.get("translation"), "dropped": text,
                        },
                    )
                    continue
                line.text = filtered
                obj.setdefault("polished_from", obj.get("translation"))
                obj["translation"] = obj["auto_translation"] = filtered
                invalidate_page_render(manifest, line.page_index)
                touched.add(line.page_index)
                committed.add(line.id)
            _j("save_manifest_raw", save_manifest_raw)(self.job.chapter_id, manifest)
            if touched:
                pipeline._sync_output_dir(self.job.chapter_id, manifest, sorted(touched))
        return (committed, dropped)

    async def _add_unboxed(self, page_index: int, boxes: list, source_lang: str) -> None:
        from app.ai_mode.checkpoints import MARGIN_PX, MAX_AREA_RATIO, MIN_SIDE_PX
        from app.dependencies import pipeline

        page = self._manifest()["pages"][page_index]
        width, height = int(page.get("width") or 0), int(page.get("height") or 0)
        existing = self._page_boxes(page_index)
        fresh = []
        for x1, y1, x2, y2 in self._in_core(page_index, [tuple(map(int, box)) for box in boxes]):
            if (x2 - x1 < MIN_SIDE_PX or y2 - y1 < MIN_SIDE_PX
                    or (width and height and (x2 - x1) * (y2 - y1) > MAX_AREA_RATIO * width * height)):
                continue
            bx2 = min(width, x2 + MARGIN_PX) if width > 0 else x2 + MARGIN_PX
            by2 = min(height, y2 + MARGIN_PX) if height > 0 else y2 + MARGIN_PX
            box = (max(0, x1 - MARGIN_PX), max(0, y1 - MARGIN_PX), bx2, by2)
            if not _covered(box, existing + fresh):
                fresh.append(box)
        if not fresh:
            return
        before = {str(obj.get("id")) for obj in page.get("text_objects") or [] if isinstance(obj, dict)}
        try:
            await asyncio.to_thread(
                pipeline.apply_review_fixes, self.job.chapter_id, page_index,
                boxes=fresh, skip_covered=True,
            )
        except (HTTPException, RuntimeError, ValueError, OSError) as exc:
            _j("_append")(
                self.report["translate_errors"],
                f"Lát {page_index + 1}: thêm chữ sót thất bại: {_j('_detail')(exc)[:150]}",
            )
            return
        await asyncio.to_thread(self._ensure_objects, page_index)
        now = {
            str(obj.get("id")) for obj in self._manifest()["pages"][page_index].get("text_objects") or []
            if isinstance(obj, dict)
        }
        if now - before:
            self.report["missed_added"] += len(now - before)
            _j("_append")(self.report["unboxed_pages"], page_index + 1)
            await self._retry_untranslated(page_index, source_lang, now - before)

    async def _create_seam_mirrors(self) -> list[tuple[int, str]]:
        from app.ai_mode.seams import create_seam_mirrors

        def create() -> tuple[list[tuple[int, str]], dict[int, list[tuple[int, int, int, int]]]]:
            with _j("get_manifest_lock", get_manifest_lock)(self.job.chapter_id):
                manifest = _j("load_manifest_raw", load_manifest_raw)(self.job.chapter_id)
                created = create_seam_mirrors(manifest)
                rects: dict[int, list[tuple[int, int, int, int]]] = {}
                for page_index, obj_id in created:
                    obj = next(
                        (o for o in manifest["pages"][page_index].get("text_objects") or []
                         if isinstance(o, dict) and str(o.get("id")) == obj_id),
                        None,
                    )
                    if obj is None or not isinstance(obj.get("region"), dict):
                        continue
                    region = obj["region"]
                    rects.setdefault(page_index, []).append(
                        (int(region["x1"]), int(region["y1"]), int(region["x2"]), int(region["y2"]))
                    )
                if created:
                    _j("save_manifest_raw", save_manifest_raw)(self.job.chapter_id, manifest)
                return created, rects

        created, rects = await asyncio.to_thread(create)
        for page_index, page_rects in sorted(rects.items()):
            try:
                await asyncio.to_thread(_j("pipeline_fix"), self.job.chapter_id, page_index, page_rects)
            except (HTTPException, RuntimeError, ValueError, OSError) as exc:
                _j("_append")(
                    self.report["translate_errors"],
                    f"Lát {page_index + 1}: xóa chữ chờ ghép thất bại: {_j('_detail')(exc)[:150]}",
                )
        if created:
            self.report["seam_mirrors_created"] = self.report.get("seam_mirrors_created", 0) + len(created)
        return created

    async def _sync_seams(self) -> list[int]:
        from app.ai_mode.seams import drop_duplicated_leading_text, drop_overlapping_letters, sync_seam_mirrors

        def sync() -> list[int]:
            with _j("get_manifest_lock", get_manifest_lock)(self.job.chapter_id):
                manifest = _j("load_manifest_raw", load_manifest_raw)(self.job.chapter_id)
                dropped = drop_overlapping_letters(manifest)
                self.report["overlaps_dropped"] = self.report.get("overlaps_dropped", 0) + len(dropped)
                changed = sorted(set(sync_seam_mirrors(manifest)) | {page_index for page_index, _ in dropped})
                trimmed = drop_duplicated_leading_text(manifest)
                if trimmed:
                    self.report["duplicated_leading_trimmed"] = self.report.get(
                        "duplicated_leading_trimmed", 0
                    ) + len(trimmed)
                changed = sorted(set(changed) | set(trimmed))
                for page_index in changed:
                    invalidate_page_render(manifest, page_index)
                if changed:
                    _j("save_manifest_raw", save_manifest_raw)(self.job.chapter_id, manifest)
                    from app.dependencies import pipeline
                    pipeline._sync_output_dir(self.job.chapter_id, manifest, changed)
                return changed

        changed = await asyncio.to_thread(sync)
        self.report["seam_copies"] = self.report.get("seam_copies", 0) + len(changed)
        return changed

    async def _translate_retry(self, page_index: int, source_lang: str, object_ids: list[str] | None) -> dict:
        from app.routers.translation import TranslateVisionPageRequest, translate_page_in_context

        remaining = self._remaining_budget()
        data = await translate_page_in_context(
            TranslateVisionPageRequest(
                chapter_id=self.job.chapter_id, page_index=page_index,
                source_lang=source_lang, target_lang=self.settings.target_lang,
                budget_usd=0.25 if remaining is None else max(0.001, min(0.25, remaining)),
                provider=self.provider.id, model=self.settings.model,
                translate_provider=self.settings.translate_provider,
                translate_model=self.settings.translate_model,
                max_objects=_j("RETRY_BATCH", 4), object_ids=object_ids,
            ),
            memory=self._memory, slice_total=self._slice_total, skip_seam_mirrors=True,
        )
        run = data.get("translation_run") or {}
        self.report["translated"] += int(run.get("translated") or 0)
        self._add_cost(run.get("estimated_cost_usd") if self.provider.tracks_cost else None)
        await self._keep_art(page_index, run)
        return run

    async def _keep_art(self, page_index: int, run: dict) -> None:
        from app.dependencies import pipeline
        from app.region_policy import text_object_in_preserve_region
        from app.translation.vision import drawn_in_art

        def corroborated(obj: dict) -> bool:
            source = str(obj.get("source_read") or obj.get("ocr_text") or "")
            if drawn_in_art(source):
                return True
            return str(obj.get("container") or "").strip().lower() == "free"

        art_regions = list(run.get("art_regions") or [])
        regions = list(art_regions)
        story_blanks = set(run.get("story_blank_ids") or [])
        blank_ids = run.get("blank_ids") or []
        restored: list[dict] = []
        reclassified: list[str] = []

        def region_key(region):
            return tuple(sorted((str(k), v) for k, v in region.items())) if isinstance(region, dict) else None

        manifest = None
        if art_regions or blank_ids:
            manifest = self._manifest()
            page = manifest["pages"][page_index]
            by_region = {
                region_key(obj["region"]): obj
                for obj in page.get("text_objects") or []
                if isinstance(obj, dict) and isinstance(obj.get("region"), dict)
            }
            for region in art_regions:
                obj = by_region.get(region_key(region))
                if obj is not None:
                    restored.append(obj)
        if blank_ids:
            for obj in page.get("text_objects") or []:
                if (isinstance(obj, dict) and str(obj.get("id")) in blank_ids
                        and str(obj.get("id")) not in story_blanks
                        and isinstance(obj.get("region"), dict)
                        and not text_object_in_preserve_region(page, obj)):
                    if corroborated(obj):
                        regions.append(dict(obj["region"]))
                        restored.append(obj)
                    else:
                        reclassified.append(str(obj["id"]))
        if reclassified:
            story = run.get("story_blank_ids")
            if not isinstance(story, list):
                story = list(story or [])
                run["story_blank_ids"] = story
            for obj_id in reclassified:
                if obj_id not in story:
                    story.append(obj_id)
        if not regions:
            return
        seen: set[tuple] = set()
        unique = []
        for region in regions:
            key = region_key(region)
            if key not in seen:
                seen.add(key)
                unique.append(region)
        regions = unique
        art_unique = len({region_key(region) for region in art_regions})
        try:
            await asyncio.to_thread(pipeline.preserve_and_reinpaint, self.job.chapter_id, page_index, regions)
            self.report["sfx_kept"] += len(art_regions)
            self.report["restored_regions"] += len(regions) - art_unique
            for obj in restored:
                key = (page_index, str(obj.get("id")))
                if key not in self._art_restore_reviewed:
                    self._art_restore_reviewed.add(key)
                    self._append_page_capped(self.report["review_list"], {
                        "page": page_index + 1, "id": obj.get("id"),
                        "reason": "Model bỏ trống (SFX/nghệ thuật); đã giữ ảnh gốc",
                    })
        except (ValueError, RuntimeError, OSError) as exc:
            _j("_append")(self.report["translate_errors"], f"Lát {page_index + 1}: giữ SFX thất bại: {_j('_detail')(exc)[:150]}")
            for obj in restored:
                self._append_page_capped(self.report["review_list"], {
                    "page": page_index + 1, "id": obj.get("id"),
                    "reason": "Model bỏ trống (SFX/nghệ thuật); giữ ảnh gốc thất bại — vùng này vẫn trống",
                })

    def _join_stacked_lines(self, indices: list[int]) -> None:
        from app.ai_mode.seams import join_stacked_lines, set_letter_bounds
        from app.text_objects import ensure_page_text_objects

        with _j("get_manifest_lock", get_manifest_lock)(self.job.chapter_id):
            manifest = _j("load_manifest_raw", load_manifest_raw)(self.job.chapter_id)
            for page_index in indices:
                ensure_page_text_objects(manifest["pages"][page_index])
                set_letter_bounds(manifest["pages"][page_index])
            self.report["lines_joined"] = join_stacked_lines(manifest, indices)
            _j("save_manifest_raw", save_manifest_raw)(self.job.chapter_id, manifest)

    def _ensure_objects(self, page_index: int) -> None:
        from app.text_objects import ensure_page_text_objects

        with _j("get_manifest_lock", get_manifest_lock)(self.job.chapter_id):
            manifest = _j("load_manifest_raw", load_manifest_raw)(self.job.chapter_id)
            _, changed = ensure_page_text_objects(manifest["pages"][page_index])
            if changed:
                _j("save_manifest_raw", save_manifest_raw)(self.job.chapter_id, manifest)

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
        from app.dependencies import pipeline

        await asyncio.to_thread(self._ensure_objects, page_index)
        blank: set[str] = set()
        _j("_append")(self.report["retried_pages"], page_index + 1)
        fruitless = 0
        retry_rounds = _j("RETRY_ROUNDS", 3)
        retry_batch = _j("RETRY_BATCH", 4)
        while fruitless < retry_rounds and not self.job.cancel_requested:
            targets = self._untranslated(page_index, only, blank)
            remaining = self._remaining_budget()
            if not targets or (remaining is not None and remaining < 0.001):
                break
            batch = [str(obj["id"]) for obj in targets[:retry_batch]]
            try:
                run = await self._translate_retry(page_index, source_lang, batch)
            except HTTPException as exc:
                _j("_append")(self.report["translate_errors"], f"Lát {page_index + 1} (thử lại): {_j('_detail')(exc)[:200]}")
                fruitless += 1
                continue
            blank.update(set(run.get("blank_ids") or []) - set(run.get("story_blank_ids") or []))
            if set(batch) <= {str(obj["id"]) for obj in self._untranslated(page_index, only, blank)}:
                fruitless += 1
        unresolved = self._untranslated(page_index, only, blank)
        if not unresolved:
            return
        regions = [dict(obj["region"]) for obj in unresolved]
        restored = True
        try:
            await asyncio.to_thread(pipeline.preserve_and_reinpaint, self.job.chapter_id, page_index, regions)
            self.report["restored_regions"] += len(regions)
        except (ValueError, RuntimeError, OSError) as exc:
            restored = False
            _j("_append")(self.report["translate_errors"], f"Lát {page_index + 1}: khôi phục thất bại: {_j('_detail')(exc)[:150]}")
        for obj in unresolved:
            self._append_page_capped(self.report["review_list"], {
                "page": page_index + 1, "id": obj["id"],
                "reason": ("AI không dịch được sau khi thử lại; đã giữ ảnh gốc" if restored
                           else "AI không dịch được sau khi thử lại; giữ ảnh gốc thất bại — vùng này vẫn trống"),
            })

    def _settle_blanks(self) -> int:
        from app.editorial_gate import UNTRANSLATED_DISPOSITION
        from app.region_policy import text_object_in_preserve_region

        chapter_id = self.job.chapter_id
        settled = 0
        has_evidence = _j("_object_has_text_evidence")
        with _j("get_manifest_lock", get_manifest_lock)(chapter_id):
            manifest = _j("load_manifest_raw", load_manifest_raw)(chapter_id)
            for page in manifest.get("pages") or []:
                if not isinstance(page, dict) or page.get("skipped"):
                    continue
                for obj in page.get("text_objects") or []:
                    if (not isinstance(obj, dict) or obj.get("source_missing") or obj.get("joined_into")
                            or obj.get("overlap_dropped") or obj.get("editorial_reviewed")
                            or obj.get("seam_owner")
                            or str(obj.get("translation") or "").strip() or text_object_in_preserve_region(page, obj)):
                        continue
                    if obj.get("translation_attempted"):
                        obj["editorial_disposition"], obj["editorial_reviewed"] = "non_story", True
                    elif not has_evidence(obj):
                        obj["editorial_disposition"], obj["editorial_reviewed"] = "non_story", True
                    else:
                        obj["editorial_disposition"] = UNTRANSLATED_DISPOSITION
                    settled += 1
            if settled:
                _j("save_manifest_raw", save_manifest_raw)(chapter_id, manifest)
        return settled
