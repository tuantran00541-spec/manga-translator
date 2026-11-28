from __future__ import annotations

import asyncio
import json

from fastapi import HTTPException

from app.ai_mode.job_compat import _j
from app import series_memory
from app.config import OUTPUT_DIR, PROCESSED_DIR
from app.manifest_utils import (
    get_manifest_lock,
    invalidate_page_render,
    load_manifest_raw,
    save_manifest_raw,
)


class AIModeRenderingMixin:
    """Render stage, skipped-object restore loop, and finalization for AIModeRunner."""

    def _invalidate_renders(self, indices: list[int]) -> None:
        from app.dependencies import pipeline
        from app.manifest_utils import invalidate_page_render, save_manifest_raw as current_save_manifest_raw

        with _j("get_manifest_lock", get_manifest_lock)(self.job.chapter_id):
            manifest = _j("load_manifest_raw", load_manifest_raw)(self.job.chapter_id)
            for page_index in indices:
                invalidate_page_render(manifest, page_index)
            current_save_manifest_raw(self.job.chapter_id, manifest)
            pipeline._sync_output_dir(self.job.chapter_id, manifest, indices)

    async def render(self) -> None:
        from app.routers.image import _current_rendered_path

        chapter_id = self.job.chapter_id
        self.report["blanks_settled"] = await asyncio.to_thread(self._settle_blanks)
        indices = self._active_pages()
        total = len(indices)
        self._progress(0, total)
        gate = asyncio.Semaphore(min(4, max(1, int(getattr(self.settings, "workers", 2) or 2))))
        done = 0

        async def _run_page(page_index: int) -> None:
            nonlocal done
            async with gate:
                self._check_cancel()
                manifest = self._manifest()
                page = manifest["pages"][page_index]
                if not (
                    page.get("skipped")
                    or (
                        not page.get("render_skipped_objects")
                        and _current_rendered_path(chapter_id, page_index, manifest) is not None
                    )
                ):
                    await self._render_one(page_index)
            done += 1
            self._progress(done, total)

        await asyncio.gather(*(_run_page(page_index) for page_index in indices))
        self._progress(total, total)

    def _clear_restored_translations(self, page_index: int, object_ids: set[str]) -> None:
        if not object_ids:
            return
        from app.text_objects import clear_page_object_translations

        mem_manifest = self._manifest()
        mem_pages = mem_manifest.get("pages") or [] if isinstance(mem_manifest, dict) else []
        if 0 <= page_index < len(mem_pages):
            clear_page_object_translations(mem_pages[page_index], object_ids)

        chapter_id = self.job.chapter_id
        processed_dir = _j("PROCESSED_DIR", PROCESSED_DIR)
        if not (processed_dir / chapter_id / "manifest.json").exists():
            return
        with _j("get_manifest_lock", get_manifest_lock)(chapter_id):
            manifest = _j("load_manifest_raw", load_manifest_raw)(chapter_id)
            pages = manifest.get("pages") or []
            if 0 <= page_index < len(pages) and clear_page_object_translations(pages[page_index], object_ids):
                _j("save_manifest_raw", save_manifest_raw)(chapter_id, manifest)

    async def _render_one(self, page_index: int) -> None:
        from app.dependencies import pipeline
        from app.routers.export import _render_request_from_page
        from app.routers.render_commit import render_page

        chapter_id = self.job.chapter_id
        await asyncio.to_thread(self._ensure_objects, page_index)
        append_fn = _j("_append")
        detail_fn = _j("_detail")
        restore_attempts = _j("RENDER_RESTORE_ATTEMPTS", 3)
        failed_pattern = _j("RENDER_FAILED_OBJECT")

        async def restore_skipped(skipped_ids: list[str]) -> tuple[bool, list[dict]]:
            page = self._manifest()["pages"][page_index]
            regions = []
            objs = []
            for item in page.get("text_objects") or []:
                if (isinstance(item, dict) and str(item.get("id")) in skipped_ids
                        and isinstance(item.get("region"), dict)):
                    regions.append(dict(item["region"]))
                    objs.append(item)
            if not regions:
                return True, []
            try:
                await asyncio.to_thread(pipeline.preserve_and_reinpaint, chapter_id, page_index, regions)
            except (ValueError, RuntimeError, OSError) as exc:
                append_fn(self.report["render_errors"], f"Lát {page_index + 1}: khôi phục thất bại: {detail_fn(exc)[:150]}")
                return False, objs
            await asyncio.to_thread(self._clear_restored_translations, page_index, {str(o["id"]) for o in objs})
            self.report["restored_regions"] += len(regions)
            for obj in objs:
                self._append_page_capped(self.report["review_list"], {
                    "page": page_index + 1, "id": obj["id"],
                    "reason": "Bản dịch không vừa vùng chữ; đã giữ ảnh gốc",
                })
            return True, objs

        async def render_current_page():
            for conflict_attempt in range(4):
                page = self._manifest()["pages"][page_index]
                try:
                    return await asyncio.to_thread(
                        render_page, _render_request_from_page(chapter_id, page_index, page))
                except HTTPException as exc:
                    if exc.status_code != 409 or conflict_attempt == 3:
                        raise
                    await asyncio.sleep(0.05 * (conflict_attempt + 1))

        def restore_candidates(result):
            if not isinstance(result, dict):
                return []
            skipped = [str(oid) for oid in result.get("skipped_objects") or []]
            issues = result.get("layout_issue_types") or {}
            page = self._manifest()["pages"][page_index]
            for obj in page.get("text_objects") or []:
                if not isinstance(obj, dict):
                    continue
                oid = str(obj.get("id") or "")
                if (issues.get(oid) in ("balloon_overset", "placement_overset")
                        and obj.get("auto_generated") and obj.get("translation")
                        and obj.get("translation") == obj.get("auto_translation")
                        and oid not in skipped):
                    skipped.append(oid)
            return skipped

        attempt = 0
        while attempt <= restore_attempts:
            page = self._manifest()["pages"][page_index]
            try:
                result = await render_current_page()
                skipped = restore_candidates(result)
                if not skipped:
                    return
                ok, objs = await restore_skipped(skipped)
                if not ok or not objs:
                    return
                if attempt == restore_attempts:
                    page = self._manifest()["pages"][page_index]
                    rerender = await render_current_page()
                    rerender_skipped = restore_candidates(rerender)
                    if not rerender_skipped:
                        return
                    final_ok, final_objs = await restore_skipped(rerender_skipped)
                    if final_ok and final_objs:
                        page = self._manifest()["pages"][page_index]
                        final_res = await render_current_page()
                        rem = restore_candidates(final_res)
                        for oid in rem:
                            self._append_page_capped(self.report["review_list"], {
                                "page": page_index + 1, "id": str(oid),
                                "reason": "Bản dịch không vừa vùng chữ; đã hết lần thử lại — chưa giữ ảnh gốc",
                            })
                    elif not final_ok:
                        for oid in rerender_skipped:
                            self._append_page_capped(self.report["review_list"], {
                                "page": page_index + 1, "id": str(oid),
                                "reason": "Bản dịch không vừa vùng chữ; đã hết lần thử lại — chưa giữ ảnh gốc",
                            })
                    return
                attempt += 1
                continue
            except HTTPException as exc:
                detail = detail_fn(exc)
                append_fn(self.report["render_errors"], f"Lát {page_index + 1}: {detail[:200]}")
                failed = failed_pattern.search(detail)
                obj = next((
                    item for item in page.get("text_objects") or []
                    if failed and isinstance(item, dict) and str(item.get("id")) == failed.group(1)
                    and isinstance(item.get("region"), dict)
                ), None)
                if obj is None or attempt == restore_attempts:
                    return
            try:
                await asyncio.to_thread(pipeline.preserve_and_reinpaint, chapter_id, page_index, [dict(obj["region"])])
            except (ValueError, RuntimeError, OSError) as exc:
                append_fn(self.report["render_errors"], f"Lát {page_index + 1}: khôi phục thất bại: {detail_fn(exc)[:150]}")
                return
            await asyncio.to_thread(self._clear_restored_translations, page_index, {str(obj["id"])})
            self.report["restored_regions"] += 1
            attempt += 1
            self._append_page_capped(self.report["review_list"], {
                "page": page_index + 1, "id": obj["id"],
                "reason": "Bản dịch không vừa vùng chữ; đã giữ ảnh gốc",
            })

    async def finish(self) -> None:
        from app.routers.export import _preflight_copy
        from app.routers.render_commit import layout_conflict_reviews

        chapter_id = self.job.chapter_id
        with _j("get_manifest_lock", get_manifest_lock)(chapter_id):
            manifest = _j("load_manifest_raw", load_manifest_raw)(chapter_id)
            manifest["workflow"] = {"stage": "review", "page_index": 0}
            _j("save_manifest_raw", save_manifest_raw)(chapter_id, manifest)
        for review in layout_conflict_reviews(manifest):
            self._append_page_capped(self.report["review_list"], review)
        preflight = _preflight_copy(manifest, require_final_approval=False)
        self.report["editorial_blockers"] = int(preflight.get("blocker_count") or 0)
        self.report["blocker_samples"] = [
            {"kind": item.get("kind"), "page": int(item.get("page_index", 0)) + 1, "reason": item.get("reason")}
            for item in (preflight.get("blockers") or [])[:10]
        ]
        key = series_memory.series_key(self.settings.url)
        if key and self._memory is not None:
            snapshot = self._memory.snapshot(settled=False)
            glossary = self.report.get("glossary") or {}

            def learn_current(current: dict) -> dict:
                return series_memory.learn(
                    current, glossary, snapshot["characters"], snapshot["address"],
                    self.settings.url, self.settings.story_notes,
                )

            series_memory.update(key, learn_current)
        output_dir = _j("OUTPUT_DIR", OUTPUT_DIR)
        report_path = output_dir / chapter_id / "ai_mode_report.json"
        self.report["manifest"] = f"data/processed/{chapter_id}/manifest.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(self.report, ensure_ascii=False, indent=1), encoding="utf-8")
        self._progress(1, 1, "Xong")
