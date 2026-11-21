from __future__ import annotations

import json
from pathlib import Path

from fastapi import HTTPException

from app import series_memory
from app.agent.tools import ToolError
from app.config import BASE_DIR
from app.security import contains_secret_token, validate_chapter_id, validate_url


def _json(data) -> str:
    return json.dumps(data, ensure_ascii=False)


def _rel(path_str: str | None) -> str:
    if not path_str:
        return ""
    try:
        path = Path(path_str).resolve(strict=False)
        return path.relative_to(BASE_DIR.resolve(strict=False)).as_posix()
    except (OSError, ValueError):
        return str(path_str)


def _http_err(exc: HTTPException) -> ToolError:
    detail = exc.detail if isinstance(exc.detail, str) else json.dumps(exc.detail, ensure_ascii=False)
    return ToolError(f"HTTP {exc.status_code}: {detail}")


def _chapters(_session, _args: dict) -> str:
    from app.routers.chapters import list_chapters

    rows = list_chapters()
    return _json([
        {
            "chapter_id": row.get("chapter_id"),
            "chapter_name": row.get("chapter_name") or "",
            "source_url": row.get("source_url") or "",
            "total_pages": int(row.get("total_pages") or 0),
            "workflow": row.get("workflow") or {},
            "updated_at": row.get("updated_at"),
        }
        for row in rows
    ])


def _page(_session, args: dict) -> str:
    from app.manifest_utils import load_manifest_raw
    from app.text_objects import ensure_page_text_objects

    chapter_id = str(args.get("chapter_id") or "").strip()
    try:
        validate_chapter_id(chapter_id)
        manifest = load_manifest_raw(chapter_id)
    except HTTPException as exc:
        raise _http_err(exc) from exc

    pages = manifest.get("pages") or []
    raw_idx = args.get("page_index")
    if raw_idx is None:
        summary = []
        for idx, page in enumerate(pages):
            if not isinstance(page, dict):
                continue
            objs = [o for o in (page.get("text_objects") or []) if isinstance(o, dict)]
            translated = sum(1 for o in objs if str(o.get("translation") or "").strip())
            summary.append({
                "page_index": idx,
                "skipped": bool(page.get("skipped")),
                "objects": len(objs),
                "translated": translated,
                "rendered": bool(page.get("rendered")),
                "render_skipped_objects": list(page.get("render_skipped_objects") or []),
            })
        return _json({
            "chapter_id": chapter_id,
            "chapter_name": manifest.get("chapter_name") or "",
            "source_url": manifest.get("source_url") or "",
            "workflow": manifest.get("workflow") or {},
            "total_pages": len(pages),
            "pages": summary,
        })

    try:
        page_index = int(raw_idx)
    except (TypeError, ValueError) as exc:
        raise ToolError("page_index must be an integer") from exc
    if not 0 <= page_index < len(pages):
        raise ToolError(f"page_index {page_index} out of range (0..{len(pages) - 1})")

    page = dict(pages[page_index])
    ensure_page_text_objects(page)
    objects = []
    for obj in page.get("text_objects") or []:
        if not isinstance(obj, dict):
            continue
        objects.append({
            "id": str(obj.get("id") or ""),
            "ocr_text": str(obj.get("ocr_text") or ""),
            "source_read": str(obj.get("source_read") or ""),
            "translation": str(obj.get("translation") or ""),
            "role": str(obj.get("typography_role") or ""),
            "speaker": str(obj.get("speaker") or ""),
            "region": obj.get("region"),
            "editorial_disposition": obj.get("editorial_disposition"),
            "editorial_reviewed": bool(obj.get("editorial_reviewed")),
            "source_missing": bool(obj.get("source_missing")),
        })
    return _json({
        "chapter_id": chapter_id,
        "page_index": page_index,
        "width": page.get("width"),
        "height": page.get("height"),
        "skipped": bool(page.get("skipped")),
        "original_path": _rel(page.get("original")),
        "cleaned_path": _rel(page.get("cleaned")),
        "rendered_path": _rel(page.get("rendered")),
        "render_skipped_objects": list(page.get("render_skipped_objects") or []),
        "text_objects": objects,
    })


def _preflight(_session, args: dict) -> str:
    from app.manifest_utils import load_manifest_raw
    from app.routers.export import _preflight_copy

    chapter_id = str(args.get("chapter_id") or "").strip()
    require_final = bool(args.get("require_final_approval", False))
    try:
        validate_chapter_id(chapter_id)
        manifest = load_manifest_raw(chapter_id)
    except HTTPException as exc:
        raise _http_err(exc) from exc
    result = _preflight_copy(manifest, require_final_approval=require_final)
    return _json({
        "chapter_id": chapter_id,
        "ready": bool(result.get("ready")),
        "blocker_count": int(result.get("blocker_count") or 0),
        "counts": result.get("counts") or {},
        "blockers": (result.get("blockers") or [])[:50],
    })


def _update_text(_session, args: dict) -> str:
    from app.dependencies import pipeline
    from app.manifest_utils import get_manifest_lock, invalidate_page_render, load_manifest_raw, save_manifest_raw

    chapter_id = str(args.get("chapter_id") or "").strip()
    updates = args.get("updates")
    if not isinstance(updates, list) or not updates:
        raise ToolError("updates must be a non-empty list of {page_index, id, translation?, ocr_text?, editorial_disposition?, editorial_reviewed?}")
    if len(updates) > 200:
        raise ToolError("at most 200 text object updates per call")

    try:
        validate_chapter_id(chapter_id)
    except HTTPException as exc:
        raise _http_err(exc) from exc

    for item in updates:
        if not isinstance(item, dict):
            raise ToolError("each update must be an object")
        for field in ("translation", "ocr_text"):
            val = item.get(field)
            if val is not None and contains_secret_token(str(val)):
                raise ToolError(f"{field} must not contain API keys or credentials")

    updated_count = 0
    touched_pages: set[int] = set()
    try:
        for item in updates:
            page_index = int(item["page_index"])
            obj_id = str(item["id"]).strip()
            if not obj_id:
                raise ToolError("each update needs a non-empty id")
            changes: dict = {}
            if "translation" in item and item["translation"] is not None:
                changes["translation"] = str(item["translation"])
            if "ocr_text" in item and item["ocr_text"] is not None:
                changes["ocr_text"] = str(item["ocr_text"])
            if changes:
                pipeline.update_text_object(chapter_id, page_index, obj_id, changes)
                updated_count += 1
                touched_pages.add(page_index)
            if "editorial_disposition" in item or "editorial_reviewed" in item:
                with get_manifest_lock(chapter_id):
                    manifest = load_manifest_raw(chapter_id)
                    pages = manifest.get("pages") or []
                    if not 0 <= page_index < len(pages):
                        raise ToolError(f"page_index {page_index} out of range")
                    obj = next(
                        (o for o in (pages[page_index].get("text_objects") or [])
                         if isinstance(o, dict) and str(o.get("id")) == obj_id),
                        None,
                    )
                    if obj is None:
                        raise ToolError(f"text object {obj_id!r} not found on page {page_index}")
                    if "editorial_disposition" in item:
                        obj["editorial_disposition"] = item["editorial_disposition"]
                    if "editorial_reviewed" in item:
                        obj["editorial_reviewed"] = bool(item["editorial_reviewed"])
                    invalidate_page_render(manifest, page_index)
                    save_manifest_raw(chapter_id, manifest)
                    if not changes:
                        updated_count += 1
                    touched_pages.add(page_index)
    except HTTPException as exc:
        raise _http_err(exc) from exc
    except (KeyError, TypeError, ValueError) as exc:
        raise ToolError(str(exc)) from exc

    return _json({"chapter_id": chapter_id, "updated": updated_count, "pages": sorted(touched_pages)})


def _render_page(_session, args: dict) -> str:
    from app.manifest_utils import load_manifest_raw
    from app.routers.export import _render_request_from_page
    from app.routers.render_commit import render_page

    chapter_id = str(args.get("chapter_id") or "").strip()
    raw_indices = args.get("page_indices")
    if raw_indices is None and args.get("page_index") is not None:
        raw_indices = [args["page_index"]]
    if not isinstance(raw_indices, list) or not raw_indices:
        raise ToolError("provide page_index or a non-empty page_indices list")
    if len(raw_indices) > 50:
        raise ToolError("at most 50 pages per render call")

    try:
        validate_chapter_id(chapter_id)
        manifest = load_manifest_raw(chapter_id)
    except HTTPException as exc:
        raise _http_err(exc) from exc

    pages = manifest.get("pages") or []
    results = []
    for raw_idx in raw_indices:
        try:
            page_index = int(raw_idx)
        except (TypeError, ValueError) as exc:
            raise ToolError("page indices must be integers") from exc
        if not 0 <= page_index < len(pages):
            raise ToolError(f"page_index {page_index} out of range (0..{len(pages) - 1})")
        page = pages[page_index]
        if page.get("skipped"):
            results.append({"page_index": page_index, "skipped": True})
            continue
        try:
            res = render_page(_render_request_from_page(chapter_id, page_index, page))
        except HTTPException as exc:
            raise _http_err(exc) from exc
        results.append({
            "page_index": page_index,
            "skipped_objects": list(res.get("skipped_objects") or []) if isinstance(res, dict) else [],
            "rendered_path": _rel(res.get("rendered") if isinstance(res, dict) else None),
        })
    return _json({"chapter_id": chapter_id, "results": results})


def _skip_pages(_session, args: dict) -> str:
    from app.dependencies import pipeline

    chapter_id = str(args.get("chapter_id") or "").strip()
    indices = args.get("page_indices")
    skipped = bool(args.get("skipped", True))
    if not isinstance(indices, list) or not indices:
        raise ToolError("page_indices must be a non-empty list of integers")
    try:
        validate_chapter_id(chapter_id)
        parsed = [int(i) for i in indices]
        pipeline.mark_skipped(chapter_id, parsed, skipped=skipped)
    except HTTPException as exc:
        raise _http_err(exc) from exc
    except (TypeError, ValueError) as exc:
        raise ToolError(str(exc)) from exc
    return _json({"chapter_id": chapter_id, "page_indices": parsed, "skipped": skipped})


def _series(_session, args: dict) -> str:
    url = str(args.get("url") or "").strip()
    key = str(args.get("key") or "").strip()
    if not url and not key:
        return _json(series_memory.list_series())
    resolved = key or (series_memory.series_key(url) or "")
    if not resolved:
        raise ToolError("URL does not identify a series; pass a series URL or key")
    try:
        data = series_memory.load(resolved)
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    return _json({
        "key": resolved,
        "text": series_memory.to_text(data),
        "notes": data.get("notes") or "",
        "chapters": len(data.get("chapters") or []),
    })


def _series_save(_session, args: dict) -> str:
    url = str(args.get("url") or "").strip()
    key = str(args.get("key") or "").strip()
    resolved = key or (series_memory.series_key(url) or "")
    if not resolved:
        raise ToolError("provide url or key for the series")
    text = args.get("text")
    notes = args.get("notes")
    if text is not None and contains_secret_token(str(text)):
        raise ToolError("series memory text must not contain API keys or credentials")
    if notes is not None and contains_secret_token(str(notes)):
        raise ToolError("series memory notes must not contain API keys or credentials")

    def edit(current: dict) -> dict:
        data = series_memory.from_text(str(text) if text is not None else series_memory.to_text(current), current)
        if notes is not None:
            data["notes"] = str(notes)[:series_memory.MAX_NOTES]
        return data

    try:
        saved = series_memory.update(resolved, edit)
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    return _json({
        "key": resolved,
        "text": series_memory.to_text(saved),
        "notes": saved.get("notes") or "",
        "chapters": len(saved.get("chapters") or []),
    })


def _providers(_session, _args: dict) -> str:
    from app.ai_providers import PROVIDERS
    from app.secret_store import SecretStoreUnavailable, list_provider_configs, provider_key_status

    rows = []
    seen: set[str] = set()
    for p in PROVIDERS.values():
        seen.add(p.id)
        try:
            st = provider_key_status(p.id, provider_label=p.label)
        except SecretStoreUnavailable:
            st = {"configured": False}
        rows.append({
            "id": p.id,
            "label": p.label,
            "configured": bool(st.get("configured")),
            "default_translation_model": p.default_translation_model,
            "default_qc_model": p.default_qc_model,
        })
    try:
        for cfg in list_provider_configs():
            pid = str(cfg.get("id") or "")
            if not pid or pid in seen:
                continue
            label = str(cfg.get("label") or pid)
            st = provider_key_status(pid, provider_label=label)
            rows.append({
                "id": pid,
                "label": label,
                "configured": bool(st.get("configured")),
                "default_translation_model": cfg.get("default_translation_model") or "",
                "default_qc_model": cfg.get("default_qc_model") or "",
            })
    except SecretStoreUnavailable:
        pass
    return _json(rows)


def _ai_mode_status(_session, args: dict) -> str:
    from app.routers.ai_mode import ai_mode_jobs, ai_mode_queues

    job_id = str(args.get("job_id") or "").strip()
    queue_id = str(args.get("queue_id") or "").strip()
    try:
        if job_id:
            return _json(ai_mode_jobs.snapshot(job_id))
        if queue_id:
            return _json(ai_mode_queues.snapshot(queue_id))
    except KeyError as exc:
        raise ToolError(f"AI Mode job/queue not found: {exc}") from exc
    job = ai_mode_jobs.active_job()
    queue = ai_mode_queues.active_queue()
    return _json({
        "job": ai_mode_jobs.snapshot(job.job_id) if job else None,
        "queue": ai_mode_queues.snapshot(queue.queue_id) if queue else None,
    })


def _ai_mode_start(_session, args: dict) -> str:
    from app.ai_mode.job import AIModeSettings
    from app.ai_providers import CLOUD_PROVIDER_ID, validate_model_name
    from app.manifest_utils import load_manifest_raw
    from app.routers.ai_mode import ai_mode_jobs, ai_mode_queues
    from app.routers.visual_qc import _resolve_configured_provider
    from app.secret_store import SecretStoreUnavailable, get_provider_api_key

    chapter_id = str(args.get("chapter_id") or "").strip() or None
    url = str(args.get("url") or "").strip()
    if bool(chapter_id) == bool(url):
        raise ToolError("provide exactly one of chapter_id or url")
    story_notes = str(args.get("story_notes") or "")[:1500]
    if contains_secret_token(story_notes):
        raise ToolError("story_notes must not contain API keys or credentials")

    provider_id = str(args.get("provider") or "gemini").strip()
    if provider_id == CLOUD_PROVIDER_ID:
        raise ToolError("Manga Cloud jobs must be started from the UI")

    try:
        if chapter_id:
            validate_chapter_id(chapter_id)
            load_manifest_raw(chapter_id)
        else:
            validate_url(url)
        if ai_mode_jobs.active_job() is not None or ai_mode_queues.active_queue() is not None:
            raise ToolError("an AI Mode job or queue is already running")
        provider = _resolve_configured_provider(provider_id)
        model = validate_model_name(str(args.get("model") or ""), default=str(provider.default_qc_model or ""))
        api_key = get_provider_api_key(provider.id, provider_label=provider.label)
    except HTTPException as exc:
        raise _http_err(exc) from exc
    except (ValueError, SecretStoreUnavailable) as exc:
        raise ToolError(str(exc)) from exc
    if not api_key:
        raise ToolError(f"API key is not configured for provider {provider.label}")

    settings = AIModeSettings(
        url=url,
        chapter_id=chapter_id,
        provider=provider.id,
        model=model,
        target_lang=str(args.get("target_lang") or "vi").strip()[:16],
        budget_usd=max(0.05, min(20.0, float(args.get("budget_usd") or 2.0))),
        workers=max(1, min(8, int(args.get("workers") or 2))),
        story_notes=story_notes,
        polish=bool(args.get("polish", True)),
    )
    snap = ai_mode_jobs.start(settings, provider=provider, api_key=api_key)
    return _json(snap)


def _ai_mode_cancel(_session, args: dict) -> str:
    from app.routers.ai_mode import ai_mode_jobs, ai_mode_queues

    job_id = str(args.get("job_id") or "").strip()
    queue_id = str(args.get("queue_id") or "").strip()
    try:
        if job_id:
            return _json(ai_mode_jobs.cancel(job_id))
        if queue_id:
            return _json(ai_mode_queues.cancel(queue_id))
        active = ai_mode_jobs.active_job()
        if active is not None:
            return _json(ai_mode_jobs.cancel(active.job_id))
        q = ai_mode_queues.active_queue()
        if q is not None:
            return _json(ai_mode_queues.cancel(q.queue_id))
    except KeyError as exc:
        raise ToolError(f"AI Mode job/queue not found: {exc}") from exc
    return _json({"cancelled": False, "message": "No active AI Mode job or queue."})


def register(reg) -> None:
    """Register built-in Manga Translator app management tools under group 'app'."""
    tools = [
        (
            {
                "name": "app_chapters",
                "description": "List all chapters in the Manga Translator app (chapter_id, chapter_name, source_url, total_pages, workflow).",
                "parameters": {"type": "object", "properties": {}},
            },
            _chapters, "read", False,
        ),
        (
            {
                "name": "app_page",
                "description": "Inspect a chapter summary (omit page_index) or one page's text_objects, translations, flags, and image paths (usable with tools.view_image).",
                "parameters": {
                    "type": "object", "required": ["chapter_id"],
                    "properties": {"chapter_id": {"type": "string"}, "page_index": {"type": "integer"}},
                },
            },
            _page, "read", False,
        ),
        (
            {
                "name": "app_preflight",
                "description": "Run editorial preflight on a chapter to list all export blockers (untranslated objects, unreviewed flags, render skips).",
                "parameters": {
                    "type": "object", "required": ["chapter_id"],
                    "properties": {"chapter_id": {"type": "string"}, "require_final_approval": {"type": "boolean"}},
                },
            },
            _preflight, "read", False,
        ),
        (
            {
                "name": "app_update_text",
                "description": "Update translations, ocr_text, or editorial review flags for one or more text objects in a chapter.",
                "parameters": {
                    "type": "object", "required": ["chapter_id", "updates"],
                    "properties": {
                        "chapter_id": {"type": "string"},
                        "updates": {
                            "type": "array",
                            "items": {
                                "type": "object", "required": ["page_index", "id"],
                                "properties": {
                                    "page_index": {"type": "integer"}, "id": {"type": "string"},
                                    "translation": {"type": "string"}, "ocr_text": {"type": "string"},
                                    "editorial_disposition": {"type": "string"}, "editorial_reviewed": {"type": "boolean"},
                                },
                            },
                        },
                    },
                },
            },
            _update_text, "edit", False,
        ),
        (
            {
                "name": "app_render_page",
                "description": "Render one or more pages of a chapter and return rendered image paths and any skipped_objects.",
                "parameters": {
                    "type": "object", "required": ["chapter_id"],
                    "properties": {
                        "chapter_id": {"type": "string"},
                        "page_index": {"type": "integer"},
                        "page_indices": {"type": "array", "items": {"type": "integer"}},
                    },
                },
            },
            _render_page, "edit", False,
        ),
        (
            {
                "name": "app_skip_pages",
                "description": "Mark or unmark pages of a chapter as skipped.",
                "parameters": {
                    "type": "object", "required": ["chapter_id", "page_indices"],
                    "properties": {
                        "chapter_id": {"type": "string"},
                        "page_indices": {"type": "array", "items": {"type": "integer"}},
                        "skipped": {"type": "boolean"},
                    },
                },
            },
            _skip_pages, "edit", False,
        ),
        (
            {
                "name": "app_series",
                "description": "List all series memories (no args) or read one series's glossary/notes by url or key.",
                "parameters": {
                    "type": "object",
                    "properties": {"url": {"type": "string"}, "key": {"type": "string"}},
                },
            },
            _series, "read", False,
        ),
        (
            {
                "name": "app_series_save",
                "description": "Update a series's memory text (names, terms, address, characters) and/or story notes.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"}, "key": {"type": "string"},
                        "text": {"type": "string"}, "notes": {"type": "string"},
                    },
                },
            },
            _series_save, "edit", False,
        ),
        (
            {
                "name": "app_providers",
                "description": "List AI providers and whether each has an API key configured (never returns secret keys).",
                "parameters": {"type": "object", "properties": {}},
            },
            _providers, "read", False,
        ),
        (
            {
                "name": "app_ai_mode_status",
                "description": "Inspect the active AI Mode job/queue or a specific job_id/queue_id status and report.",
                "parameters": {
                    "type": "object",
                    "properties": {"job_id": {"type": "string"}, "queue_id": {"type": "string"}},
                },
            },
            _ai_mode_status, "read", False,
        ),
        (
            {
                "name": "app_ai_mode_start",
                "description": "Start an AI Mode translation job for an existing chapter_id or remote chapter url (always asks user approval).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "chapter_id": {"type": "string"}, "url": {"type": "string"},
                        "provider": {"type": "string"}, "model": {"type": "string"},
                        "target_lang": {"type": "string"}, "budget_usd": {"type": "number"},
                        "workers": {"type": "integer"}, "story_notes": {"type": "string"},
                        "polish": {"type": "boolean"},
                    },
                },
            },
            _ai_mode_start, "exec", True,
        ),
        (
            {
                "name": "app_ai_mode_cancel",
                "description": "Cancel a running AI Mode job or queue.",
                "parameters": {
                    "type": "object",
                    "properties": {"job_id": {"type": "string"}, "queue_id": {"type": "string"}},
                },
            },
            _ai_mode_cancel, "exec", False,
        ),
    ]
    for spec, handler, kind, always_ask in tools:
        if spec["name"] not in reg.tools and spec["name"] not in reg.reserved:
            reg.tool(spec, handler, kind=kind, group="app", always_ask=always_ask)
