from __future__ import annotations

import copy

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, field_validator

from app.config import PROCESSED_DIR, RAW_DIR
from app.ai_providers import (
    PROVIDERS,
    cloud_job_provider,
    normalize_provider_id,
    resolve_provider,
    validate_model_name,
)
from app.manifest_utils import (
    get_manifest_lock,
    invalidate_page_render,
    load_manifest_raw,
    save_manifest_raw,
    urlify_manifest,
)
from app.ocr.quality import should_block_translation
from app.render.font_catalog import FontNotFoundError, resolve_font_id
from app.secret_store import (
    SecretStoreUnavailable,
    get_provider_api_key,
    get_provider_config,
)
from app.security import validate_chapter_id, validate_url, validate_managed_path
from app.schemas import RenderRequest
from app.routers.render_commit import render_page
from app.text_objects import ensure_page_text_objects
from app.region_policy import text_object_in_preserve_region
from app.translation import DeepSeekTranslator, TranslationBudgetExceeded
from app.translation.deepseek import PRICING_VERSION, _preflight_cost_usd
from app.translation.vision import VisionPageTranslator
from app.translation.context import ChapterMemory


router = APIRouter(prefix="/api/translate", tags=["translation"])
MAX_CHAPTER_TRANSLATION_OBJECTS = 300


def validate_lang_code(value: str) -> str:
    value = (value or "").strip().lower()
    if not value or len(value) > 20 or not all(c.isalnum() or c in "-_" for c in value):
        raise ValueError("Invalid language code")
    return value


class TranslateChapterRequest(BaseModel):
    chapter_id: str
    source_lang: str = "ja"
    target_lang: str = "vi"
    budget_usd: float = 0.02
    force: bool = False
    provider: str = "deepseek"
    model: str | None = None

    @field_validator("provider")
    @classmethod
    def _provider(cls, value: str) -> str:
        return normalize_provider_id(value)

    @field_validator("model")
    @classmethod
    def _model(cls, value: str | None) -> str | None:
        return None if value is None else validate_model_name(value, default="")

    @field_validator("source_lang", "target_lang")
    @classmethod
    def _lang(cls, value: str) -> str:
        return validate_lang_code(value)

    @field_validator("budget_usd")
    @classmethod
    def _budget(cls, value: float) -> float:
        if value < 0.001 or value > 0.25:
            raise ValueError("budget_usd must be between 0.001 and 0.25")
        return float(value)


def _resolve_translation_provider(provider_id: str):
    normalized = normalize_provider_id(provider_id)
    if normalized in PROVIDERS:
        provider = PROVIDERS[normalized]
    else:
        stored = get_provider_config(normalized)
        if not isinstance(stored, dict):
            raise ValueError(
                f"Custom AI provider is not configured: {normalized}"
            )
        provider = resolve_provider(
            normalized,
            label=stored.get("label"),
            protocol=stored.get("protocol"),
            api_base=stored.get("api_base"),
        )
        validate_url(str(provider.chat_url))
    if provider.protocol != "openai" or not provider.supports_translation:
        raise ValueError(f"{provider.label} is not available for translation")
    return provider


def _find_object(page: dict, object_id: str) -> dict | None:
    return next(
        (
            obj
            for obj in (page.get("text_objects") or [])
            if isinstance(obj, dict) and str(obj.get("id")) == object_id
        ),
        None,
    )


def _apply_ai_font_choice(obj: dict, choice: dict | None) -> None:
    if not isinstance(choice, dict):
        return
    font_id = str(choice.get("font_id") or "").strip()
    font_mode = str(choice.get("font_mode") or "ai").strip().lower()
    if not font_id:
        return
    if font_id == "auto":
        obj["font_ai_id"] = "auto"
        obj["font_selection_mode"] = "auto"
        obj.setdefault("style", {})["font"] = "auto"
        return
    try:
        resolve_font_id(font_id)
    except (FontNotFoundError, OSError, ValueError):
        obj["font_ai_rejection"] = {"font_id": font_id, "reason": "unknown_catalog_id"}
        return
    if font_mode == "ai" and obj.get("font_selection_mode") != "user":
        obj["font_ai_id"] = font_id
        obj["font_selection_mode"] = "ai"


@router.post("/chapter")
async def translate_chapter(req: TranslateChapterRequest) -> dict:
    validate_chapter_id(req.chapter_id)
    try:
        provider = _resolve_translation_provider(req.provider)
        model = validate_model_name(
            req.model,
            default=str(provider.default_translation_model or ""),
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except SecretStoreUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    translator = DeepSeekTranslator(
        model=model,
        api_url=str(provider.chat_url),
        provider_id=provider.id,
        provider_label=provider.label,
        provider=provider,
    )
    try:
        api_key = get_provider_api_key(
            provider.id,
            provider_label=provider.label,
        )
    except SecretStoreUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    if not api_key:
        raise HTTPException(409, f"{provider.label} API key is not configured")

    skipped_ocr_reject = 0
    skipped_source_missing = 0
    skipped_preserve_region = 0
    with get_manifest_lock(req.chapter_id):
        manifest = load_manifest_raw(req.chapter_id)
        ensured_pages: set[int] = set()
        candidates: list[dict] = []
        for page_index, page in enumerate(manifest.get("pages", [])):
            if page.get("skipped") or page.get("process_required"):
                continue
            _, changed = ensure_page_text_objects(page)
            if changed:
                ensured_pages.add(page_index)
            for obj in page.get("text_objects") or []:
                if not isinstance(obj, dict) or not obj.get("id"):
                    continue
                if obj.get("source_missing"):
                    skipped_source_missing += 1
                    continue
                if text_object_in_preserve_region(page, obj):
                    skipped_preserve_region += 1
                    continue
                source = str(obj.get("ocr_text") or "").strip()
                current_translation = str(obj.get("translation") or "")
                if not source or (current_translation.strip() and not req.force):
                    continue
                if should_block_translation(obj):
                    skipped_ocr_reject += 1
                    continue
                candidates.append(
                    {
                        "id": str(obj["id"]),
                        "page_index": page_index,
                        "text": source,
                        "initial_translation": current_translation,
                    }
                )
        if ensured_pages:
            for page_index in ensured_pages:
                invalidate_page_render(manifest, page_index)
            save_manifest_raw(req.chapter_id, manifest)

    if len(candidates) > MAX_CHAPTER_TRANSLATION_OBJECTS:
        raise HTTPException(
            400,
            f"Chapter has {len(candidates)} translatable regions; maximum is {MAX_CHAPTER_TRANSLATION_OBJECTS}",
        )
    if not candidates:
        result = urlify_manifest(manifest)
        result["translation_run"] = {
            "translated": 0,
            "stale": 0,
            "skipped_ocr_reject": skipped_ocr_reject,
            "skipped_source_missing": skipped_source_missing,
            "skipped_preserve_region": skipped_preserve_region,
            "model": translator.model,
            "estimated_cost_usd": 0.0,
            "budget_usd": req.budget_usd,
            "pricing_version": PRICING_VERSION,
        }
        return result

    try:
        translated = await run_in_threadpool(
            translator.translate,
            candidates,
            api_key=api_key,
            source_lang=req.source_lang,
            target_lang=req.target_lang,
            budget_usd=req.budget_usd,
        )
    except TranslationBudgetExceeded as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(502, str(exc)) from exc

    committed = 0
    stale = 0
    changed_pages: set[int] = set()
    with get_manifest_lock(req.chapter_id):
        latest = load_manifest_raw(req.chapter_id)
        pages = latest.get("pages", [])
        for item in candidates:
            page_index = int(item["page_index"])
            if page_index < 0 or page_index >= len(pages):
                stale += 1
                continue
            page = pages[page_index]
            if page.get("skipped") or page.get("process_required"):
                stale += 1
                continue
            obj = _find_object(page, str(item["id"]))
            if obj is None or obj.get("source_missing"):
                stale += 1
                continue
            if text_object_in_preserve_region(page, obj):
                stale += 1
                continue
            if str(obj.get("ocr_text") or "").strip() != str(item["text"]).strip():
                stale += 1
                continue
            if str(obj.get("translation") or "") != str(item["initial_translation"]):
                stale += 1
                continue
            value = translated.translations.get(str(item["id"]))
            if not value:
                stale += 1
                continue
            obj["translation"] = value
            obj["translation_source"] = provider.id
            obj["translation_model"] = translated.model
            obj["translation_input_text"] = str(item["text"])
            obj["auto_translation"] = value
            _apply_ai_font_choice(obj, getattr(translated, "font_choices", {}).get(str(item["id"])))
            changed_pages.add(page_index)
            committed += 1

        for page_index in changed_pages:
            invalidate_page_render(latest, page_index)
        if changed_pages:
            save_manifest_raw(req.chapter_id, latest)

    result = urlify_manifest(latest)
    result["translation_run"] = {
        "translated": committed,
        "stale": stale,
        "skipped_ocr_reject": skipped_ocr_reject,
        "skipped_source_missing": skipped_source_missing,
        "model": translated.model,
        "usage": translated.usage,
        "estimated_cost_usd": round(translated.estimated_cost_usd, 6),
        "budget_usd": req.budget_usd,
        "pricing_version": PRICING_VERSION,
    }
    return result


class TranslateVisionPageRequest(TranslateChapterRequest):
    page_index: int = Field(ge=0)
    # Cap per request; retries use a small batch so the reply fits.
    max_objects: int | None = Field(default=None, ge=1, le=100)
    # Only these objects (still untranslated ones); None means every untranslated object.
    object_ids: list[str] | None = Field(default=None, max_length=100)


def _resolve_vision_provider(provider_id: str):
    normalized = normalize_provider_id(provider_id)
    cloud = cloud_job_provider(normalized)
    if cloud is not None:
        provider = cloud
    elif normalized in PROVIDERS:
        provider = PROVIDERS[normalized]
    else:
        stored = get_provider_config(normalized)
        if not isinstance(stored, dict):
            raise ValueError(f"Custom AI provider is not configured: {normalized}")
        provider = resolve_provider(
            normalized, label=stored.get("label"), protocol=stored.get("protocol"),
            api_base=stored.get("api_base"),
        )
        validate_url(str(provider.chat_url))
    if not provider.supports_visual_qc:
        raise ValueError(f"{provider.label} does not support image inputs")
    return provider


def _vision_candidates(page: dict, *, force: bool) -> list[dict]:
    candidates: list[dict] = []
    seen: set[str] = set()
    for obj in page.get("text_objects") or []:
        if not isinstance(obj, dict) or not obj.get("id") or obj.get("source_missing"):
            continue
        if text_object_in_preserve_region(page, obj):
            continue
        obj_id = str(obj["id"])
        if obj_id in seen:
            raise ValueError(f"Duplicate text-object ID: {obj_id}")
        seen.add(obj_id)
        if not force and str(obj.get("translation") or "").strip():
            continue
        region = obj.get("region")
        if not isinstance(region, dict):
            continue
        try:
            rect = [int(region[key]) for key in ("x1", "y1", "x2", "y2")]
        except (KeyError, ValueError, TypeError):
            continue
        if rect[2] <= rect[0] or rect[3] <= rect[1]:
            continue
        candidates.append({
            "id": obj_id,
            "text": str(obj.get("ocr_text") or "").strip(),
            "region": rect,
            "initial_translation": str(obj.get("translation") or ""),
        })
    return candidates


@router.post("/page/vision")
async def translate_page_with_images(req: TranslateVisionPageRequest) -> dict:
    return await translate_page_in_context(req, repair=False)


async def translate_page_in_context(
    req: TranslateVisionPageRequest, memory: ChapterMemory | None = None, slice_total: int | None = None,
    repair: bool = True,
) -> dict:
    """Translate one slice; ``repair`` lets the model keep art text and report missed text (A.I mode only)."""
    validate_chapter_id(req.chapter_id)
    try:
        provider = _resolve_vision_provider(req.provider)
        model = validate_model_name(req.model, default=str(provider.default_qc_model or ""))
        api_key = get_provider_api_key(provider.id, provider_label=provider.label)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except SecretStoreUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    if not api_key:
        raise HTTPException(409, f"{provider.label} API key is not configured")

    with get_manifest_lock(req.chapter_id):
        manifest = load_manifest_raw(req.chapter_id)
        pages = manifest.get("pages", [])
        if req.page_index >= len(pages):
            raise HTTPException(404, "Page is not present in this chapter")
        page = pages[req.page_index]
        if page.get("skipped") or page.get("process_required") or not page.get("clean"):
            raise HTTPException(409, "Page must finish inpainting before vision translation")
        if not page.get("original"):
            raise HTTPException(409, "Original image is missing")
        _, ensured = ensure_page_text_objects(page)
        if ensured:
            invalidate_page_render(manifest, req.page_index)
            save_manifest_raw(req.chapter_id, manifest)
        try:
            candidates = _vision_candidates(page, force=req.force)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        if len(candidates) > 100:
            raise HTTPException(400, "Too many text objects on one slice (maximum 100)")
        if req.object_ids is not None:
            wanted = set(req.object_ids)
            candidates = [candidate for candidate in candidates if candidate["id"] in wanted]
        total_candidates = len(candidates)
        if req.max_objects is not None:
            candidates = candidates[:req.max_objects]
        original_path = validate_managed_path(page["original"], RAW_DIR / req.chapter_id)
        clean_path = validate_managed_path(page["clean"], PROCESSED_DIR / req.chapter_id)
        snapshot = {
            key: copy.deepcopy(page.get(key))
            for key in ("original", "clean", "source_revision", "clean_revision",
                        "skipped", "process_required")
        }
        page_number = req.page_index

    if not candidates:
        result = urlify_manifest(manifest)
        result["translation_run"] = {
            "translated": 0, "unreadable": 0, "stale": 0, "model": model,
            "rendered_pages": [], "estimated_cost_usd": 0.0 if provider.tracks_cost else None,
        }
        return result
    if not original_path.is_file() or not clean_path.is_file():
        raise HTTPException(404, "Original or cleaned image is unavailable")

    if provider.tracks_cost and _preflight_cost_usd(candidates) > req.budget_usd:
        raise HTTPException(409, "Text-only lower-bound estimate exceeds remaining budget")
    translator = VisionPageTranslator(provider, model)
    try:
        translated = await run_in_threadpool(
            translator.translate_page, original_path, clean_path, candidates,
            api_key=api_key, source_lang=req.source_lang, target_lang=req.target_lang,
            memory=memory, slice_number=req.page_index + 1, slice_total=slice_total, repair=repair,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(502, str(exc)) from exc

    committed = stale = unreadable = review = 0
    keep_regions: list[list[int]] = []
    blank_ids: list[str] = []
    with get_manifest_lock(req.chapter_id):
        latest = load_manifest_raw(req.chapter_id)
        pages = latest.get("pages", [])
        if page_number >= len(pages) or any(
            pages[page_number].get(key) != value for key, value in snapshot.items()
        ):
            raise HTTPException(409, "Slice changed while AI was translating; no text was overwritten")
        page = pages[page_number]
        changed = False
        for candidate in candidates:
            obj = _find_object(page, candidate["id"])
            if (
                obj is None or obj.get("source_missing")
                or text_object_in_preserve_region(page, obj)
                or str(obj.get("ocr_text") or "").strip() != candidate["text"]
                or str(obj.get("translation") or "") != candidate["initial_translation"]
                or not isinstance(obj.get("region"), dict)
                or [obj["region"].get(k) for k in ("x1", "y1", "x2", "y2")]
                   != candidate["region"]
            ):
                stale += 1
                continue
            if candidate["id"] in getattr(translated, "keep_ids", ()):
                keep_regions.append(list(candidate["region"]))
                continue
            value = translated.translations[candidate["id"]]
            if not value:
                unreadable += 1
                if candidate["id"] not in getattr(translated, "missing_ids", ()):
                    blank_ids.append(candidate["id"])
                continue
            obj["translation"] = value
            obj["translation_source"] = provider.id
            obj["translation_model"] = translated.model
            obj["translation_input_text"] = candidate["text"]
            obj["auto_translation"] = value
            _apply_ai_font_choice(obj, getattr(translated, "font_choices", {}).get(candidate["id"]))
            role = getattr(translated, "roles", {}).get(candidate["id"])
            if role:
                obj["typography_role"] = role
            if candidate["id"] in getattr(translated, "review_ids", ()):
                obj["needs_review"] = True
                review += 1
            committed += 1
            changed = True
        if changed:
            invalidate_page_render(latest, page_number)
            save_manifest_raw(req.chapter_id, latest)

    rendered_pages: list[int] = []
    render_error = None
    if committed:
        try:
            await run_in_threadpool(
                render_page, RenderRequest(
                    chapter_id=req.chapter_id, page_index=page_number, translations={},
                )
            )
            rendered_pages.append(page_number)
        except HTTPException as exc:
            render_error = str(exc.detail)
        except Exception:
            render_error = "Automatic rendering failed; translations were saved"

    with get_manifest_lock(req.chapter_id):
        current = load_manifest_raw(req.chapter_id)
    result = urlify_manifest(current)
    result["translation_run"] = {
        "translated": committed, "unreadable": unreadable, "stale": stale, "review": review,
        "model": translated.model, "usage": translated.usage,
        "estimated_cost_usd": (
            round(translated.estimated_cost_usd, 6)
            if translated.estimated_cost_usd is not None else None
        ),
        "budget_exceeded": (
            translated.estimated_cost_usd is not None
            and translated.estimated_cost_usd >= req.budget_usd
        ),
        "rendered_pages": rendered_pages, "render_error": render_error,
        "keep_regions": keep_regions,
        "missing_ids": sorted(getattr(translated, "missing_ids", ())),
        "blank_ids": blank_ids,
        "missed_boxes": [list(box) for box in getattr(translated, "missed_boxes", ())],
        "remaining": max(0, total_candidates - len(candidates)),
    }
    return result
