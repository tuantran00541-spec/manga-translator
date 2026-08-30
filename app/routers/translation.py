from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, field_validator

from app.manifest_utils import (
    get_manifest_lock,
    invalidate_page_render,
    load_manifest_raw,
    save_manifest_raw,
    urlify_manifest,
)
from app.ocr.quality import should_block_translation
from app.secret_store import SecretStoreUnavailable, get_deepseek_api_key
from app.security import validate_chapter_id
from app.text_objects import ensure_page_text_objects
from app.translation import DeepSeekTranslator, TranslationBudgetExceeded
from app.translation.deepseek import PRICING_VERSION


router = APIRouter(prefix="/api/translate", tags=["translation"])
translator = DeepSeekTranslator()
MAX_CHAPTER_TRANSLATION_OBJECTS = 300


class TranslateChapterRequest(BaseModel):
    chapter_id: str
    source_lang: str = "ja"
    target_lang: str = "vi"
    budget_usd: float = 0.02
    force: bool = False

    @field_validator("source_lang", "target_lang")
    @classmethod
    def _lang(cls, value: str) -> str:
        value = (value or "").strip().lower()
        if not value or len(value) > 20 or not all(c.isalnum() or c in "-_" for c in value):
            raise ValueError("Invalid language code")
        return value

    @field_validator("budget_usd")
    @classmethod
    def _budget(cls, value: float) -> float:
        if value < 0.001 or value > 0.25:
            raise ValueError("budget_usd must be between 0.001 and 0.25")
        return float(value)


def _find_object(page: dict, object_id: str) -> dict | None:
    return next(
        (
            obj
            for obj in (page.get("text_objects") or [])
            if isinstance(obj, dict) and str(obj.get("id")) == object_id
        ),
        None,
    )


@router.post("/chapter")
async def translate_chapter(req: TranslateChapterRequest) -> dict:
    validate_chapter_id(req.chapter_id)
    try:
        api_key = get_deepseek_api_key()
    except SecretStoreUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    if not api_key:
        raise HTTPException(409, "DeepSeek API key is not configured")

    skipped_ocr_reject = 0
    with get_manifest_lock(req.chapter_id):
        manifest = load_manifest_raw(req.chapter_id)
        ensured = False
        candidates: list[dict] = []
        for page_index, page in enumerate(manifest.get("pages", [])):
            if page.get("skipped"):
                continue
            _, changed = ensure_page_text_objects(page)
            ensured = ensured or changed
            for obj in page.get("text_objects") or []:
                if not isinstance(obj, dict) or not obj.get("id"):
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
        if ensured:
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
            obj = _find_object(pages[page_index], str(item["id"]))
            if obj is None:
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
            obj["translation_source"] = "deepseek"
            obj["translation_model"] = translated.model
            obj["translation_input_text"] = str(item["text"])
            obj["auto_translation"] = value
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
        "model": translated.model,
        "usage": translated.usage,
        "estimated_cost_usd": round(translated.estimated_cost_usd, 6),
        "budget_usd": req.budget_usd,
        "pricing_version": PRICING_VERSION,
    }
    return result
