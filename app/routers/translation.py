from __future__ import annotations

import copy
import difflib

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
from app.render.font_catalog import FontNotFoundError, resolve_font_id
from app.render.page_renderer import letter_box
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
from app.translation.cost import preflight_cost_usd
from app.translation.vision import VisionPageTranslator, collapse_repeated_lines, drawn_in_art
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


class TranslateVisionPageRequest(TranslateChapterRequest):
    page_index: int = Field(ge=0)
    # Cap per request; retries use a small batch so the reply fits.
    max_objects: int | None = Field(default=None, ge=1, le=100)
    # Only these objects (still untranslated ones); None means every untranslated object.
    object_ids: list[str] | None = Field(default=None, max_length=100)
    # A second model re-translates the source lines the vision model read; unset means the vision model's wording stands.
    translate_provider: str | None = None
    translate_model: str | None = None

    @field_validator("translate_provider")
    @classmethod
    def _translate_provider(cls, value: str | None) -> str | None:
        return normalize_provider_id(value) if value else None

    @field_validator("translate_model")
    @classmethod
    def _translate_model(cls, value: str | None) -> str | None:
        return None if value is None else validate_model_name(value, default="")


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


def _resolve_text_provider(provider_id: str):
    """A chat provider for text-only work; unlike vision it need not take images."""
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
    return provider


def _vision_candidates(page: dict, *, force: bool) -> list[dict]:
    candidates: list[dict] = []
    seen: set[str] = set()
    for obj in page.get("text_objects") or []:
        if not isinstance(obj, dict) or not obj.get("id") or obj.get("source_missing") or obj.get("joined_into"):
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
            # The renderer letters where the source lettering was, so that is the room a translation has.
            "fit_region": list(letter_box(obj, tuple(rect)) or rect),
            "initial_translation": str(obj.get("translation") or ""),
        })
    return candidates


_SFX_LIKE_ROLES = frozenset({"sfx", "free_text"})


def _norm_text(value: str) -> str:
    return " ".join(str(value or "").split()).lower()


_DUPLICATE_SOURCE_SIMILARITY = 0.85  # normalised sources this alike are the same text read twice


def _rects_overlap(a: list[int], b: list[int]) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def _duplicate_reason(value: str, source: str, rect: list[int],
                      other_value: str, other_source: str, other_rect: list[int]) -> str | None:
    """Why one translation is a double-read of another: "overlap" when the
    letterings cover the same place, "source" when they read the same text
    in different places. Translation alone never proves it — two bubbles
    can each say "Của tôi" — so one of the two must hold as well."""
    if _norm_text(value) != _norm_text(other_value) or not _norm_text(value):
        return None
    if _rects_overlap(rect, other_rect):
        return "overlap"
    a, b = _norm_text(source), _norm_text(other_source)
    if a and a == b:
        # The same text read twice, however short (a two-syllable SFX is a word).
        return "source"
    if len(a) >= 3 and len(b) >= 3 and difflib.SequenceMatcher(None, a, b).ratio() >= _DUPLICATE_SOURCE_SIMILARITY:
        return "source"
    return None


def _duplicate_translation_ids(candidates: list[dict], translated) -> dict[str, str]:
    """Ids of a text drawn twice, to keep once; the value is the reason ("overlap"/"source").

    The artist sometimes draws one SFX twice in a panel (ch111 p002: the
    cheer, top-left and mid-right); the detector correctly finds both, the
    model translates both to the same word, and the page shows the word
    twice. Keep the larger instance and leave the other as the artist
    drew it. The same double-read happens to dialogue — one bubble boxed
    twice — so the check is role-agnostic: same translation, and either
    the letterings overlap or the sources read the same. Dialogue that
    repeats in different bubbles is never a duplicate on translation
    alone: its sources differ.
    """
    sources = getattr(translated, "sources", {}) or {}
    translations = getattr(translated, "translations", {}) or {}
    by_value: dict[str, list[dict]] = {}
    for candidate in candidates:
        value = _norm_text(translations.get(candidate["id"]))
        if value:
            by_value.setdefault(value, []).append(candidate)
    duplicates: dict[str, str] = {}
    for group in by_value.values():
        if len(group) < 2:
            continue
        group.sort(key=lambda c: -((c["region"][2] - c["region"][0])
                                    * (c["region"][3] - c["region"][1])))
        kept = [group[0]]
        for other in group[1:]:
            other_source = str(sources.get(other["id"]) or other.get("text") or "")
            other_rect = list(other.get("fit_region") or other["region"])
            reason = None
            for keeper in kept:
                keeper_source = str(sources.get(keeper["id"]) or keeper.get("text") or "")
                keeper_rect = list(keeper.get("fit_region") or keeper["region"])
                reason = _duplicate_reason(
                    str(translations.get(other["id"]) or ""), other_source, other_rect,
                    str(translations.get(keeper["id"]) or ""), keeper_source, keeper_rect)
                if reason is not None:
                    break
            if reason is not None:
                duplicates[str(other["id"])] = reason
            else:
                kept.append(other)
    return duplicates


def _committed_duplicate(page: dict, candidate: dict, value: str, source: str) -> str | None:
    """The same double-read against objects a previous call already
    translated: the batch dedupe only sees this call's candidates, so a
    retry re-reading a bubble the first call translated would letter it
    twice. The candidate's own object and untranslated objects are not
    evidence."""
    rect = list(candidate.get("fit_region") or candidate["region"])
    for obj in page.get("text_objects") or []:
        if (not isinstance(obj, dict) or str(obj.get("id")) == str(candidate["id"])
                or obj.get("source_missing") or obj.get("joined_into") or obj.get("overlap_dropped")
                or text_object_in_preserve_region(page, obj)):
            continue
        other_value = str(obj.get("translation") or "")
        if not other_value.strip():
            continue
        other_rect = obj.get("letter_bounds")
        if isinstance(other_rect, dict):
            try:
                other_rect = [int(other_rect[k]) for k in ("x1", "y1", "x2", "y2")]
            except (KeyError, TypeError, ValueError):
                other_rect = None
        if other_rect is None:
            region = obj.get("region")
            other_rect = ([int(region[k]) for k in ("x1", "y1", "x2", "y2")]
                          if isinstance(region, dict) else None)
        if other_rect is None:
            continue
        reason = _duplicate_reason(
            value, source, rect, other_value,
            str(obj.get("source_read") or obj.get("ocr_text") or ""), other_rect)
        if reason is not None:
            return reason
    return None


def _add_fit_budgets(original_path, candidates: list[dict]) -> None:
    """Measure each region's source letter size and the characters that fit it at that size."""
    from app.render.font_catalog import resolve_font_id
    from app.render.font_guide import DEFAULT_LETTERING_FONT
    from app.render.source_size import char_budget, source_cap_px, source_ink_hex

    from app.image_io import read_image

    raw = read_image(original_path)
    font_path = resolve_font_id(DEFAULT_LETTERING_FONT)
    for candidate in candidates:
        ink = source_ink_hex(raw, candidate["region"])
        if ink:
            candidate["source_ink"] = ink
        # A lone "!" or "?" is one or two ink pieces, too few to trust in longer text but all the letters it has.
        short = len("".join(str(candidate.get("text") or "").split())) <= 2
        # The cap is the source lettering's size, so it is measured where
        # the lettering was (fit_region), never over the whole region:
        # art strokes in a roomy region read as giant letters otherwise.
        cap = source_cap_px(raw, candidate["region"], min_pieces=1 if short else 3,
                            bounds=candidate.get("fit_region"))
        if cap:
            candidate["source_cap_px"] = cap
            candidate["max_chars"] = char_budget(font_path, candidate.get("fit_region") or candidate["region"], cap)


async def translate_page_in_context(
    req: TranslateVisionPageRequest, memory: ChapterMemory | None = None, slice_total: int | None = None,
    skip_seam_mirrors: bool = False,
) -> dict:
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

    translate_provider = translate_model = translate_api_key = None
    if req.translate_provider and req.translate_model:
        try:
            translate_provider = _resolve_text_provider(req.translate_provider)
            translate_model = validate_model_name(req.translate_model, default="")
            translate_api_key = get_provider_api_key(translate_provider.id, provider_label=translate_provider.label)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except SecretStoreUnavailable as exc:
            raise HTTPException(503, str(exc)) from exc
        if not translate_api_key:
            raise HTTPException(409, f"{translate_provider.label} API key is not configured")

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
        elsewhere: list[list[int]] = []
        if skip_seam_mirrors:
            from app.ai_mode.seams import seam_mirror_ids

            # The slice that owns a text crossing the cut translates it; this one gets a copy.
            mirrors = seam_mirror_ids(manifest, req.page_index)
            elsewhere = [candidate["region"] for candidate in candidates if candidate["id"] in mirrors]
            candidates = [candidate for candidate in candidates if candidate["id"] not in mirrors]
        if req.object_ids is not None:
            wanted = set(req.object_ids)
            candidates = [candidate for candidate in candidates if candidate["id"] in wanted]
        # The cap applies to what was actually asked for: a retry naming a
        # few objects must not 400 because the slice holds more than 100.
        if len(candidates) > 100:
            raise HTTPException(400, "Too many text objects on one slice (maximum 100)")
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

    if ensured:
        # The render was invalidated above: drop its stale file too (M2),
        # so a translate/render failure below leaves no stale PNG behind.
        from app.dependencies import pipeline
        pipeline._sync_output_dir(req.chapter_id, manifest, [req.page_index])

    if not candidates:
        result = urlify_manifest(manifest)
        result["translation_run"] = {
            "translated": 0, "unreadable": 0, "stale": 0, "model": model,
            "rendered_pages": [], "estimated_cost_usd": 0.0 if provider.tracks_cost else None,
        }
        return result
    if not original_path.is_file() or not clean_path.is_file():
        raise HTTPException(404, "Original or cleaned image is unavailable")

    if provider.tracks_cost and preflight_cost_usd(candidates) > req.budget_usd:
        raise HTTPException(409, "Text-only lower-bound estimate exceeds remaining budget")
    await run_in_threadpool(_add_fit_budgets, original_path, candidates)
    translator = VisionPageTranslator(
        provider, model, translate_provider=translate_provider,
        translate_model=translate_model, translate_api_key=translate_api_key,
    )
    try:
        translated = await run_in_threadpool(
            translator.translate_page, original_path, clean_path, candidates,
            api_key=api_key, source_lang=req.source_lang, target_lang=req.target_lang,
            memory=memory, slice_number=req.page_index + 1, slice_total=slice_total, elsewhere=elsewhere,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(502, str(exc)) from exc

    committed = stale = unreadable = review = 0
    blank_ids: list[str] = []
    story_blank_ids: list[str] = []  # blanks the model left on a story box (any role but sfx)
    art_regions: list[dict] = []  # sound effects left as art get their original pixels back
    duplicate_ids = _duplicate_translation_ids(candidates, translated)
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
            value = collapse_repeated_lines(translated.translations[candidate["id"]])
            if isinstance(value, str):
                value = value.strip()
            role = getattr(translated, "roles", {}).get(candidate["id"])
            source = getattr(translated, "sources", {}).get(candidate["id"]) or candidate["text"]
            if role == "sfx" and drawn_in_art(source):
                value = ""  # a sound effect the artist drew stays as drawn, whatever the model wrote
            duplicate_reason = duplicate_ids.get(candidate["id"])
            if duplicate_reason is not None:
                value = ""  # the same text read twice: keep one instance
            elif value:
                # The batch dedupe only sees this call's candidates; a
                # retry re-reading a bubble the first call translated
                # would otherwise letter it twice.
                duplicate_reason = _committed_duplicate(page, candidate, value, source)
                if duplicate_reason is not None:
                    value = ""
            if not value:
                unreadable += 1
                if candidate["id"] not in getattr(translated, "missing_ids", ()):
                    blank_ids.append(candidate["id"])
                    # Attempted and left blank on purpose; distinguishes blank-by-choice from blank-by-budget.
                    obj["translation_attempted"] = True
                    # A blank on a story box is a failure to read, not a
                    # choice: it is retried, not restored to source art.
                    # Only a sound effect the artist drew is art.
                    # A duplicate SFX is blank by decision, not by failure,
                    # so it is kept as art and never retried.
                    if role != "sfx" and duplicate_reason is None:
                        story_blank_ids.append(candidate["id"])
                    # Only a sound effect the artist drew is art; a watermark or a review box around missed text stays erased.
                    # A duplicate is restored as art only when it is a
                    # separate instance of an SFX (the same text drawn
                    # again): an overlapping duplicate shares the keeper's
                    # lettering, so restoring it would paint the source
                    # text back under the keeper's translation.
                    manual = {str(box.get("id")) for box in page.get("boxes") or []
                              if isinstance(box, dict) and box.get("manual")}
                    if ((duplicate_reason == "source" and str(role or "").strip().lower() in _SFX_LIKE_ROLES)
                            or (role == "sfx" and drawn_in_art(source)
                                and not set(map(str, obj.get("source_boxes") or [])) & manual)):
                        art_regions.append(dict(obj["region"]))
                continue
            obj["translation"] = value
            obj["translation_source"] = provider.id
            if candidate.get("source_cap_px"):
                obj["source_cap_px"] = candidate["source_cap_px"]
            obj["translation_model"] = translated.model
            obj["translation_input_text"] = candidate["text"]
            source_read = getattr(translated, "sources", {}).get(candidate["id"])
            if source_read:
                obj["source_read"] = source_read
            speaker = getattr(translated, "speakers", {}).get(candidate["id"])
            if speaker:
                obj["speaker"] = speaker
            obj["auto_translation"] = value
            _apply_ai_font_choice(obj, getattr(translated, "font_choices", {}).get(candidate["id"]))
            if role:
                obj["typography_role"] = role
            container = getattr(translated, "containers", {}).get(candidate["id"])
            if container:
                obj["container"] = container
            if candidate["id"] in getattr(translated, "review_ids", ()):
                obj["needs_review"] = True
                review += 1
            color = getattr(translated, "colors", {}).get(candidate["id"]) or candidate.get("source_ink")
            if color:
                obj["lettering_color"] = color
            if candidate["id"] in getattr(translated, "enlarge_ids", ()):
                obj["enlarge"] = True
            else:
                obj.pop("enlarge", None)
            committed += 1
            changed = True
        if changed:
            invalidate_page_render(latest, page_number)
            save_manifest_raw(req.chapter_id, latest)
        elif blank_ids:
            # A blank-only run still mutates the manifest
            # (translation_attempted), and a crash before the next save
            # would lose it — including the all-blank page, whose run
            # would otherwise leave no trace at all.
            save_manifest_raw(req.chapter_id, latest)

    if changed:
        # The render was invalidated above: drop its stale file too (M2)
        # before the re-render below, so a render failure leaves no stale
        # PNG behind.
        from app.dependencies import pipeline
        pipeline._sync_output_dir(req.chapter_id, latest, [page_number])

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
        "missing_ids": sorted(getattr(translated, "missing_ids", ())),
        "blank_ids": blank_ids,
        "story_blank_ids": story_blank_ids,
        "art_regions": art_regions,
        "unboxed": [list(box) for box in getattr(translated, "unboxed", ())],
        "remaining": max(0, total_candidates - len(candidates)),
    }
    return result
