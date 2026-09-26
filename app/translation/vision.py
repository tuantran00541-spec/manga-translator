from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import requests

from app.ai_providers import AIProvider
from app.parameters import TRANSLATION_CONNECT_TIMEOUT_SECONDS, TRANSLATION_READ_TIMEOUT_SECONDS
from app.render.font_catalog import load_font_catalog
from app.security import validate_url
from app.translation.context import TYPOGRAPHY_ROLES, ChapterMemory, system_prompt
from app.translation.deepseek import _language_name, _usage_cost_usd
from app.visual_qc.deepseek_region_client import _extract_output_text, _safe_error_detail
from app.visual_qc.gemini import _encode_for_gemini, _read_image
from app.visual_qc.gemini_interactions import (
    GEMINI_INTERACTIONS_URL,
    extract_output_text as gemini_output_text,
    safe_error_detail as gemini_safe_error,
)

_TRANSLATIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "translations": {"type": "array", "items": {
            "type": "object",
            "properties": {"id": {"type": "string"}, "translated_text": {"type": "string"}},
            "required": ["id", "translated_text"],
        }},
        "font_choices": {"type": "object"},
        "speakers": {"type": "object"},
        "characters": {"type": "array", "items": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "note": {"type": "string"}},
            "required": ["name"],
        }},
        "address": {"type": "array", "items": {
            "type": "object",
            "properties": {k: {"type": "string"} for k in ("from", "to", "self", "other")},
            "required": ["from", "to"],
        }},
        "keep": {"type": "array", "items": {"type": "string"}},
        "missed": {"type": "array", "items": {
            "type": "object",
            "properties": {"box_2d": {"type": "array", "items": {"type": "number"}}, "text": {"type": "string"}},
            "required": ["box_2d"],
        }},
    },
    "required": ["translations"],
}
# Missed-text boxes the model reports; anything larger or more numerous is a misread.
MISSED_MAX_BOXES = 8
MISSED_MIN_SIDE_PX = 10
MISSED_MAX_AREA_RATIO = 0.4


@dataclass(frozen=True)
class VisionTranslationResult:
    translations: dict[str, str]
    model: str
    usage: dict
    estimated_cost_usd: float | None
    font_choices: dict[str, dict] = field(default_factory=dict)
    roles: dict[str, str] = field(default_factory=dict)
    review_ids: frozenset[str] = frozenset()
    # Objects the model did not answer for at all (as opposed to answering "").
    missing_ids: frozenset[str] = frozenset()
    # Objects that are part of the artwork and should stay as drawn.
    keep_ids: frozenset[str] = frozenset()
    # Source text still visible in the cleaned slice with no object: (x1, y1, x2, y2, text).
    missed_boxes: tuple[tuple[int, int, int, int, str], ...] = ()


def _font_catalog_hint(target_lang: str) -> str:
    """Compact role -> font_id list; Vietnamese output only gets fonts with its diacritics."""
    try:
        records = load_font_catalog().records
    except (OSError, ValueError):
        return "{}"
    vietnamese = str(target_lang or "").lower() in {"vi", "vie", "vietnamese"}
    groups: dict[str, list[str]] = {}
    for record in sorted(records, key=lambda item: (item.category, item.default_rank, item.id)):
        if vietnamese and not record.vietnamese:
            continue
        groups.setdefault(record.category, []).append(record.id)
    return json.dumps(groups, ensure_ascii=False, separators=(",", ":"))


def parse_vision_translation(content: str, expected_ids: set[str], *, allow_missing: bool = False) -> dict[str, str]:
    source = str(content or "").strip()
    if source.startswith(chr(96) * 3):
        source = source.split("\n", 1)[-1].rsplit(chr(96) * 3, 1)[0].strip()
    try:
        data = json.loads(source)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Vision model returned invalid JSON") from exc
    entries = data.get("translations") if isinstance(data, dict) else None
    if isinstance(entries, dict):
        entries = [{"id": key, "translated_text": value} for key, value in entries.items()]
    if isinstance(entries, list):
        entries = [
            {**entry, "id": str(entry["id"])}
            if isinstance(entry, dict) and isinstance(entry.get("id"), int) and not isinstance(entry.get("id"), bool)
            else entry
            for entry in entries
        ]
    if not isinstance(entries, list):
        shape = sorted(data)[:8] if isinstance(data, dict) else type(data).__name__
        raise RuntimeError(f"Vision model must return a translations array (got {shape})")
    results: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            raise RuntimeError("Vision model returned an invalid object ID")
        obj_id = entry["id"]
        value = entry.get("translated_text")
        if allow_missing and (obj_id not in expected_ids or obj_id in results):
            continue
        if obj_id not in expected_ids or obj_id in results or not isinstance(value, str):
            raise RuntimeError("Vision model returned an unknown, repeated or malformed translation")
        if len(value) > 4000:
            raise RuntimeError("Vision model returned oversized text")
        results[obj_id] = value.strip()
    if set(results) != expected_ids:
        if not allow_missing or not results:
            raise RuntimeError("Vision model omitted one or more text-object IDs")
        results.update({item_id: "" for item_id in expected_ids - set(results)})
    return results


def _parse_vision_payload(content: str, expected_ids: set[str]) -> tuple[dict[str, str], dict[str, dict], dict]:
    source = str(content or "").strip()
    if source.startswith(chr(96) * 3):
        source = source.split("\n", 1)[-1].rsplit(chr(96) * 3, 1)[0].strip()
    try:
        data = json.loads(source)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Vision model returned invalid JSON") from exc
    translations = parse_vision_translation(source, expected_ids, allow_missing=True)
    if isinstance(data, dict) and isinstance(data.get("translations"), dict):
        data["translations"] = [{"id": str(k), "translated_text": v} for k, v in data["translations"].items()]
    choices: dict[str, dict] = {}
    raw_choices = data.get("font_choices") if isinstance(data, dict) else None
    if isinstance(raw_choices, dict):
        for item_id, choice in raw_choices.items():
            if item_id not in expected_ids or not isinstance(choice, dict):
                continue
            font_id, font_mode = choice.get("font_id"), choice.get("font_mode", "ai")
            if isinstance(font_id, str) and isinstance(font_mode, str):
                choices[str(item_id)] = {"font_id": font_id.strip(), "font_mode": font_mode.strip().lower() or "ai"}
    return translations, choices, data if isinstance(data, dict) else {}


def _unalias(result: VisionTranslationResult, data: dict, real: dict[str, str]) -> tuple[VisionTranslationResult, dict]:
    back = lambda key: real.get(str(key), str(key))  # noqa: E731
    data = dict(data)
    if isinstance(data.get("translations"), list):
        data["translations"] = [
            {**entry, "id": back(entry.get("id"))} if isinstance(entry, dict) else entry
            for entry in data["translations"]
        ]
    for key in ("speakers", "font_choices"):
        if isinstance(data.get(key), dict):
            data[key] = {back(k): v for k, v in data[key].items()}
    if isinstance(data.get("keep"), list):
        data["keep"] = [back(item) for item in data["keep"] if isinstance(item, (str, int)) and not isinstance(item, bool)]
    return replace(
        result,
        translations={back(k): v for k, v in result.translations.items()},
        font_choices={back(k): v for k, v in (result.font_choices or {}).items()},
    ), data


def _missed_boxes(raw, width: int, height: int) -> tuple[tuple[int, int, int, int, str], ...]:
    """Pixel boxes for source text the model still sees in the cleaned slice (box_2d is 0-1000)."""
    boxes = []
    for entry in raw if isinstance(raw, list) else []:
        box = entry.get("box_2d") if isinstance(entry, dict) else None
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        try:
            ymin, xmin, ymax, xmax = (max(0.0, min(1000.0, float(v))) for v in box)
        except (TypeError, ValueError):
            continue
        x1, x2 = round(xmin / 1000 * width), round(xmax / 1000 * width)
        y1, y2 = round(ymin / 1000 * height), round(ymax / 1000 * height)
        if x2 - x1 < MISSED_MIN_SIDE_PX or y2 - y1 < MISSED_MIN_SIDE_PX:
            continue
        if (x2 - x1) * (y2 - y1) > MISSED_MAX_AREA_RATIO * width * height:
            continue
        boxes.append((x1, y1, x2, y2, str(entry.get("text") or "")[:200]))
        if len(boxes) >= MISSED_MAX_BOXES:
            break
    return tuple(boxes)


class VisionPageTranslator:
    def __init__(self, provider: AIProvider, model: str):
        if not provider.supports_visual_qc:
            raise ValueError(f"{provider.label} cannot receive image inputs")
        self.provider, self.model = provider, model

    def translate_page(
        self, original_path: Path, cleaned_path: Path, items: list[dict],
        *, api_key: str, source_lang: str, target_lang: str,
        memory: ChapterMemory | None = None, slice_number: int | None = None, slice_total: int | None = None,
        repair: bool = True,
    ) -> VisionTranslationResult:
        if not api_key.strip():
            raise ValueError(f"{self.provider.label} API key is not configured")
        if not items:
            return VisionTranslationResult({}, self.model, {}, 0.0 if self.provider.tracks_cost else None)
        original, cleaned = _read_image(original_path), _read_image(cleaned_path)
        if original.shape[:2] != cleaned.shape[:2]:
            raise ValueError("Original and cleaned slices have different dimensions")
        h, w = original.shape[:2]
        # Send short numeric ids; models copy them more reliably.
        real = {str(n): str(item["id"]) for n, item in enumerate(items, start=1)}
        objects = [{"id": alias, "source_text": item["text"], "bbox_xyxy": item["region"]}
                   for alias, item in zip(real, items)]
        source_name = (
            "the original language shown in the image"
            if str(source_lang or "").lower() in {"", "auto"}
            else _language_name(source_lang)
        )
        system = system_prompt(_language_name(target_lang), target_lang, _font_catalog_hint(target_lang), repair=repair)
        where = f"SLICE {slice_number} of {slice_total}. " if slice_number and slice_total else ""
        prompt = (
            (f"CHAPTER MEMORY (read-only context from earlier slices; never copy it into your answer): "
             f"{json.dumps(memory.snapshot(), ensure_ascii=False, separators=(',', ':'))}\n\n"
             if memory is not None else "")
            + f"{where}Translate these text objects from {source_name}.\n"
            + json.dumps({"image_width": w, "image_height": h, "objects": objects},
                         ensure_ascii=False, separators=(",", ":"))
            + '\n\nAnswer with one JSON object that starts with {"translations":[ and contains every id above.'
        )
        original_b64, cleaned_b64 = _encode_for_gemini(original), _encode_for_gemini(cleaned)
        ids = set(real)
        max_tokens = min(4096, max(1200, 160 * len(items) + 700))
        if self.provider.protocol == "gemini":
            result, data = self._gemini(system, prompt, original_b64, cleaned_b64, api_key=api_key, ids=ids, max_tokens=max_tokens)
        else:
            result, data = self._openai(system, prompt, original_b64, cleaned_b64, api_key=api_key, ids=ids, max_tokens=max_tokens)
        result, data = _unalias(result, data, real)
        ids = set(real.values())
        if memory is not None:
            memory.update(slice_number or 0, data, result.translations, [str(item["id"]) for item in items])
        roles, review, answered = {}, set(), set()
        for entry in data.get("translations") or []:
            if not isinstance(entry, dict) or str(entry.get("id")) not in ids:
                continue
            answered.add(str(entry["id"]))
            role = str(entry.get("role") or "").strip().lower()
            if role in TYPOGRAPHY_ROLES:
                roles[str(entry["id"])] = role
            if entry.get("review") is True:
                review.add(str(entry["id"]))
        keep = frozenset(str(item) for item in data.get("keep") or [] if str(item) in ids) if repair else frozenset()
        return replace(
            result, roles=roles, review_ids=frozenset(review), missing_ids=frozenset(ids - answered),
            keep_ids=keep, missed_boxes=_missed_boxes(data.get("missed"), w, h) if repair else (),
        )

    def _openai(self, system, prompt, original, cleaned, *, api_key, ids, max_tokens):
        url = str(self.provider.chat_url or "")
        if not url:
            raise ValueError("Provider does not have chat completions")
        if not self.provider.builtin:
            validate_url(url)
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "text", "text": "IMAGE 1: ORIGINAL"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{original}"}},
                {"type": "text", "text": "IMAGE 2: CLEAN"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{cleaned}"}},
            ]}],
            "response_format": {"type": "json_object"},
            "max_tokens": max_tokens,
            "stream": False,
        }
        payload.update(self.provider.chat_completion_extras())
        try:
            response = requests.post(
                url,
                headers={"Authorization": f"Bearer {api_key.strip()}", "Content-Type": "application/json"},
                json=payload,
                timeout=(TRANSLATION_CONNECT_TIMEOUT_SECONDS, TRANSLATION_READ_TIMEOUT_SECONDS),
                allow_redirects=False,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"{self.provider.label} vision request failed") from exc
        if 300 <= response.status_code < 400:
            raise RuntimeError(f"{self.provider.label} redirected the image request")
        if not response.ok:
            raise RuntimeError(
                f"{self.provider.label} HTTP {response.status_code}: "
                f"{_safe_error_detail(response, api_key.strip())}"
            )
        try:
            body = response.json()
            answer = _extract_output_text(body)
        except (TypeError, KeyError, ValueError) as exc:
            choice = (body.get("choices") or [{}])[0] if isinstance(body, dict) else {}
            reason = choice.get("finish_reason") if isinstance(choice, dict) else None
            raise RuntimeError(f"{self.provider.label} returned no translation text ({exc}; finish_reason={reason})") from exc
        translations, font_choices, data = _parse_vision_payload(answer, ids)
        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        cost = _usage_cost_usd(usage) if self.provider.tracks_cost else None
        return VisionTranslationResult(translations, str(body.get("model") or self.model), usage, cost, font_choices), data

    def _gemini(self, system, prompt, original, cleaned, *, api_key, ids, max_tokens):
        payload = {
            "model": self.model, "store": False,
            "input": [
                {"type": "text", "text": system},
                {"type": "text", "text": prompt},
                {"type": "text", "text": "IMAGE 1: ORIGINAL"},
                {"type": "image", "data": original, "mime_type": "image/jpeg"},
                {"type": "text", "text": "IMAGE 2: CLEAN"},
                {"type": "image", "data": cleaned, "mime_type": "image/jpeg"},
            ],
            "response_format": {"type": "text", "mime_type": "application/json", "schema": _TRANSLATIONS_SCHEMA},
            "generation_config": {"thinking_level": "low", "max_output_tokens": max_tokens},
        }
        try:
            response = requests.post(
                GEMINI_INTERACTIONS_URL,
                headers={"Content-Type": "application/json", "x-goog-api-key": api_key.strip(), "Api-Revision": "2026-05-20"},
                json=payload,
                timeout=(TRANSLATION_CONNECT_TIMEOUT_SECONDS, TRANSLATION_READ_TIMEOUT_SECONDS),
                allow_redirects=False,
            )
        except requests.RequestException as exc:
            raise RuntimeError("Gemini vision translation failed") from exc
        if not response.ok:
            raise RuntimeError(f"Gemini HTTP {response.status_code}: {gemini_safe_error(response)}")
        try:
            body = response.json()
            answer = gemini_output_text(body)
        except (TypeError, KeyError, ValueError) as exc:
            raise RuntimeError("Gemini returned no translation text") from exc
        translations, font_choices, data = _parse_vision_payload(answer, ids)
        return VisionTranslationResult(
            translations, self.model,
            body.get("usage", {}) if isinstance(body.get("usage"), dict) else {}, None,
            font_choices,
        ), data
