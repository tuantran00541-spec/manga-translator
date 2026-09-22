"""Two-image vision translation. Model output is restricted to existing object IDs."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import requests

from app.ai_providers import AIProvider
from app.parameters import TRANSLATION_CONNECT_TIMEOUT_SECONDS, TRANSLATION_READ_TIMEOUT_SECONDS
from app.security import validate_url
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
    },
    "required": ["translations"],
}


@dataclass(frozen=True)
class VisionTranslationResult:
    translations: dict[str, str]
    model: str
    usage: dict
    estimated_cost_usd: float | None
    font_choices: dict[str, dict] = field(default_factory=dict)


def parse_vision_translation(content: str, expected_ids: set[str]) -> dict[str, str]:
    """Never assign a model answer to a different text object."""
    source = str(content or "").strip()
    if source.startswith(chr(96) * 3):
        source = source.split("\n", 1)[-1].rsplit(chr(96) * 3, 1)[0].strip()
    try:
        data = json.loads(source)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Vision model returned invalid JSON") from exc
    entries = data.get("translations") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        raise RuntimeError("Vision model must return a translations array")
    results: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            raise RuntimeError("Vision model returned an invalid object ID")
        obj_id = entry["id"]
        value = entry.get("translated_text")
        if obj_id not in expected_ids or obj_id in results or not isinstance(value, str):
            raise RuntimeError("Vision model returned an unknown, repeated or malformed translation")
        if len(value) > 4000:
            raise RuntimeError("Vision model returned oversized text")
        results[obj_id] = value.strip()
    if set(results) != expected_ids:
        raise RuntimeError("Vision model omitted one or more text-object IDs")
    return results


def _parse_vision_payload(content: str, expected_ids: set[str]) -> tuple[dict[str, str], dict[str, dict]]:
    """Parse translations and optional AI font choices without widening IDs."""
    source = str(content or "").strip()
    if source.startswith(chr(96) * 3):
        source = source.split("\n", 1)[-1].rsplit(chr(96) * 3, 1)[0].strip()
    try:
        data = json.loads(source)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Vision model returned invalid JSON") from exc
    translations = parse_vision_translation(source, expected_ids)
    choices: dict[str, dict] = {}
    raw_choices = data.get("font_choices") if isinstance(data, dict) else None
    if isinstance(raw_choices, dict):
        for item_id, choice in raw_choices.items():
            if item_id not in expected_ids or not isinstance(choice, dict):
                continue
            font_id, font_mode = choice.get("font_id"), choice.get("font_mode", "ai")
            if isinstance(font_id, str) and isinstance(font_mode, str):
                choices[str(item_id)] = {"font_id": font_id.strip(), "font_mode": font_mode.strip().lower() or "ai"}
    return translations, choices


class VisionPageTranslator:
    """Use the existing visual-QC image encoding and provider authentication."""

    def __init__(self, provider: AIProvider, model: str):
        if not provider.supports_visual_qc:
            raise ValueError(f"{provider.label} cannot receive image inputs")
        self.provider, self.model = provider, model

    def translate_page(
        self, original_path: Path, cleaned_path: Path, items: list[dict],
        *, api_key: str, source_lang: str, target_lang: str,
    ) -> VisionTranslationResult:
        if not api_key.strip():
            raise ValueError(f"{self.provider.label} API key is not configured")
        if not items:
            return VisionTranslationResult({}, self.model, {}, 0.0 if self.provider.tracks_cost else None)
        original, cleaned = _read_image(original_path), _read_image(cleaned_path)
        if original.shape[:2] != cleaned.shape[:2]:
            raise ValueError("Original and cleaned slices have different dimensions")
        h, w = original.shape[:2]
        objects = [{"id": item["id"], "source_text": item["text"], "bbox_xyxy": item["region"]} for item in items]
        prompt = (
            "Translate manga/manhwa/webtoon text naturally and accurately from "
            f"{_language_name(source_lang)} to {_language_name(target_lang)}. "
            "Image 1 is ORIGINAL (source text); image 2 is CLEAN (after inpainting). "
            "They are the SAME slice. Use ORIGINAL and the supplied bbox_xyxy in image pixels "
            "to read the text, and both images for scene, speaker and reading-order context. "
            "source_text is an optional OCR hint; it can be blank or wrong. "
            "Keep dialogue concise to fit its existing bubble. "
            "Only translate the listed text objects. Never change or invent IDs. "
            "Do not return geometry, color, size or images: they are already stored. "
            "You may optionally return font_choices with an installed catalog font_id and font_mode=ai. "
            "If a glyph is genuinely unreadable, return translated_text empty for its ID. "
            "Return JSON only as "
            '{"translations":[{"id":"existing id","translated_text":"translated text"}],"font_choices":{"existing id":{"font_id":"catalog id","font_mode":"ai"}}}.'
            "\n" + json.dumps(
                {"image_width": w, "image_height": h, "objects": objects},
                ensure_ascii=False, separators=(",", ":"),
            )
        )
        original_b64, cleaned_b64 = _encode_for_gemini(original), _encode_for_gemini(cleaned)
        ids = {str(item["id"]) for item in items}
        max_tokens = min(4096, max(1200, 140 * len(items) + 400))
        if self.provider.protocol == "gemini":
            return self._gemini(prompt, original_b64, cleaned_b64, api_key=api_key, ids=ids, max_tokens=max_tokens)
        return self._openai(prompt, original_b64, cleaned_b64, api_key=api_key, ids=ids, max_tokens=max_tokens)

    def _openai(self, prompt, original, cleaned, *, api_key, ids, max_tokens):
        url = str(self.provider.chat_url or "")
        if not url:
            raise ValueError("Provider does not have chat completions")
        if not self.provider.builtin:
            validate_url(url)
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": [
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
            raise RuntimeError(f"{self.provider.label} returned no translation text") from exc
        translations, font_choices = _parse_vision_payload(answer, ids)
        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        cost = _usage_cost_usd(usage) if self.provider.tracks_cost else None
        return VisionTranslationResult(translations, str(body.get("model") or self.model), usage, cost, font_choices)

    def _gemini(self, prompt, original, cleaned, *, api_key, ids, max_tokens):
        payload = {
            "model": self.model, "store": False,
            "input": [
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
        translations, font_choices = _parse_vision_payload(answer, ids)
        return VisionTranslationResult(
            translations, self.model,
            body.get("usage", {}) if isinstance(body.get("usage"), dict) else {}, None,
            font_choices,
        )
