from __future__ import annotations

import json
from dataclasses import dataclass

import requests

from app.parameters import (
    DEEPSEEK_API_URL,
    DEEPSEEK_INPUT_CACHE_HIT_USD_PER_M as INPUT_CACHE_HIT_USD_PER_M,
    DEEPSEEK_INPUT_CACHE_MISS_USD_PER_M as INPUT_CACHE_MISS_USD_PER_M,
    DEEPSEEK_OUTPUT_USD_PER_M as OUTPUT_USD_PER_M,
    DEEPSEEK_PRICING_VERSION as PRICING_VERSION,
    DEEPSEEK_TRANSLATION_MODEL as DEFAULT_TRANSLATION_MODEL,
    TRANSLATION_CONNECT_TIMEOUT_SECONDS,
    TRANSLATION_MAX_TOKENS,
    TRANSLATION_PREFLIGHT_MIN_TOKENS,
    TRANSLATION_PREFLIGHT_OUTPUT_MULTIPLIER,
    TRANSLATION_PREFLIGHT_PROMPT_OVERHEAD,
    TRANSLATION_READ_TIMEOUT_SECONDS,
)


class TranslationBudgetExceeded(ValueError):
    pass


@dataclass(frozen=True)
class TranslationResult:
    translations: dict[str, str]
    usage: dict
    estimated_cost_usd: float
    model: str


def _language_name(code: str) -> str:
    names = {
        "ja": "Japanese",
        "japan": "Japanese",
        "ch": "Chinese",
        "zh": "Chinese",
        "korean": "Korean",
        "ko": "Korean",
        "en": "English",
        "vi": "Vietnamese",
        "th": "Thai",
        "id": "Indonesian",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "pt": "Portuguese",
    }
    return names.get((code or "").lower(), code or "auto-detected")


def _usage_cost_usd(usage: dict) -> float:
    prompt = max(0, int(usage.get("prompt_tokens") or 0))
    hit = max(0, int(usage.get("prompt_cache_hit_tokens") or 0))
    miss_raw = usage.get("prompt_cache_miss_tokens")
    miss = max(0, int(miss_raw)) if miss_raw is not None else max(0, prompt - hit)
    completion = max(0, int(usage.get("completion_tokens") or 0))
    return (
        hit * INPUT_CACHE_HIT_USD_PER_M
        + miss * INPUT_CACHE_MISS_USD_PER_M
        + completion * OUTPUT_USD_PER_M
    ) / 1_000_000.0


def _preflight_cost_usd(items: list[dict]) -> float:
    source_chars = sum(len(str(item.get("text") or "")) for item in items)
    estimated_input_tokens = max(
        TRANSLATION_PREFLIGHT_MIN_TOKENS,
        source_chars + TRANSLATION_PREFLIGHT_PROMPT_OVERHEAD,
    )
    estimated_output_tokens = max(
        TRANSLATION_PREFLIGHT_MIN_TOKENS,
        int(round(source_chars * TRANSLATION_PREFLIGHT_OUTPUT_MULTIPLIER)),
    )
    return (
        estimated_input_tokens * INPUT_CACHE_MISS_USD_PER_M
        + estimated_output_tokens * OUTPUT_USD_PER_M
    ) / 1_000_000.0


class DeepSeekTranslator:
    def __init__(self, model: str = DEFAULT_TRANSLATION_MODEL):
        self.model = model

    def translate(
        self,
        items: list[dict],
        *,
        api_key: str,
        source_lang: str,
        target_lang: str,
        budget_usd: float,
    ) -> TranslationResult:
        if not api_key or not api_key.strip():
            raise ValueError("DeepSeek API key is not configured")
        if not items:
            return TranslationResult({}, {}, 0.0, self.model)
        if budget_usd <= 0:
            raise ValueError("Translation budget must be greater than zero")

        preflight = _preflight_cost_usd(items)
        if preflight > budget_usd:
            raise TranslationBudgetExceeded(
                f"Estimated translation cost ${preflight:.4f} exceeds chapter budget ${budget_usd:.4f}"
            )

        source_name = _language_name(source_lang)
        target_name = _language_name(target_lang)
        payload_items = [
            {
                "id": str(item["id"]),
                "page": int(item["page_index"]),
                "text": str(item["text"]),
            }
            for item in items
        ]
        system = (
            "You are a professional manga/manhua/webtoon localization translator. "
            f"Translate from {source_name} to {target_name}. "
            "Keep dialogue natural and concise enough for speech balloons. Preserve names, honorifics, "
            "sound-effect intent, punctuation, and line breaks when useful. Do not explain. "
            "Return valid JSON exactly as {\"translations\": {\"<id>\": \"<translated text>\"}}. "
            "Every input id must appear exactly once and no extra ids may be invented."
        )
        user = json.dumps(
            {
                "source_language": source_name,
                "target_language": target_name,
                "items": payload_items,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

        response = requests.post(
            DEEPSEEK_API_URL,
            headers={
                "Authorization": f"Bearer {api_key.strip()}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "thinking": {"type": "disabled"},
                "response_format": {"type": "json_object"},
                "max_tokens": TRANSLATION_MAX_TOKENS,
            },
            timeout=(
                TRANSLATION_CONNECT_TIMEOUT_SECONDS,
                TRANSLATION_READ_TIMEOUT_SECONDS,
            ),
        )
        try:
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"DeepSeek translation request failed with HTTP {response.status_code}") from exc

        try:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            raw_translations = parsed["translations"]
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError("DeepSeek returned an invalid translation response") from exc
        if not isinstance(raw_translations, dict):
            raise RuntimeError("DeepSeek translation payload is not an object")

        expected_ids = {str(item["id"]) for item in items}
        translations: dict[str, str] = {}
        for item_id in expected_ids:
            value = raw_translations.get(item_id)
            if isinstance(value, str) and value.strip():
                translations[item_id] = value.strip()
        missing = expected_ids.difference(translations)
        if missing:
            raise RuntimeError(f"DeepSeek omitted {len(missing)} translation(s)")

        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        actual_cost = _usage_cost_usd(usage)
        return TranslationResult(
            translations=translations,
            usage=usage,
            estimated_cost_usd=actual_cost,
            model=str(data.get("model") or self.model),
        )
