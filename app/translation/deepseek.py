from __future__ import annotations

import json
from dataclasses import dataclass

import requests

from app import parameters as _parameters
from app.ai_providers import AIProvider, get_translation_provider
from app.security import validate_url
from app.parameters import (
    DEEPSEEK_INPUT_CACHE_HIT_USD_PER_M as INPUT_CACHE_HIT_USD_PER_M,
    DEEPSEEK_INPUT_CACHE_MISS_USD_PER_M as INPUT_CACHE_MISS_USD_PER_M,
    DEEPSEEK_OUTPUT_USD_PER_M as OUTPUT_USD_PER_M,
    TRANSLATION_CONNECT_TIMEOUT_SECONDS,
    TRANSLATION_MAX_TOKENS,
    TRANSLATION_PREFLIGHT_MIN_TOKENS,
    TRANSLATION_PREFLIGHT_OUTPUT_MULTIPLIER,
    TRANSLATION_PREFLIGHT_PROMPT_OVERHEAD,
    TRANSLATION_READ_TIMEOUT_SECONDS,
)

PRICING_VERSION = _parameters.DEEPSEEK_PRICING_VERSION


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


class OpenAICompatibleTranslator:
    """Text translation through the provider contract used by the current UI.

    The browser supplies only provider/model. Endpoint selection and vendor-only
    request fields are resolved here from ``AIProvider`` so OpenRouter/OpenAI do
    not accidentally receive DeepSeek-specific parameters.
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        api_url: str | None = None,
        provider_id: str = "deepseek",
        provider_label: str | None = None,
        provider: AIProvider | None = None,
    ):
        provider = provider or get_translation_provider(provider_id)
        if provider.protocol != "openai" or not provider.supports_translation:
            raise ValueError(
                f"{provider.label} is not available for translation"
            )
        self.model = (model or provider.default_translation_model or "").strip()
        if not self.model:
            raise ValueError(f"{provider.label} translation model is not configured")
        self.api_url = api_url or str(provider.chat_url)
        self.provider_id = provider.id
        self.provider_label = provider_label or provider.label
        self._request_extras = provider.chat_completion_extras()
        self._priced = provider.tracks_cost
        self._validate_remote = not provider.builtin

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
            raise ValueError(f"{self.provider_label} API key is not configured")
        if not items:
            return TranslationResult({}, {}, 0.0, self.model)
        if budget_usd <= 0:
            raise ValueError("Translation budget must be greater than zero")

        preflight = _preflight_cost_usd(items) if self._priced else 0.0
        if self._priced and preflight > budget_usd:
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

        request_body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": TRANSLATION_MAX_TOKENS,
        }
        request_body.update(self._request_extras)
        if self._validate_remote:
            validate_url(self.api_url)
        response = requests.post(
            self.api_url,
            headers={
                "Authorization": f"Bearer {api_key.strip()}",
                "Content-Type": "application/json",
            },
            json=request_body,
            timeout=(
                TRANSLATION_CONNECT_TIMEOUT_SECONDS,
                TRANSLATION_READ_TIMEOUT_SECONDS,
            ),
            allow_redirects=False,
        )
        if 300 <= response.status_code < 400:
            raise RuntimeError(
                f"{self.provider_label} translation API redirect was refused"
            )
        try:
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                f"{self.provider_label} translation request failed with HTTP {response.status_code}"
            ) from exc

        try:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            raw_translations = parsed["translations"]
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"{self.provider_label} returned an invalid translation response") from exc
        if not isinstance(raw_translations, dict):
            raise RuntimeError(f"{self.provider_label} translation payload is not an object")

        expected_ids = {str(item["id"]) for item in items}
        translations: dict[str, str] = {}
        for item_id in expected_ids:
            value = raw_translations.get(item_id)
            if isinstance(value, str) and value.strip():
                translations[item_id] = value.strip()
        missing = expected_ids.difference(translations)
        if missing:
            raise RuntimeError(f"{self.provider_label} omitted {len(missing)} translation(s)")

        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        actual_cost = _usage_cost_usd(usage) if self._priced else 0.0
        return TranslationResult(
            translations=translations,
            usage=usage,
            estimated_cost_usd=actual_cost,
            model=str(data.get("model") or self.model),
        )


# Existing imports use this name; keep it as an alias while the implementation
# is now provider-neutral.
DeepSeekTranslator = OpenAICompatibleTranslator
