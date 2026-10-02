"""Language names and DeepSeek token pricing for translation calls."""
from __future__ import annotations

from app.parameters import (
    DEEPSEEK_INPUT_CACHE_HIT_USD_PER_M as INPUT_CACHE_HIT_USD_PER_M,
    DEEPSEEK_INPUT_CACHE_MISS_USD_PER_M as INPUT_CACHE_MISS_USD_PER_M,
    DEEPSEEK_OUTPUT_USD_PER_M as OUTPUT_USD_PER_M,
    TRANSLATION_PREFLIGHT_MIN_TOKENS,
    TRANSLATION_PREFLIGHT_OUTPUT_MULTIPLIER,
    TRANSLATION_PREFLIGHT_PROMPT_OVERHEAD,
)


def language_name(code: str) -> str:
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


def usage_cost_usd(usage: dict) -> float:
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


def preflight_cost_usd(items: list[dict]) -> float:
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
