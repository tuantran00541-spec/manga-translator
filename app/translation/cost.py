"""Language names and DeepSeek token pricing for translation calls."""
from __future__ import annotations

import json
import math
import os

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


def usage_cost_usd(usage: dict, provider_id: str = "deepseek") -> float | None:
    """The call's cost at DeepSeek prices; None for another provider, whose prices are not DeepSeek's."""
    if provider_id != "deepseek":
        return None
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


def _settled_cost(value) -> float | None:
    try:
        cost = float(value)
    except (TypeError, ValueError):
        return None
    return cost if math.isfinite(cost) and cost >= 0 else None


def _configured_price_usd(usage: dict, model: str) -> float | None:
    try:
        table = json.loads(os.getenv("MANGA_MODEL_PRICES_JSON") or "{}")
    except ValueError:
        return None
    entry = table.get(model) if isinstance(table, dict) else None
    if not isinstance(entry, dict):
        return None
    in_rate = _settled_cost(entry.get("input"))
    out_rate = _settled_cost(entry.get("output"))
    if in_rate is None or out_rate is None:
        return None
    prompt = max(0, int(usage.get("prompt_tokens") or 0))
    completion = max(0, int(usage.get("completion_tokens") or 0))
    return (prompt * in_rate + completion * out_rate) / 1_000_000.0


def response_cost_usd(usage: dict, provider_id: str, model: str, headers) -> float | None:
    """One call's actual USD cost: DeepSeek's token price, then the gateway's settled charge, then a user-configured model price."""
    cost = usage_cost_usd(usage, provider_id)
    if cost is not None:
        return cost
    if headers is not None:
        cost = _settled_cost(headers.get("x-minirouter-cost-usd"))
        if cost is not None:
            return cost
    cost = _settled_cost((usage or {}).get("cost"))
    if cost is not None:
        return cost
    return _configured_price_usd(usage or {}, model)


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
