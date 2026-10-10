"""What is known about particular models: prompt advice for their quirks, prices, whether they can see images."""
from __future__ import annotations

import re

QUIRKS: list[tuple[re.Pattern, dict]] = [
    (re.compile(r"qwen", re.I), {"prompt_extra": (
        "Your tool calls must be valid JSON: write newlines inside strings as \\n and never leave a bare double quote inside a string. "
        "Keep each argument under about 1500 characters; to hand over long text, save it with write_file and give the path.")}),
    (re.compile(r"agnes", re.I), {"prompt_extra": (
        "This provider limits requests per minute: start at most three helpers at a time and prefer fewer, larger steps.")}),
]
# Dollars per million tokens; a profile's "prices" adds or replaces entries.
PRICES: list[tuple[re.Pattern, dict]] = [
    (re.compile(r"agnes", re.I), {"in": 0.05, "out": 0.15, "cached": 0.05}),
    (re.compile(r"qwen", re.I), {"in": 0.20, "out": 0.75, "cached": 0.20}),
]
VISION = re.compile(r"(vision|[-_]vl\b|4o|gpt-5|gpt-4\.1|claude|gemini|pixtral|llava|agnes)", re.I)


def _match(table, model: str, extra: dict | None) -> dict:
    found: dict = {}
    for pattern, row in table:
        if pattern.search(model):
            found.update(row)
    for key, row in (extra or {}).items():
        try:
            if re.search(key, model, re.I) and isinstance(row, dict):
                found.update(row)
        except re.error:
            continue
    return found


def quirks(model: str, overrides: dict | None = None) -> dict:
    return _match(QUIRKS, model, overrides)


def prices(model: str, overrides: dict | None = None) -> dict:
    return _match(PRICES, model, overrides)


def sees_images(provider_id: str, model: str, override: bool | None = None) -> bool:
    return override if override is not None else provider_id == "gemini" or bool(VISION.search(model))
