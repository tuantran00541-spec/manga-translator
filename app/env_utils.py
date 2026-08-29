from __future__ import annotations

import os
from collections.abc import Collection


def env_enabled(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_choice(
    name: str,
    *,
    default: str,
    allowed: Collection[str],
) -> str:
    """Read a normalized enum-like environment value with a safe fallback."""
    normalized_default = str(default).strip().lower()
    normalized_allowed = {str(value).strip().lower() for value in allowed}
    if normalized_default not in normalized_allowed:
        raise ValueError(f"Default {default!r} is not valid for {name}")

    value = os.getenv(name)
    if value is None:
        return normalized_default
    normalized = value.strip().lower()
    return normalized if normalized in normalized_allowed else normalized_default
