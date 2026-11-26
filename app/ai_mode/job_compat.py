from __future__ import annotations

import importlib
import sys
from typing import Any

_MISSING = object()


def _j(name: str, fallback: Any = _MISSING) -> Any:
    """Resolve an attribute on ``app.ai_mode.job`` at call time.

    Preserves monkeypatching of module-level symbols on ``app.ai_mode.job``
    from helper mixins without creating circular import cycles.
    """
    job_mod = sys.modules.get("app.ai_mode.job")
    if job_mod is None:
        job_mod = importlib.import_module("app.ai_mode.job")
    value = getattr(job_mod, name, _MISSING)
    if value is not _MISSING:
        return value
    if fallback is not _MISSING:
        return fallback
    raise AttributeError(f"module 'app.ai_mode.job' has no attribute {name!r}")
