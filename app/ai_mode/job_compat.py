from __future__ import annotations

import sys


def _j(name: str, fallback=None):
    job_mod = sys.modules.get("app.ai_mode.job")
    if job_mod is not None and hasattr(job_mod, name):
        return getattr(job_mod, name)
    return fallback
