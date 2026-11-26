from __future__ import annotations

import threading
from typing import Any

_INIT_LOCK = threading.Lock()
_CPU_CONFIGURED = False
_EXPORTS = ("pipeline", "ocr")


def _ensure_cpu_headroom() -> None:
    global _CPU_CONFIGURED
    if _CPU_CONFIGURED:
        return
    from app.runtime_responsiveness import configure_local_cpu_headroom

    configure_local_cpu_headroom()
    _CPU_CONFIGURED = True


def get_pipeline():
    existing = globals().get("pipeline")
    if existing is not None:
        return existing
    with _INIT_LOCK:
        existing = globals().get("pipeline")
        if existing is None:
            _ensure_cpu_headroom()
            from app.processing_pipeline_factory import build_processing_pipeline

            existing = build_processing_pipeline()
            globals()["pipeline"] = existing
        return existing


def get_ocr():
    existing = globals().get("ocr")
    if existing is not None:
        return existing
    with _INIT_LOCK:
        existing = globals().get("ocr")
        if existing is None:
            _ensure_cpu_headroom()
            from app.ocr.multi_lang_ocr import MultiLangOCR

            existing = MultiLangOCR()
            globals()["ocr"] = existing
        return existing


def __getattr__(name: str) -> Any:
    if name == "pipeline":
        return get_pipeline()
    if name == "ocr":
        return get_ocr()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))
