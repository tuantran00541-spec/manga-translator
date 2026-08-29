from __future__ import annotations

from functools import lru_cache
from importlib import metadata
import os
from pathlib import Path

OCR_PIPELINE_VERSION = "phase44-v2-ppocrv6"
OCR_CACHE_FIELDS = (
    "ocr_text",
    "ocr_lang",
    "ocr_source",
    "ocr_engine",
    "ocr_source_revision",
    "ocr_file_revision",
    "ocr_geometry",
)


def file_revision(path: Path) -> tuple[int, int, int]:
    st = path.stat()
    return (int(st.st_size), int(st.st_mtime_ns), int(st.st_ctime_ns))


def geometry_signature(box: dict) -> tuple[int, int, int, int]:
    return tuple(int(box.get(key, 0)) for key in ("x1", "y1", "x2", "y2"))


@lru_cache(maxsize=8)
def _package_version(package_name: str) -> str:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return "unknown"


def _env_enabled(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=16)
def engine_identity(lang: str) -> str:
    normalized = (lang or "").strip().lower()
    tier = os.getenv("MANGA_PPOCRV6_TIER", "small").strip().lower()
    if tier not in {"small", "medium"}:
        tier = "small"
    paddle_version = _package_version("paddleocr")

    if normalized in {"ko", "korean"}:
        backend = f"paddleocr:{paddle_version}:korean-ppocrv5-mobile"
    elif normalized in {"ja", "japan"}:
        backend = f"paddleocr:{paddle_version}:ppocrv6-{tier}"
        if _env_enabled("MANGA_OCR_JA_FALLBACK", True):
            backend += f"+manga-ocr-fallback:{_package_version('manga-ocr')}"
    else:
        backend = f"paddleocr:{paddle_version}:ppocrv6-{tier}"
    return f"{OCR_PIPELINE_VERSION}:{backend}"


def clear_ocr_cache(record: dict) -> None:
    for key in OCR_CACHE_FIELDS:
        record.pop(key, None)


def machine_cache_valid(
    box: dict,
    *,
    lang: str,
    engine: str,
    source_revision: int,
    original_revision: tuple[int, int, int],
) -> bool:
    if box.get("ocr_source") != "machine":
        return False
    if box.get("ocr_lang") != lang or box.get("ocr_engine") != engine:
        return False
    try:
        cached_source_revision = int(box.get("ocr_source_revision"))
    except (TypeError, ValueError):
        return False
    if cached_source_revision != int(source_revision):
        return False
    try:
        cached_file_revision = tuple(int(value) for value in box.get("ocr_file_revision", ()))
    except (TypeError, ValueError):
        return False
    if cached_file_revision != tuple(original_revision):
        return False
    try:
        cached_geometry = tuple(int(value) for value in box.get("ocr_geometry", ()))
    except (TypeError, ValueError):
        return False
    return cached_geometry == geometry_signature(box)


def stamp_machine_cache(
    box: dict,
    *,
    text: str,
    lang: str,
    engine: str,
    source_revision: int,
    original_revision: tuple[int, int, int],
) -> None:
    box["ocr_text"] = text or ""
    box["ocr_lang"] = lang
    box["ocr_source"] = "machine"
    box["ocr_engine"] = engine
    box["ocr_source_revision"] = int(source_revision)
    box["ocr_file_revision"] = [int(value) for value in original_revision]
    box["ocr_geometry"] = list(geometry_signature(box))
