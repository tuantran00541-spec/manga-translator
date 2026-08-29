from __future__ import annotations

from functools import lru_cache
from importlib import metadata
import os
from pathlib import Path
from typing import Any

from app.env_utils import env_enabled

OCR_PIPELINE_VERSION = "phase44-v3-hybrid"
OCR_CACHE_FIELDS = (
    "ocr_text",
    "ocr_lang",
    "ocr_source",
    "ocr_engine",
    "ocr_source_revision",
    "ocr_file_revision",
    "ocr_geometry",
    "ocr_confidence",
    "ocr_model",
    "ocr_orientation",
    "ocr_region_count",
    "ocr_quality",
    "ocr_quality_reason",
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


@lru_cache(maxsize=16)
def engine_identity(lang: str) -> str:
    normalized = (lang or "").strip().lower()
    tier = os.getenv("MANGA_PPOCRV6_TIER", "small").strip().lower()
    if tier not in {"small", "medium"}:
        tier = "small"

    if normalized in {"ja", "japan"}:
        backend = f"manga-ocr:{_package_version('manga-ocr')}"
    else:
        paddle_version = _package_version("paddleocr")
        orientation = "ori-on" if env_enabled(
            "MANGA_PPOCRV6_TEXTLINE_ORIENTATION", False
        ) else "ori-off"
        if normalized in {"ko", "korean"}:
            backend = (
                f"paddleocr:{paddle_version}:korean-ppocrv5-mobile:{orientation}"
            )
        else:
            backend = f"paddleocr:{paddle_version}:ppocrv6-{tier}:{orientation}"
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


def _metadata_value(metadata: Any, key: str, default=None):
    if metadata is None:
        return default
    if isinstance(metadata, dict):
        return metadata.get(key, default)
    return getattr(metadata, key, default)


def stamp_machine_cache(
    box: dict,
    *,
    text: str,
    lang: str,
    engine: str,
    source_revision: int,
    original_revision: tuple[int, int, int],
    metadata: Any = None,
) -> None:
    box["ocr_text"] = text or ""
    box["ocr_lang"] = lang
    box["ocr_source"] = "machine"
    box["ocr_engine"] = engine
    box["ocr_source_revision"] = int(source_revision)
    box["ocr_file_revision"] = [int(value) for value in original_revision]
    box["ocr_geometry"] = list(geometry_signature(box))

    confidence = _metadata_value(metadata, "confidence")
    if confidence is None:
        box.pop("ocr_confidence", None)
    else:
        try:
            box["ocr_confidence"] = float(confidence)
        except (TypeError, ValueError):
            box.pop("ocr_confidence", None)

    model = str(_metadata_value(metadata, "model", "") or "").strip()
    if model:
        box["ocr_model"] = model
    else:
        box.pop("ocr_model", None)

    orientation = str(_metadata_value(metadata, "orientation", "") or "").strip()
    if orientation:
        box["ocr_orientation"] = orientation
    else:
        box.pop("ocr_orientation", None)

    try:
        region_count = int(_metadata_value(metadata, "region_count", 0) or 0)
    except (TypeError, ValueError):
        region_count = 0
    box["ocr_region_count"] = max(0, region_count)

    quality = str(_metadata_value(metadata, "quality", "unknown") or "unknown").strip().lower()
    box["ocr_quality"] = quality if quality in {"good", "review", "reject"} else "unknown"

    reason = str(_metadata_value(metadata, "quality_reason", "") or "").strip()
    if reason:
        box["ocr_quality_reason"] = reason
    else:
        box.pop("ocr_quality_reason", None)
