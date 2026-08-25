from __future__ import annotations

import hashlib
import json
from pathlib import Path

RENDER_IDENTITY_VERSION = "phase45-v1"


def file_revision(path: Path) -> tuple[int, int, int]:
    st = path.stat()
    return (int(st.st_size), int(st.st_mtime_ns), int(st.st_ctime_ns))


def _normalized_box(box: dict) -> dict:
    return {
        "id": str(box.get("id") or ""),
        "x1": int(box.get("x1", 0)),
        "y1": int(box.get("y1", 0)),
        "x2": int(box.get("x2", 0)),
        "y2": int(box.get("y2", 0)),
        "removed": bool(box.get("removed", False)),
    }


def _normalized_text_object(obj: dict) -> dict:
    region = obj.get("region") or {}
    return {
        "id": str(obj.get("id") or ""),
        "shape": str(obj.get("shape") or "rectangle"),
        "region": {
            "x1": int(region.get("x1", 0)),
            "y1": int(region.get("y1", 0)),
            "x2": int(region.get("x2", 0)),
            "y2": int(region.get("y2", 0)),
        },
        "translation": str(obj.get("translation") or ""),
        "style": obj.get("style") or {},
    }


def _page_drafts(manifest: dict, page_index: int) -> dict:
    prefix = f"{page_index}_"
    drafts = manifest.get("drafts") or {}
    return {
        str(key): value
        for key, value in drafts.items()
        if str(key).startswith(prefix)
    }


def render_input_signature(manifest: dict, page_index: int) -> str:
    pages = manifest.get("pages", [])
    if page_index < 0 or page_index >= len(pages):
        raise IndexError(page_index)
    page = pages[page_index]
    base_value = page.get("clean") or page.get("original")
    if not base_value:
        raise FileNotFoundError("Page has no render base image")
    base_path = Path(str(base_value))
    if not base_path.is_file():
        raise FileNotFoundError(base_path)

    text_objects = [obj for obj in (page.get("text_objects") or []) if isinstance(obj, dict)]
    if text_objects:
        render_content = {
            "mode": "text_objects",
            "objects": [_normalized_text_object(obj) for obj in text_objects],
        }
    else:
        render_content = {
            "mode": "legacy_boxes",
            "boxes": [
                _normalized_box(box)
                for box in (page.get("boxes") or [])
                if isinstance(box, dict)
            ],
            "drafts": _page_drafts(manifest, page_index),
        }

    payload = {
        "version": RENDER_IDENTITY_VERSION,
        "original": page.get("original"),
        "clean": page.get("clean"),
        "base_revision": list(file_revision(base_path)),
        "source_revision": int(page.get("source_revision") or 0),
        "process_revision": int(page.get("process_revision") or 0),
        "clean_revision": int(page.get("clean_revision") or 0),
        "skipped": bool(page.get("skipped", False)),
        "content": render_content,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stamp_render_artifact(page: dict, *, input_signature: str, output_path: Path) -> None:
    page["rendered"] = True
    page["render_identity_version"] = RENDER_IDENTITY_VERSION
    page["render_input_signature"] = input_signature
    page["render_output_revision"] = list(file_revision(output_path))


def render_artifact_is_current(
    manifest: dict,
    page_index: int,
    output_path: Path,
) -> bool:
    pages = manifest.get("pages", [])
    if page_index < 0 or page_index >= len(pages):
        return False
    page = pages[page_index]
    if not page.get("rendered"):
        return False
    if page.get("render_identity_version") != RENDER_IDENTITY_VERSION:
        return False
    expected_signature = page.get("render_input_signature")
    expected_output_revision = page.get("render_output_revision")
    if not expected_signature or not isinstance(expected_output_revision, (list, tuple)):
        return False
    try:
        current_signature = render_input_signature(manifest, page_index)
        current_output_revision = file_revision(output_path)
        stored_output_revision = tuple(int(value) for value in expected_output_revision)
    except (IndexError, OSError, TypeError, ValueError):
        return False
    return (
        current_signature == expected_signature
        and current_output_revision == stored_output_revision
    )
