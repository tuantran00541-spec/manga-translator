import json
import os
import uuid
from contextlib import contextmanager
from fastapi import HTTPException
from filelock import FileLock, Timeout

from app.config import PROCESSED_DIR
from app.security import validate_chapter_id

_MANIFEST_LOCK_TIMEOUT = 30
_PAGE_LOCK_TIMEOUT = 60


@contextmanager
def get_manifest_lock(chapter_id: str):
    lock_path = PROCESSED_DIR / chapter_id / "manifest.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(lock_path)
    try:
        with lock.acquire(timeout=_MANIFEST_LOCK_TIMEOUT):
            yield
    except Timeout:
        raise RuntimeError(f"Manifest lock timeout for chapter {chapter_id}")


@contextmanager
def get_page_lock(chapter_id: str, page_index: int):
    lock_path = PROCESSED_DIR / chapter_id / f"page_{page_index:03d}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(lock_path)
    try:
        with lock.acquire(timeout=_PAGE_LOCK_TIMEOUT):
            yield
    except Timeout:
        raise RuntimeError(f"Page lock timeout for chapter {chapter_id} page {page_index}")


def load_manifest_raw(chapter_id: str) -> dict:
    validate_chapter_id(chapter_id)
    manifest_path = PROCESSED_DIR / chapter_id / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(404, f"Chapter {chapter_id} not found")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def save_manifest_raw(chapter_id: str, manifest: dict) -> None:
    processed_dir = PROCESSED_DIR / chapter_id
    processed_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = processed_dir / f"manifest.json.{uuid.uuid4().hex}.tmp"
    final_path = processed_dir / "manifest.json"
    tmp_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(tmp_path, final_path)


def urlify_manifest(manifest: dict) -> dict:
    result = json.loads(json.dumps(manifest))
    chapter_id = result["chapter_id"]
    for i, page in enumerate(result["pages"]):
        page["original"] = f"/api/image/{chapter_id}/{i}/original"
        if page.get("clean"):
            page["clean"] = f"/api/image/{chapter_id}/{i}/clean"
        for box in page.get("boxes", []):
            box.pop("mask", None)
    return result


def invalidate_page_render(manifest: dict, page_index: int) -> None:
    pages = manifest.get("pages", [])
    if 0 <= page_index < len(pages):
        pages[page_index]["rendered"] = False
