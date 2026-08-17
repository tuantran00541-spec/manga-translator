import copy
import json
import os
import uuid
from contextlib import contextmanager
from pathlib import Path
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


def _get_manual_mask_state(processed_dir: Path, img_path: Path) -> tuple[bool, int, int]:
    if not img_path.name:
        return (False, 0, 0)
    mask_path = processed_dir / f"manual_mask_{img_path.name}"
    try:
        st = mask_path.stat()
        return (True, st.st_mtime_ns, st.st_size)
    except OSError:
        return (False, 0, 0)


def capture_processing_state(manifest: dict, page_index: int, processed_dir: Path) -> dict | None:
    """Capture snapshot of canonical inputs for page detection and inpainting.

    Derived outputs (such as 'clean') are deliberately excluded to prevent
    self-referential validation.
    """
    pages = manifest.get("pages", [])
    if page_index < 0 or page_index >= len(pages):
        return None
    page = pages[page_index]
    img_path = Path(page.get("original", ""))
    return {
        "original": page.get("original"),
        "skipped": page.get("skipped", False),
        "excluded_regions": copy.deepcopy(page.get("excluded_regions", [])),
        "boxes": copy.deepcopy(page.get("boxes", [])),
        "manual_mask": page.get("manual_mask"),
        "disk_mask_state": _get_manual_mask_state(processed_dir, img_path),
    }


def is_processing_state_current(
    manifest: dict, page_index: int, snapshot: dict | None, processed_dir: Path
) -> bool:
    """Check whether a page's canonical processing inputs still match the snapshot."""
    if snapshot is None:
        return False
    current = capture_processing_state(manifest, page_index, processed_dir)
    return current == snapshot
