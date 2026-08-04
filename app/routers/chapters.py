"""API Router for chapter creation, listing, upload, and page processing."""

import os
import json
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from app.config import PROCESSED_DIR
from app.dependencies import pipeline
from app.logging_config import logger
from app.manifest_utils import get_manifest_lock, load_manifest_raw, save_manifest_raw, urlify_manifest
from app.schemas import (
    ChapterRequest,
    ProcessPagesRequest,
    SaveExcludedRegionsRequest,
    SkipPagesRequest,
)
from app.security import (
    MAX_UPLOAD_FILES,
    MAX_UPLOAD_TOTAL_BYTES,
    validate_chapter_id,
    validate_url,
)

router = APIRouter(prefix="/api", tags=["chapters"])


def _clamp_workers(n: int | None) -> int:
    try:
        v = int(n) if n is not None else 2
    except (TypeError, ValueError):
        v = 2
    return max(1, min(v, 8))


@router.get("/chapters")
def list_chapters() -> list[dict]:
    chapters = []
    if not PROCESSED_DIR.exists():
        return chapters
    for d in list(PROCESSED_DIR.iterdir()):
        if not d.is_dir():
            continue
        manifest_path = d / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            mtime = manifest_path.stat().st_mtime
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            pages = manifest.get("pages", [])
            if not any(p.get("clean") for p in pages):
                continue
            chapters.append({
                "chapter_id": manifest.get("chapter_id", d.name),
                "source_url": manifest.get("source_url", ""),
                "total_pages": len(pages),
                "updated_at": mtime,
            })
        except (json.JSONDecodeError, OSError, KeyError):
            continue
    chapters.sort(key=lambda x: x["updated_at"], reverse=True)
    return chapters


@router.post("/chapter")
def create_chapter(req: ChapterRequest) -> dict:
    validate_url(req.url)
    chapter_id = os.urandom(4).hex()
    workers = _clamp_workers(req.workers)
    logger.info(f"Creating chapter {chapter_id} from {req.url} (workers={workers})")
    manifest = pipeline.download_chapter(req.url, chapter_id, workers=workers)
    return urlify_manifest(manifest)


@router.post("/chapter/upload")
async def create_chapter_from_upload(
    files: list[UploadFile] = File(...),
    workers: int = Form(2),
) -> dict:
    if not files:
        raise HTTPException(400, "No files uploaded")
    if len(files) > MAX_UPLOAD_FILES:
        raise HTTPException(400, f"Too many files: max {MAX_UPLOAD_FILES} per upload")

    uploads = []
    total_bytes = 0
    for f in files:
        data = await f.read()
        total_bytes += len(data)
        if total_bytes > MAX_UPLOAD_TOTAL_BYTES:
            raise HTTPException(413, f"Total upload size exceeds {MAX_UPLOAD_TOTAL_BYTES // (1024*1024)}MB")
        uploads.append((f.filename or "unnamed", data))

    chapter_id = os.urandom(4).hex()
    workers_n = _clamp_workers(workers)
    logger.info(f"Creating chapter {chapter_id} from {len(uploads)} uploaded files (workers={workers_n})")
    manifest = await run_in_threadpool(pipeline.create_chapter_from_uploads, chapter_id, uploads, workers=workers_n)
    return urlify_manifest(manifest)


@router.post("/process_pages")
def process_pages(req: ProcessPagesRequest) -> dict:
    validate_chapter_id(req.chapter_id)
    workers = _clamp_workers(req.workers)
    logger.info(
        f"Chapter {req.chapter_id}: processing pages {req.page_indices} (workers={workers})"
    )
    manifest = pipeline.process_pages(req.chapter_id, req.page_indices, workers=workers)
    return urlify_manifest(manifest)


@router.post("/skip_pages")
def skip_pages(req: SkipPagesRequest) -> dict:
    validate_chapter_id(req.chapter_id)
    manifest = pipeline.mark_skipped(req.chapter_id, req.page_indices, req.skipped)
    return urlify_manifest(manifest)


@router.get("/chapter/{chapter_id}")
def get_chapter(chapter_id: str) -> dict:
    return urlify_manifest(load_manifest_raw(chapter_id))


@router.post("/save_excluded_regions")
def save_excluded_regions(req: SaveExcludedRegionsRequest) -> dict:
    validate_chapter_id(req.chapter_id)
    with get_manifest_lock(req.chapter_id):
        manifest = load_manifest_raw(req.chapter_id)
        pages = manifest.get("pages", [])
        if 0 <= req.page_index < len(pages):
            pages[req.page_index]["excluded_regions"] = [r.model_dump() for r in req.excluded_regions]
            save_manifest_raw(req.chapter_id, manifest)
    return urlify_manifest(manifest)


@router.post("/chapters/{chapter_id}/pages/{page_index}/excluded-regions")
def set_page_excluded_regions(chapter_id: str, page_index: int, regions: list[dict]) -> dict:
    validate_chapter_id(chapter_id)
    with get_manifest_lock(chapter_id):
        manifest = load_manifest_raw(chapter_id)
        pages = manifest.get("pages", [])
        if 0 <= page_index < len(pages):
            pages[page_index]["excluded_regions"] = regions
            save_manifest_raw(chapter_id, manifest)
    return urlify_manifest(manifest)
