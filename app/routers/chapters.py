import json
import os
import shutil

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from filelock import FileLock, Timeout

from app.config import OUTPUT_DIR, PROCESSED_DIR, RAW_DIR
from app.dependencies import pipeline
from app.logging_config import logger
from app.manifest_utils import bump_page_revision, get_manifest_lock, get_page_lock, invalidate_page_render, load_manifest_raw, save_manifest_raw, urlify_manifest
from app.page_processing_jobs import ChapterProcessingJobManager
from app.parameters import PIPELINE_DEFAULT_WORKERS
from app.pipeline import StaleProcessingStateError
from app.schemas import (
    ChapterRequest,
    ProcessPagesRequest,
    RegionModel,
    SaveExcludedRegionsRequest,
    SavePreserveRegionsRequest,
    SkipPagesRequest,
    WorkflowCheckpointRequest,
)
from app.security import (
    MAX_RENDER_TRANSLATIONS,
    MAX_UPLOAD_FILES,
    MAX_UPLOAD_TOTAL_BYTES,
    validate_chapter_id,
    validate_url,
)
from app.upload_utils import read_upload_limited
from app.text_objects import ensure_page_text_objects

router = APIRouter(prefix="/api", tags=["chapters"])
_CHAPTER_LIST_CACHE: dict[str, tuple[tuple[int, int, int], dict]] = {}


def _allocate_chapter_id() -> str:
    """Reserve a collision-free raw directory for one new chapter."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for _attempt in range(256):
        chapter_id = os.urandom(4).hex()
        raw_dir = RAW_DIR / chapter_id
        try:
            raw_dir.mkdir()
        except FileExistsError:
            continue
        if (PROCESSED_DIR / chapter_id).exists() or (OUTPUT_DIR / chapter_id).exists():
            try:
                raw_dir.rmdir()
            except OSError:
                pass
            continue
        return chapter_id
    raise HTTPException(503, "Could not allocate chapter storage")


def _cleanup_uncommitted_chapter(chapter_id: str) -> None:
    """Remove only storage reserved for a creation that never committed."""
    manifest_path = PROCESSED_DIR / chapter_id / "manifest.json"
    if manifest_path.exists():
        return
    _CHAPTER_LIST_CACHE.pop(chapter_id, None)
    for root in (RAW_DIR, PROCESSED_DIR, OUTPUT_DIR):
        path = root / chapter_id
        try:
            if path.is_symlink():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
        except OSError as exc:
            logger.warning(
                "Could not roll back uncommitted chapter {} at {}: {}",
                chapter_id,
                path,
                exc,
            )


def _clamp_workers(n: int | None) -> int:
    try:
        v = int(n) if n is not None else PIPELINE_DEFAULT_WORKERS
    except (TypeError, ValueError):
        v = PIPELINE_DEFAULT_WORKERS
    return max(1, min(v, 8))


def _region_payload(regions: list[RegionModel]) -> list[dict]:
    if len(regions) > MAX_RENDER_TRANSLATIONS:
        raise HTTPException(400, "Too many preserve regions")
    return [region.model_dump() for region in regions]


def _manifest_revision(path) -> tuple[tuple[int, int, int], float]:
    stat = path.stat()
    return (
        (int(stat.st_size), int(stat.st_mtime_ns), int(stat.st_ctime_ns)),
        float(stat.st_mtime),
    )


def _validated_process_plan(req: ProcessPagesRequest) -> tuple[list[int], int]:
    validate_chapter_id(req.chapter_id)
    manifest_raw = load_manifest_raw(req.chapter_id)
    total_pages = len(manifest_raw.get("pages", []))
    if not req.page_indices:
        raise HTTPException(400, "page_indices cannot be empty")
    page_indices = list(req.page_indices)
    for idx in page_indices:
        if not isinstance(idx, int) or idx < 0 or idx >= total_pages:
            raise HTTPException(
                400,
                f"Invalid page_index {idx} for chapter {req.chapter_id} (total pages: {total_pages})",
            )
    return page_indices, _clamp_workers(req.workers)


def _process_job_plan(
    chapter_id: str,
    page_indices: list[int],
    workers: int,
    progress_callback,
):
    logger.info(
        "Chapter {} background job: processing {} pages in one queue (workers={})",
        chapter_id,
        len(page_indices),
        workers,
    )
    return pipeline.process_pages(
        chapter_id,
        page_indices,
        workers=workers,
        progress_callback=progress_callback,
    )


def _mark_processing_complete(chapter_id: str) -> None:
    """Make post-refresh navigation deterministic once the full job finishes."""
    with get_manifest_lock(chapter_id):
        manifest = load_manifest_raw(chapter_id)
        workflow = manifest.get("workflow") or {}
        if str(workflow.get("stage") or "preview") == "preview":
            manifest["workflow"] = {"stage": "review", "page_index": 0}
            save_manifest_raw(chapter_id, manifest)
    _CHAPTER_LIST_CACHE.pop(chapter_id, None)


chapter_processing_jobs = ChapterProcessingJobManager(
    _process_job_plan,
    on_completed=_mark_processing_complete,
)


@router.get("/chapters")
def list_chapters() -> list[dict]:
    chapters = []
    if not PROCESSED_DIR.exists():
        _CHAPTER_LIST_CACHE.clear()
        return chapters

    active_chapters: set[str] = set()
    for d in PROCESSED_DIR.iterdir():
        if not d.is_dir():
            continue
        manifest_path = d / "manifest.json"
        if not manifest_path.exists():
            continue
        active_chapters.add(d.name)
        try:
            revision, mtime = _manifest_revision(manifest_path)
            cached = _CHAPTER_LIST_CACHE.get(d.name)
            if cached is not None and cached[0] == revision:
                chapters.append(dict(cached[1]))
                continue

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            pages = manifest.get("pages", [])
            if not pages:
                _CHAPTER_LIST_CACHE.pop(d.name, None)
                continue
            item = {
                "chapter_id": manifest.get("chapter_id", d.name),
                "chapter_name": manifest.get("chapter_name", ""),
                "source_url": manifest.get("source_url", ""),
                "total_pages": len(pages),
                "cover": f"/api/image/{d.name}/0/original",
                "workflow": manifest.get("workflow", {"stage": "preview", "page_index": 0}),
                "updated_at": mtime,
            }
            _CHAPTER_LIST_CACHE[d.name] = (revision, item)
            chapters.append(dict(item))
        except (json.JSONDecodeError, OSError, KeyError):
            _CHAPTER_LIST_CACHE.pop(d.name, None)
            continue

    for chapter_id in set(_CHAPTER_LIST_CACHE) - active_chapters:
        _CHAPTER_LIST_CACHE.pop(chapter_id, None)
    chapters.sort(key=lambda x: x["updated_at"], reverse=True)
    return chapters


@router.post("/workflow_checkpoint")
def set_workflow_checkpoint(req: WorkflowCheckpointRequest) -> dict:
    validate_chapter_id(req.chapter_id)
    with get_manifest_lock(req.chapter_id):
        manifest = load_manifest_raw(req.chapter_id)
        pages = manifest.get("pages", [])
        if req.page_index < 0 or (pages and req.page_index >= len(pages)):
            raise HTTPException(400, f"Invalid page_index: {req.page_index}")
        manifest["workflow"] = {
            "stage": req.stage,
            "page_index": req.page_index,
        }
        save_manifest_raw(req.chapter_id, manifest)
    return {"ok": True, "workflow": manifest["workflow"]}


@router.post("/chapter")
def create_chapter(req: ChapterRequest) -> dict:
    validate_url(req.url)
    chapter_id = _allocate_chapter_id()
    workers = _clamp_workers(req.workers)
    logger.info(f"Creating chapter {chapter_id} from {req.url} (workers={workers})")
    try:
        manifest = pipeline.download_chapter(req.url, chapter_id, workers=workers)
        return urlify_manifest(manifest)
    except HTTPException:
        _cleanup_uncommitted_chapter(chapter_id)
        raise
    except Exception as exc:
        _cleanup_uncommitted_chapter(chapter_id)
        logger.opt(exception=True).error(
            "Chapter {} operation 'create_chapter' failed for URL {}: {}",
            chapter_id,
            req.url,
            exc,
        )
        raise HTTPException(500, f"Download chapter failed: {exc}") from exc


@router.post("/chapter/upload")
async def create_chapter_from_upload(
    files: list[UploadFile] = File(...),
    workers: int = Form(PIPELINE_DEFAULT_WORKERS),
) -> dict:
    if not files:
        raise HTTPException(400, "No files uploaded")
    if len(files) > MAX_UPLOAD_FILES:
        raise HTTPException(400, f"Too many files: max {MAX_UPLOAD_FILES} per upload")

    uploads = []
    total_bytes = 0
    for f in files:
        remaining = MAX_UPLOAD_TOTAL_BYTES - total_bytes
        try:
            data = await read_upload_limited(f, remaining)
        except HTTPException as exc:
            if exc.status_code == 413:
                raise HTTPException(
                    413,
                    f"Total upload size exceeds {MAX_UPLOAD_TOTAL_BYTES // (1024*1024)}MB",
                ) from exc
            raise
        total_bytes += len(data)
        uploads.append((f.filename or "unnamed", data))

    chapter_id = _allocate_chapter_id()
    workers_n = _clamp_workers(workers)
    logger.info(f"Creating chapter {chapter_id} from {len(uploads)} uploaded files (workers={workers_n})")
    try:
        manifest = await run_in_threadpool(pipeline.create_chapter_from_uploads, chapter_id, uploads, workers=workers_n)
        return urlify_manifest(manifest)
    except HTTPException:
        _cleanup_uncommitted_chapter(chapter_id)
        raise
    except ValueError as exc:
        _cleanup_uncommitted_chapter(chapter_id)
        logger.error("Chapter {} operation 'create_chapter_from_upload' invalid payload: {}", chapter_id, exc)
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        _cleanup_uncommitted_chapter(chapter_id)
        logger.opt(exception=True).error("Chapter {} operation 'create_chapter_from_upload' failed: {}", chapter_id, exc)
        raise HTTPException(500, f"Upload chapter failed: {exc}") from exc


@router.post("/process/chapter")
async def start_chapter_processing(req: ProcessPagesRequest) -> dict:
    page_indices, workers = _validated_process_plan(req)
    try:
        return chapter_processing_jobs.start(
            req.chapter_id,
            page_indices=page_indices,
            workers=workers,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/process/chapter/{chapter_id}")
def get_chapter_processing_job(chapter_id: str) -> dict:
    validate_chapter_id(chapter_id)
    snapshot = chapter_processing_jobs.latest_for_chapter(chapter_id)
    if snapshot is not None:
        return snapshot
    return {
        "job_id": None,
        "chapter_id": chapter_id,
        "status": "idle",
        "active": False,
        "reused": False,
        "workers": None,
        "total": 0,
        "completed": 0,
        "remaining": 0,
        "current_batch": [],
        "errors": [],
    }


@router.get("/process/jobs/{job_id}")
def get_processing_job(job_id: str) -> dict:
    value = str(job_id or "").strip().lower()
    if len(value) != 32 or any(ch not in "0123456789abcdef" for ch in value):
        raise HTTPException(400, "Invalid processing job id")
    try:
        return chapter_processing_jobs.snapshot(value)
    except KeyError as exc:
        raise HTTPException(404, "Processing job not found") from exc


@router.post("/process_pages")
def process_pages(req: ProcessPagesRequest) -> dict:
    page_indices, workers = _validated_process_plan(req)
    processing_lock = FileLock(
        PROCESSED_DIR / req.chapter_id / "processing.lock"
    )
    try:
        processing_lock.acquire(timeout=0)
    except Timeout as exc:
        logger.warning(
            "Chapter {}: rejected overlapping page processing request for {}",
            req.chapter_id,
            page_indices,
        )
        raise HTTPException(
            409,
            "This chapter is already being processed. Wait for the active batch to finish.",
        ) from exc
    logger.info(
        f"Chapter {req.chapter_id}: processing pages {page_indices} (workers={workers})"
    )
    try:
        manifest = pipeline.process_pages(req.chapter_id, page_indices, workers=workers)
        return urlify_manifest(manifest)
    except StaleProcessingStateError as exc:
        logger.warning(
            "Chapter {} pages {} operation 'process_pages' discarded stale output: {}",
            req.chapter_id,
            page_indices,
            exc,
        )
        raise HTTPException(409, str(exc)) from exc
    except RuntimeError as exc:
        logger.opt(exception=True).error(
            "Chapter {} pages {} operation 'process_pages' failed: {}",
            req.chapter_id,
            page_indices,
            exc,
        )
        raise HTTPException(500, str(exc)) from exc
    except Exception as exc:
        logger.opt(exception=True).error(
            "Chapter {} pages {} operation 'process_pages' failed unexpectedly: {}",
            req.chapter_id,
            page_indices,
            exc,
        )
        raise HTTPException(500, f"Process pages failed: {exc}") from exc
    finally:
        processing_lock.release()


@router.post("/skip_pages")
def skip_pages(req: SkipPagesRequest) -> dict:
    validate_chapter_id(req.chapter_id)
    manifest_raw = load_manifest_raw(req.chapter_id)
    total_pages = len(manifest_raw.get("pages", []))
    for idx in req.page_indices:
        if not isinstance(idx, int) or idx < 0 or idx >= total_pages:
            raise HTTPException(400, f"Invalid page_index {idx} for chapter {req.chapter_id} (total pages: {total_pages})")
    try:
        manifest = pipeline.mark_skipped(req.chapter_id, req.page_indices, req.skipped)
        return urlify_manifest(manifest)
    except Exception as exc:
        logger.opt(exception=True).error(
            "Chapter {} pages {} operation 'skip_pages' failed: {}",
            req.chapter_id,
            req.page_indices,
            exc,
        )
        raise HTTPException(500, f"Skip pages failed: {exc}") from exc


@router.get("/chapter/{chapter_id}")
def get_chapter(chapter_id: str) -> dict:
    return urlify_manifest(load_manifest_raw(chapter_id))


def _set_page_preserve_regions(chapter_id: str, page_index: int, regions: list[RegionModel]) -> dict:
    validate_chapter_id(chapter_id)
    if page_index < 0:
        raise HTTPException(400, f"Invalid page_index: {page_index}")
    payload = _region_payload(regions)
    with get_page_lock(chapter_id, page_index), get_manifest_lock(chapter_id):
        manifest = load_manifest_raw(chapter_id)
        pages = manifest.get("pages", [])
        if page_index >= len(pages):
            raise HTTPException(400, f"Invalid page_index: {page_index}")
        page = pages[page_index]
        changed = page.get("preserve_regions", []) != payload
        page["preserve_regions"] = payload
        if changed:
            ensure_page_text_objects(page)
            if not page.get("skipped"):
                page["process_required"] = True
                if page.get("clean") is not None:
                    page["clean"] = None
                    bump_page_revision(page, "clean_revision")
            invalidate_page_render(manifest, page_index)
            save_manifest_raw(chapter_id, manifest)
            pipeline._sync_output_dir(chapter_id, manifest, [page_index])
    return urlify_manifest(manifest)


@router.post("/save_preserve_regions")
def save_preserve_regions(req: SavePreserveRegionsRequest) -> dict:
    return _set_page_preserve_regions(req.chapter_id, req.page_index, req.preserve_regions)


@router.post("/chapters/{chapter_id}/pages/{page_index}/preserve-regions")
def set_page_preserve_regions(chapter_id: str, page_index: int, regions: list[RegionModel]) -> dict:
    return _set_page_preserve_regions(chapter_id, page_index, regions)


@router.post("/save_excluded_regions", deprecated=True)
def save_excluded_regions(req: SaveExcludedRegionsRequest) -> dict:
    return _set_page_preserve_regions(req.chapter_id, req.page_index, req.excluded_regions)


@router.post("/chapters/{chapter_id}/pages/{page_index}/excluded-regions", deprecated=True)
def set_page_excluded_regions(chapter_id: str, page_index: int, regions: list[RegionModel]) -> dict:
    return _set_page_preserve_regions(chapter_id, page_index, regions)
