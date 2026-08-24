from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from app.logging_config import logger
from app.manifest_utils import load_manifest_raw
from app.schemas import VisualQCInspectRequest, VisualQCKeyRequest
from app.secret_store import (
    SecretStoreUnavailable,
    delete_gemini_api_key,
    gemini_key_status,
    get_gemini_api_key,
    set_gemini_api_key,
)
from app.security import validate_chapter_id
from app.visual_qc.batch_runner import RegionBatchRunner
from app.visual_qc.gemini import DEFAULT_GEMINI_MODEL, GeminiVisualQC, GeminiVisualQCTimeout
from app.visual_qc.jobs import VisualQCJobManager
from app.visual_qc.region_client import GeminiRegionQC
from app.visual_qc.schemas import VisualQCChapterRequest
from app.visual_qc.service import ChapterQCService

router = APIRouter(prefix="/api/visual_qc", tags=["visual-qc"])
visual_qc = GeminiVisualQC()
region_qc = GeminiRegionQC()
chapter_qc_jobs = VisualQCJobManager()
chapter_qc_service = ChapterQCService(
    RegionBatchRunner(region_qc),
    chapter_qc_jobs,
    model=region_qc.model,
)


def _file_revision(path: Path) -> tuple[int, int, int]:
    """Cheap identity token used to reject stale QC results after a page changes."""
    st = path.stat()
    return (st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def _page_paths(manifest: dict, page_index: int) -> tuple[Path, Path]:
    page = manifest["pages"][page_index]
    return Path(page.get("original") or ""), Path(page.get("clean") or "")


@router.get("/settings")
def visual_qc_settings() -> dict:
    status = gemini_key_status()
    return {**status, "model": DEFAULT_GEMINI_MODEL}


@router.post("/key")
def save_visual_qc_key(req: VisualQCKeyRequest) -> dict:
    try:
        set_gemini_api_key(req.api_key)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except SecretStoreUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    return {"configured": True, "source": "os_secure_storage", "model": DEFAULT_GEMINI_MODEL}


@router.delete("/key")
def clear_visual_qc_key() -> dict:
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        return {
            "configured": True,
            "source": "environment",
            "model": DEFAULT_GEMINI_MODEL,
            "detail": "Environment-provided keys must be removed from the process environment.",
        }
    try:
        delete_gemini_api_key()
    except SecretStoreUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    return {"configured": False, "source": "none", "model": DEFAULT_GEMINI_MODEL}


@router.post("/inspect")
async def inspect_visual_qc(req: VisualQCInspectRequest) -> dict:
    validate_chapter_id(req.chapter_id)
    manifest = load_manifest_raw(req.chapter_id)
    pages = manifest.get("pages", [])
    if req.page_index < 0 or req.page_index >= len(pages):
        raise HTTPException(400, f"Invalid page_index: {req.page_index}")

    original_path, cleaned_path = _page_paths(manifest, req.page_index)
    if not original_path.is_file():
        raise HTTPException(404, "Original page image not found")
    if not cleaned_path.is_file():
        raise HTTPException(409, "Page has not been cleaned yet")

    try:
        original_revision = _file_revision(original_path)
        cleaned_revision = _file_revision(cleaned_path)
    except OSError as exc:
        raise HTTPException(409, "Page image changed before visual QC could start") from exc

    try:
        api_key = get_gemini_api_key()
    except SecretStoreUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    if not api_key:
        raise HTTPException(409, "Gemini API key is not configured")

    try:
        issues = await run_in_threadpool(visual_qc.inspect, original_path, cleaned_path, api_key)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except GeminiVisualQCTimeout as exc:
        logger.warning(
            "Chapter {} page {} Gemini visual QC timed out: {}",
            req.chapter_id,
            req.page_index,
            exc,
        )
        raise HTTPException(504, str(exc)) from exc
    except Exception as exc:
        # Do not attach a diagnostic traceback here: Gemini inspect receives the
        # API key as an argument, and diagnostic tracebacks can expose locals.
        logger.error(
            "Chapter {} page {} Gemini visual QC failed: {}",
            req.chapter_id,
            req.page_index,
            exc,
        )
        raise HTTPException(502, str(exc)) from exc

    # A Gemini request can take long enough for the user to repaint/reset the page
    # in another tab. Never apply coordinates computed against an obsolete image.
    try:
        latest_manifest = load_manifest_raw(req.chapter_id)
        latest_pages = latest_manifest.get("pages", [])
        if req.page_index >= len(latest_pages):
            raise HTTPException(409, "Page changed while Gemini was inspecting it; run AI QC again")
        latest_original, latest_cleaned = _page_paths(latest_manifest, req.page_index)
        if (
            latest_original != original_path
            or latest_cleaned != cleaned_path
            or not latest_original.is_file()
            or not latest_cleaned.is_file()
            or _file_revision(latest_original) != original_revision
            or _file_revision(latest_cleaned) != cleaned_revision
        ):
            raise HTTPException(409, "Page changed while Gemini was inspecting it; run AI QC again")
    except HTTPException:
        raise
    except OSError as exc:
        raise HTTPException(409, "Page changed while Gemini was inspecting it; run AI QC again") from exc

    return {
        "model": visual_qc.model,
        "issues": [
            {
                "issue_type": issue.issue_type,
                "confidence": issue.confidence,
                "label": issue.label,
                "box_2d": list(issue.box_2d),
                "polygon": [[x, y] for x, y in issue.polygon],
            }
            for issue in issues
        ],
    }


@router.post("/chapter")
async def start_chapter_visual_qc(req: VisualQCChapterRequest) -> dict:
    validate_chapter_id(req.chapter_id)
    try:
        job = await chapter_qc_service.start(req.chapter_id, concurrency=req.concurrency)
    except SecretStoreUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    except ValueError as exc:
        status = 409 if "API key" in str(exc) else 400
        raise HTTPException(status, str(exc)) from exc
    except Exception as exc:
        logger.error("Chapter {} visual QC job failed to start: {}", req.chapter_id, exc)
        raise HTTPException(500, "Could not start chapter visual QC") from exc
    return chapter_qc_jobs.snapshot(job.job_id)


def _job_snapshot_or_404(job_id: str) -> dict:
    try:
        return chapter_qc_jobs.snapshot(job_id)
    except KeyError as exc:
        raise HTTPException(404, "Visual QC job not found") from exc


@router.get("/chapter/{job_id}")
def chapter_visual_qc_status(job_id: str) -> dict:
    return _job_snapshot_or_404(job_id)


@router.post("/chapter/{job_id}/cancel")
def cancel_chapter_visual_qc(job_id: str) -> dict:
    snapshot = _job_snapshot_or_404(job_id)
    if snapshot["status"] not in {"completed", "cancelled"}:
        chapter_qc_jobs.cancel(job_id)
    return chapter_qc_jobs.snapshot(job_id)


@router.post("/chapter/{job_id}/retry")
async def retry_failed_chapter_visual_qc(job_id: str) -> dict:
    previous = _job_snapshot_or_404(job_id)
    if previous["status"] not in {"completed", "cancelled"}:
        raise HTTPException(409, "Visual QC job is still running")
    if int(previous.get("failed") or 0) <= 0:
        raise HTTPException(409, "Visual QC job has no failed regions to retry")
    try:
        job = await chapter_qc_service.start(
            previous["chapter_id"],
            concurrency=int(previous.get("concurrency") or 2),
        )
    except SecretStoreUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    except ValueError as exc:
        status = 409 if "API key" in str(exc) else 400
        raise HTTPException(status, str(exc)) from exc
    return chapter_qc_jobs.snapshot(job.job_id)
