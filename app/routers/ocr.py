from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from app.dependencies import ocr, pipeline
from app.logging_config import logger
from app.manifest_utils import urlify_manifest
from app.ocr.jobs import ChapterOCRJobManager
from app.ocr.optimized_service import CachedOCRService
from app.ocr.schemas import ChapterOCRRequest
from app.ocr.service import OCRCancelled, OCRResultStale
from app.schemas import OcrBoxRequest, OcrTextObjectRequest
from app.security import validate_chapter_id

router = APIRouter(prefix="/api", tags=["ocr"])
ocr_service = CachedOCRService(ocr, pipeline)
chapter_ocr_jobs = ChapterOCRJobManager(ocr_service)
OCR_JOB_NOT_FOUND = "OCR job not found"


@router.post(
    "/ocr_box",
    responses={400: {"description": "Invalid OCR target"}, 404: {"description": "Source image missing"}, 409: {"description": "OCR result became stale"}, 500: {"description": "OCR failed"}},
)
def ocr_box(req: OcrBoxRequest) -> dict:
    validate_chapter_id(req.chapter_id)
    try:
        result = ocr_service.inspect_box_index(
            req.chapter_id,
            req.page_index,
            req.box_index,
            req.lang,
        )
        return result
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except (OCRResultStale, OCRCancelled) as exc:
        raise HTTPException(409, str(exc)) from exc
    except Exception as exc:
        logger.opt(exception=True).error(
            "Chapter {} page {} box {} operation 'ocr_box' failed: {}",
            req.chapter_id,
            req.page_index,
            req.box_index,
            exc,
        )
        raise HTTPException(500, "OCR failed") from exc


@router.post(
    "/text_object/ocr",
    responses={400: {"description": "Invalid text object"}, 404: {"description": "Source image missing"}, 500: {"description": "OCR failed"}},
)
async def text_object_ocr(req: OcrTextObjectRequest) -> dict:
    validate_chapter_id(req.chapter_id)
    try:
        manifest = await run_in_threadpool(
            ocr_service.group_text_object,
            req.chapter_id,
            req.page_index,
            req.id,
            req.lang,
        )
        return urlify_manifest(manifest)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:
        logger.opt(exception=True).error(
            "Chapter {} page {} object {} operation 'text_object_ocr' failed: {}",
            req.chapter_id,
            req.page_index,
            req.id,
            exc,
        )
        raise HTTPException(500, "Text object OCR failed") from exc


@router.post(
    "/ocr/chapter",
    responses={400: {"description": "Invalid OCR request"}, 409: {"description": "OCR already running"}, 429: {"description": "Too many OCR jobs"}},
)
async def start_chapter_ocr(req: ChapterOCRRequest) -> dict:
    validate_chapter_id(req.chapter_id)
    try:
        return chapter_ocr_jobs.start(
            req.chapter_id,
            lang=req.lang,
            concurrency=req.concurrency,
            force=req.force,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        detail = str(exc)
        status = 429 if detail == "Too many active OCR jobs" else 409
        raise HTTPException(status, detail) from exc


def _snapshot_or_404(job_id: str) -> dict:
    try:
        return chapter_ocr_jobs.snapshot(job_id)
    except KeyError as exc:
        raise HTTPException(404, OCR_JOB_NOT_FOUND) from exc


@router.get("/ocr/chapter/{job_id}", responses={404: {"description": OCR_JOB_NOT_FOUND}})
def chapter_ocr_status(job_id: str) -> dict:
    return _snapshot_or_404(job_id)


@router.post("/ocr/chapter/{job_id}/cancel", responses={404: {"description": OCR_JOB_NOT_FOUND}})
def cancel_chapter_ocr(job_id: str) -> dict:
    try:
        return chapter_ocr_jobs.cancel(job_id)
    except KeyError as exc:
        raise HTTPException(404, OCR_JOB_NOT_FOUND) from exc


@router.post(
    "/ocr/chapter/{job_id}/retry",
    responses={400: {"description": "Nothing to retry"}, 404: {"description": OCR_JOB_NOT_FOUND}, 409: {"description": "OCR job still running"}},
)
async def retry_chapter_ocr(job_id: str) -> dict:
    try:
        return chapter_ocr_jobs.retry(job_id)
    except KeyError as exc:
        raise HTTPException(404, OCR_JOB_NOT_FOUND) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
