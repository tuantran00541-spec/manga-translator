from __future__ import annotations

import math
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from app.ai_mode.job import AIModeJobManager, AIModeSettings
from app import cloud
from app.ai_providers import normalize_provider_id, validate_model_name
from app.config import OUTPUT_DIR
from app.parameters import PIPELINE_DEFAULT_WORKERS
from app.routers.translation import _resolve_vision_provider, validate_lang_code
from app.secret_store import SecretStoreUnavailable, get_provider_api_key
from app.security import MAX_REMOTE_URL_LENGTH, validate_url

router = APIRouter(prefix="/api/ai_mode", tags=["ai-mode"])
ai_mode_jobs = AIModeJobManager()


class AIModeStartRequest(BaseModel):
    url: str
    provider: str = "deepseek"
    model: str | None = None
    target_lang: str = "vi"
    budget_usd: float = 0.30
    workers: int = Field(default=PIPELINE_DEFAULT_WORKERS, ge=1, le=8)

    @field_validator("url")
    @classmethod
    def _url(cls, value: str) -> str:
        value = str(value or "").strip()
        if not value or len(value) > MAX_REMOTE_URL_LENGTH:
            raise ValueError("A chapter URL is required")
        return value

    @field_validator("provider")
    @classmethod
    def _provider(cls, value: str) -> str:
        return normalize_provider_id(value)

    @field_validator("model")
    @classmethod
    def _model(cls, value: str | None) -> str | None:
        value = (value or "").strip()
        return validate_model_name(value, default="") if value else None

    @field_validator("target_lang")
    @classmethod
    def _lang(cls, value: str) -> str:
        return validate_lang_code(value)

    @field_validator("budget_usd")
    @classmethod
    def _budget(cls, value: float) -> float:
        value = float(value)
        if not math.isfinite(value) or value < 0.01 or value > 2.0:
            raise ValueError("budget_usd must be between 0.01 and 2.0")
        return value


def _snapshot_or_404(job_id: str) -> dict:
    try:
        return ai_mode_jobs.snapshot(job_id)
    except KeyError as exc:
        raise HTTPException(404, "A.I mode job not found") from exc


def _cloud_finish_hook(job_token: str):
    def finish(job) -> None:
        cloud.finish_job(job_token, job.status if job.status in {"completed", "cancelled"} else "failed")
    return finish


@router.post("/start")
async def start_ai_mode(req: AIModeStartRequest) -> dict:
    await run_in_threadpool(validate_url, req.url)
    if ai_mode_jobs.active_job() is not None:
        raise HTTPException(409, "An A.I mode job is already running")
    if req.provider == cloud.CLOUD_PROVIDER_ID:
        reservation = await run_in_threadpool(cloud.reserve_job)
        provider = cloud.cloud_provider(str(reservation.get("model") or ""))
        settings = AIModeSettings(
            url=req.url, provider=provider.id, model=provider.default_qc_model, target_lang=req.target_lang,
            budget_usd=float(reservation.get("cost_cap_usd") or req.budget_usd), workers=req.workers,
        )
        hook = _cloud_finish_hook(str(reservation["job_token"]))
        try:
            return ai_mode_jobs.start(settings, provider=provider, api_key=str(reservation["job_token"]), on_finish=hook)
        except RuntimeError as exc:
            await run_in_threadpool(cloud.finish_job, str(reservation["job_token"]), "cancelled")
            raise HTTPException(409, str(exc)) from exc
    await run_in_threadpool(cloud.require_feature, "byok")
    # Resolve the provider and key before downloading anything, so a missing
    # key fails in a second instead of after the chapter was fetched.
    try:
        provider = _resolve_vision_provider(req.provider)
        model = validate_model_name(req.model, default=str(provider.default_qc_model or ""))
        api_key = get_provider_api_key(provider.id, provider_label=provider.label)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except SecretStoreUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    if not api_key:
        raise HTTPException(409, f"{provider.label} API key is not configured")
    settings = AIModeSettings(
        url=req.url, provider=provider.id, model=model, target_lang=req.target_lang,
        budget_usd=req.budget_usd, workers=req.workers,
    )
    try:
        return ai_mode_jobs.start(settings, provider=provider, api_key=api_key)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get("/active")
async def active_ai_mode_job() -> dict:
    job = ai_mode_jobs.active_job()
    return {"job": ai_mode_jobs.snapshot(job.job_id) if job else None}


@router.get("/jobs/{job_id}")
async def ai_mode_job_status(job_id: str) -> dict:
    return _snapshot_or_404(job_id)


@router.post("/jobs/{job_id}/cancel")
async def cancel_ai_mode_job(job_id: str) -> dict:
    try:
        return ai_mode_jobs.cancel(job_id)
    except KeyError as exc:
        raise HTTPException(404, "A.I mode job not found") from exc


@router.get("/jobs/{job_id}/download")
async def download_ai_mode_zip(job_id: str):
    snapshot = _snapshot_or_404(job_id)
    archive = ai_mode_jobs.archive_path(job_id)
    if not archive:
        raise HTTPException(409, "The ZIP is not ready yet")
    path = Path(archive).resolve()
    chapter_dir = (OUTPUT_DIR / str(snapshot["chapter_id"])).resolve()
    if chapter_dir not in path.parents or not path.is_file():
        raise HTTPException(404, "The ZIP is no longer available")
    return FileResponse(
        path, filename=f"manga-translator-ai-{snapshot['chapter_id']}.zip", media_type="application/zip",
    )
