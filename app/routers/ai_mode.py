from __future__ import annotations

import math

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, field_validator, model_validator

from app.ai_mode.job import AIModeJobManager, AIModeSettings
from app import cloud, series_memory
from app.ai_providers import bind_cloud_job, normalize_provider_id, unbind_cloud_job, validate_model_name
from app.manifest_utils import load_manifest_raw
from app.parameters import PIPELINE_DEFAULT_WORKERS
from app.routers.translation import _resolve_text_provider, _resolve_vision_provider, validate_lang_code
from app.secret_store import SecretStoreUnavailable, get_provider_api_key
from app.security import MAX_REMOTE_URL_LENGTH, validate_chapter_id, validate_url

router = APIRouter(prefix="/api/ai_mode", tags=["ai-mode"])
ai_mode_jobs = AIModeJobManager()


class AIModeStartRequest(BaseModel):
    url: str | None = None
    chapter_id: str | None = None
    provider: str = "deepseek"
    model: str | None = None
    target_lang: str = "vi"
    budget_usd: float = 0.30
    workers: int = Field(default=PIPELINE_DEFAULT_WORKERS, ge=1, le=8)
    story_notes: str = Field(default="", max_length=1500)
    polish: bool = False
    images_per_request: int = Field(default=2, ge=1, le=8)
    # A second model re-translates the source lines the vision model read; unset means the vision model's wording stands.
    translate_provider: str | None = None
    translate_model: str | None = None

    @field_validator("translate_provider")
    @classmethod
    def _translate_provider(cls, value: str | None) -> str | None:
        return normalize_provider_id(value) if value else None

    @field_validator("translate_model")
    @classmethod
    def _translate_model(cls, value: str | None) -> str | None:
        value = (value or "").strip()
        return validate_model_name(value, default="") if value else None

    @field_validator("url")
    @classmethod
    def _url(cls, value: str | None) -> str | None:
        value = str(value or "").strip()
        if len(value) > MAX_REMOTE_URL_LENGTH:
            raise ValueError("Chapter URL is too long")
        return value or None

    @model_validator(mode="after")
    def _one_source(self) -> "AIModeStartRequest":
        if bool(self.url) == bool(self.chapter_id):
            raise ValueError("Provide exactly one of url and chapter_id")
        return self

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
    if req.chapter_id:
        await run_in_threadpool(validate_chapter_id, req.chapter_id)
        await run_in_threadpool(load_manifest_raw, req.chapter_id)
    else:
        await run_in_threadpool(validate_url, req.url)
    if ai_mode_jobs.active_job() is not None:
        raise HTTPException(409, "An A.I mode job is already running")
    if req.provider == cloud.CLOUD_PROVIDER_ID:
        if req.chapter_id:
            raise HTTPException(400, "chapter_id is local-only; Manga Cloud starts from a URL")
        reservation = await run_in_threadpool(cloud.reserve_job)
        provider = cloud.cloud_provider(str(reservation.get("model") or ""))
        # The text re-translate runs through Manga Cloud too: the gateway picks its
        # model from the retranslate stage, and the job token is its key, resolved
        # inside the job where the cloud binding lives. A direct provider has no
        # key on the cost-run runner, so it is refused here instead of failing per slice.
        translate_provider_id = translate_model_name = None
        if req.translate_provider and req.translate_model:
            if req.translate_provider != cloud.CLOUD_PROVIDER_ID:
                raise HTTPException(400, "A.I mode on Manga Cloud re-translates through Manga Cloud")
            translate_provider_id = cloud.CLOUD_PROVIDER_ID
            translate_model_name = validate_model_name(req.translate_model, default="")
        settings = AIModeSettings(
            url=req.url or "", chapter_id=req.chapter_id, provider=provider.id, model=provider.default_qc_model, target_lang=req.target_lang,
            budget_usd=float(reservation.get("cost_cap_usd") or req.budget_usd), workers=req.workers,
            # Manga Cloud reads on a free model that takes two images a request; it is not the user's to raise.
            story_notes=req.story_notes, polish=req.polish, images_per_request=2,
            translate_provider=translate_provider_id, translate_model=translate_model_name,
        )
        job_token = str(reservation["job_token"])
        bound = bind_cloud_job(provider, job_token)
        try:
            return ai_mode_jobs.start(settings, provider=provider, api_key=job_token, on_finish=_cloud_finish_hook(job_token))
        except RuntimeError as exc:
            await run_in_threadpool(cloud.finish_job, job_token, "cancelled")
            raise HTTPException(409, str(exc)) from exc
        finally:
            unbind_cloud_job(bound)
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
    # Resolve the text model too, so a missing key fails in a second instead of after the chapter was fetched.
    translate_provider_id = translate_model_name = None
    if req.translate_provider and req.translate_model:
        try:
            text_provider = _resolve_text_provider(req.translate_provider)
            translate_model_name = validate_model_name(req.translate_model, default="")
            text_api_key = get_provider_api_key(text_provider.id, provider_label=text_provider.label)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except SecretStoreUnavailable as exc:
            raise HTTPException(503, str(exc)) from exc
        if not text_api_key:
            raise HTTPException(409, f"{text_provider.label} API key is not configured")
        translate_provider_id = text_provider.id
    settings = AIModeSettings(
        url=req.url or "", chapter_id=req.chapter_id, provider=provider.id, model=model, target_lang=req.target_lang,
        budget_usd=req.budget_usd, workers=req.workers, story_notes=req.story_notes, polish=req.polish,
        images_per_request=req.images_per_request,
        translate_provider=translate_provider_id, translate_model=translate_model_name,
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


class SeriesMemoryRequest(BaseModel):
    url: str = Field(max_length=MAX_REMOTE_URL_LENGTH)
    text: str | None = Field(default=None, max_length=60000)
    notes: str | None = Field(default=None, max_length=1500)


def _series_key(url: str) -> str:
    key = series_memory.series_key(url)
    if not key:
        raise HTTPException(400, "This URL does not name a series")
    return key


@router.get("/series")
def list_series_memory() -> list[dict]:
    return series_memory.list_series()


@router.post("/series/read")
def read_series_memory(req: SeriesMemoryRequest) -> dict:
    """What the series behind a chapter URL has settled, as the lines the user edits."""
    key = _series_key(req.url)
    data = series_memory.load(key)
    return {"key": key, "text": series_memory.to_text(data), "notes": data["notes"], "chapters": len(data["chapters"])}


@router.post("/series/save")
def save_series_memory(req: SeriesMemoryRequest) -> dict:
    key = _series_key(req.url)
    current = series_memory.load(key)
    data = series_memory.from_text(req.text if req.text is not None else series_memory.to_text(current), current)
    if req.notes is not None:
        data["notes"] = req.notes
    data = series_memory.save(key, data)
    return {"key": key, "text": series_memory.to_text(data), "notes": data["notes"], "chapters": len(data["chapters"])}
