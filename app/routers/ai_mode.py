from __future__ import annotations

import asyncio
import math

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, field_validator, model_validator

from app import cloud, series_memory
from app.ai_mode.job import AIModeJobManager, AIModeSettings
from app.ai_mode.queue import AIModeQueueManager, MAX_QUEUE_URLS
from app.ai_providers import bind_cloud_job, normalize_provider_id, unbind_cloud_job, validate_model_name
from app.downloader.series_toc import discover_chapter_urls
from app.manifest_utils import load_manifest_raw
from app.parameters import PIPELINE_DEFAULT_WORKERS
from app.routers.translation import _resolve_text_provider, _resolve_vision_provider, validate_lang_code
from app.secret_store import SecretStoreUnavailable, get_provider_api_key
from app.security import MAX_REMOTE_URL_LENGTH, contains_secret_token, validate_chapter_id, validate_url

router = APIRouter(prefix="/api/ai_mode", tags=["ai-mode"])
ai_mode_jobs = AIModeJobManager()
ai_mode_queues = AIModeQueueManager(ai_mode_jobs)
_START_LOCK = asyncio.Lock()


class _AIModeBaseRequest(BaseModel):
    provider: str = "deepseek"
    model: str | None = None
    target_lang: str = "vi"
    budget_usd: float = 0.30
    workers: int = Field(default=PIPELINE_DEFAULT_WORKERS, ge=1, le=8)
    story_notes: str = Field(default="", max_length=1500)
    polish: bool = False
    images_per_request: int = Field(default=2, ge=1, le=8)
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

    @field_validator("story_notes")
    @classmethod
    def _story_notes(cls, value: str) -> str:
        if contains_secret_token(str(value or "")):
            raise ValueError("story_notes must not contain API keys or credentials")
        return value

    @field_validator("budget_usd")
    @classmethod
    def _budget(cls, value: float) -> float:
        value = float(value)
        if not math.isfinite(value) or value < 0.01 or value > 2.0:
            raise ValueError("budget_usd must be between 0.01 and 2.0")
        return value


class AIModeStartRequest(_AIModeBaseRequest):
    url: str | None = None
    chapter_id: str | None = None

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


class AIModeDiscoverRequest(BaseModel):
    url: str = Field(min_length=1, max_length=MAX_REMOTE_URL_LENGTH)

    @field_validator("url")
    @classmethod
    def _url(cls, value: str) -> str:
        cleaned = str(value or "").strip()
        if not cleaned:
            raise ValueError("URL không được để trống.")
        return cleaned


class AIModeQueueStartRequest(_AIModeBaseRequest):
    urls: list[str] = Field(min_length=1, max_length=MAX_QUEUE_URLS)
    continue_on_error: bool = True

    @field_validator("urls")
    @classmethod
    def _clean_urls(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()
        for raw in values:
            url = str(raw or "").strip()
            if not url:
                continue
            if len(url) > MAX_REMOTE_URL_LENGTH:
                raise ValueError("Chapter URL is too long")
            key = url.rstrip("/")
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(url)
        if not cleaned:
            raise ValueError("Cần ít nhất 1 liên kết chương hợp lệ.")
        if len(cleaned) > MAX_QUEUE_URLS:
            raise ValueError(f"Mỗi hàng đợi tối đa {MAX_QUEUE_URLS} chương.")
        return cleaned


def _snapshot_or_404(job_id: str) -> dict:
    try:
        return ai_mode_jobs.snapshot(job_id)
    except KeyError as exc:
        raise HTTPException(404, "Không tìm thấy tác vụ Chế độ AI.") from exc


def _queue_snapshot_or_404(queue_id: str) -> dict:
    try:
        return ai_mode_queues.snapshot(queue_id)
    except KeyError as exc:
        raise HTTPException(404, "Không tìm thấy hàng đợi Chế độ AI.") from exc


def _cloud_finish_hook(job_token: str):
    def finish(job) -> None:
        cloud.finish_job(
            job_token,
            job.status if job.status in {"completed", "completed_with_blockers", "cancelled"} else "failed",
        )

    return finish


def _ensure_idle() -> None:
    if ai_mode_jobs.active_job() is not None or ai_mode_queues.active_queue() is not None:
        raise HTTPException(409, "Một tác vụ hoặc hàng đợi Chế độ AI khác đang chạy.")


def _validate_cloud_translate(req: _AIModeBaseRequest) -> tuple[str | None, str | None]:
    if not (req.translate_provider and req.translate_model):
        return (None, None)
    if req.translate_provider != cloud.CLOUD_PROVIDER_ID:
        raise HTTPException(400, "Chế độ AI trên Manga Cloud sẽ dịch lại thông qua Manga Cloud.")
    return (cloud.CLOUD_PROVIDER_ID, validate_model_name(req.translate_model, default=""))


def _resolve_direct_execution(req: _AIModeBaseRequest) -> tuple[object, str, AIModeSettings]:
    try:
        provider = _resolve_vision_provider(req.provider)
        model = validate_model_name(req.model, default=str(provider.default_qc_model or ""))
        api_key = get_provider_api_key(provider.id, provider_label=provider.label)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except SecretStoreUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    if not api_key:
        raise HTTPException(409, f"Chưa cấu hình khóa API cho {provider.label}.")

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
            raise HTTPException(409, f"Chưa cấu hình khóa API cho {text_provider.label}.")
        translate_provider_id = text_provider.id

    settings = AIModeSettings(
        url="",
        provider=provider.id,
        model=model,
        target_lang=req.target_lang,
        budget_usd=req.budget_usd,
        workers=req.workers,
        story_notes=req.story_notes,
        polish=req.polish,
        images_per_request=req.images_per_request,
        translate_provider=translate_provider_id,
        translate_model=translate_model_name,
    )
    return provider, api_key, settings


async def _start_cloud_job(
    url: str,
    req: _AIModeBaseRequest,
    translate_provider_id: str | None,
    translate_model_name: str | None,
) -> dict:
    reservation = await run_in_threadpool(cloud.reserve_job)
    provider = cloud.cloud_provider(str(reservation.get("model") or ""))
    settings = AIModeSettings(
        url=url,
        chapter_id=None,
        provider=provider.id,
        model=provider.default_qc_model,
        target_lang=req.target_lang,
        budget_usd=float(reservation.get("cost_cap_usd") or req.budget_usd),
        workers=req.workers,
        story_notes=req.story_notes,
        polish=req.polish,
        images_per_request=2,
        translate_provider=translate_provider_id,
        translate_model=translate_model_name,
    )
    job_token = str(reservation["job_token"])
    bound = bind_cloud_job(provider, job_token)
    try:
        return ai_mode_jobs.start(settings, provider=provider, api_key=job_token, on_finish=_cloud_finish_hook(job_token))
    except Exception:
        await run_in_threadpool(cloud.finish_job, job_token, "cancelled")
        raise
    finally:
        unbind_cloud_job(bound)


@router.post("/start")
async def start_ai_mode(req: AIModeStartRequest) -> dict:
    if req.chapter_id:
        await run_in_threadpool(validate_chapter_id, req.chapter_id)
        await run_in_threadpool(load_manifest_raw, req.chapter_id)
    else:
        await run_in_threadpool(validate_url, req.url)
    async with _START_LOCK:
        _ensure_idle()
        if req.provider == cloud.CLOUD_PROVIDER_ID:
            if req.chapter_id:
                raise HTTPException(400, "chapter_id chỉ dùng cục bộ; Manga Cloud bắt đầu từ URL.")
            translate_provider_id, translate_model_name = _validate_cloud_translate(req)
            try:
                return await _start_cloud_job(req.url or "", req, translate_provider_id, translate_model_name)
            except RuntimeError as exc:
                raise HTTPException(409, str(exc)) from exc

        provider, api_key, base_settings = await run_in_threadpool(_resolve_direct_execution, req)
        settings = AIModeSettings(
            url=req.url or "",
            chapter_id=req.chapter_id,
            provider=base_settings.provider,
            model=base_settings.model,
            target_lang=base_settings.target_lang,
            budget_usd=base_settings.budget_usd,
            workers=base_settings.workers,
            story_notes=base_settings.story_notes,
            polish=base_settings.polish,
            images_per_request=base_settings.images_per_request,
            translate_provider=base_settings.translate_provider,
            translate_model=base_settings.translate_model,
        )
        try:
            return ai_mode_jobs.start(settings, provider=provider, api_key=api_key)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc


@router.post("/queue/discover")
async def discover_queue_chapters(req: AIModeDiscoverRequest) -> dict:
    await run_in_threadpool(validate_url, req.url)
    chapters = await run_in_threadpool(discover_chapter_urls, req.url)
    return {"url": req.url, "chapters": chapters}


@router.post("/queue/start")
async def start_ai_mode_queue(req: AIModeQueueStartRequest) -> dict:
    for url in req.urls:
        await run_in_threadpool(validate_url, url)
    async with _START_LOCK:
        _ensure_idle()

        if req.provider == cloud.CLOUD_PROVIDER_ID:
            translate_provider_id, translate_model_name = _validate_cloud_translate(req)
            base_settings = AIModeSettings(
                url=req.urls[0],
                provider=cloud.CLOUD_PROVIDER_ID,
                model="",
                target_lang=req.target_lang,
                budget_usd=req.budget_usd,
                workers=req.workers,
                story_notes=req.story_notes,
                polish=req.polish,
                images_per_request=2,
                translate_provider=translate_provider_id,
                translate_model=translate_model_name,
            )

            async def starter(item_settings: AIModeSettings) -> dict:
                return await _start_cloud_job(item_settings.url, req, translate_provider_id, translate_model_name)
        else:
            provider, api_key, base_settings = await run_in_threadpool(_resolve_direct_execution, req)

            async def starter(item_settings: AIModeSettings) -> dict:
                return ai_mode_jobs.start(item_settings, provider=provider, api_key=api_key)

        try:
            return ai_mode_queues.start(
                req.urls,
                base_settings,
                item_starter=starter,
                continue_on_error=req.continue_on_error,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc


@router.get("/active")
async def active_ai_mode_job() -> dict:
    job = ai_mode_jobs.active_job()
    return {
        "job": ai_mode_jobs.snapshot(job.job_id) if job else None,
        "queue": ai_mode_queues.active_snapshot(),
    }


@router.get("/queue/{queue_id}")
async def ai_mode_queue_status(queue_id: str) -> dict:
    return _queue_snapshot_or_404(queue_id)


@router.post("/queue/{queue_id}/cancel")
async def cancel_ai_mode_queue(queue_id: str) -> dict:
    try:
        return ai_mode_queues.cancel(queue_id)
    except KeyError as exc:
        raise HTTPException(404, "Không tìm thấy hàng đợi Chế độ AI.") from exc


@router.get("/jobs/{job_id}")
async def ai_mode_job_status(job_id: str) -> dict:
    return _snapshot_or_404(job_id)


@router.post("/jobs/{job_id}/cancel")
async def cancel_ai_mode_job(job_id: str) -> dict:
    try:
        return ai_mode_jobs.cancel(job_id)
    except KeyError as exc:
        raise HTTPException(404, "Không tìm thấy tác vụ Chế độ AI.") from exc


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
    key = _series_key(req.url)
    data = series_memory.load(key)
    return {"key": key, "text": series_memory.to_text(data), "notes": data["notes"], "chapters": len(data["chapters"])}


@router.post("/series/save")
def save_series_memory(req: SeriesMemoryRequest) -> dict:
    key = _series_key(req.url)

    def edit(current: dict) -> dict:
        data = series_memory.from_text(req.text if req.text is not None else series_memory.to_text(current), current)
        if req.notes is not None:
            data["notes"] = req.notes
        return data

    data = series_memory.update(key, edit)
    return {"key": key, "text": series_memory.to_text(data), "notes": data["notes"], "chapters": len(data["chapters"])}
