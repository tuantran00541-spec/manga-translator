from __future__ import annotations

import os
from pathlib import Path

import requests

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, field_validator

from app import cloud
from app.ai_providers import (
    PROVIDERS,
    cloud_job_provider,
    normalize_provider_id,
    resolve_provider,
    validate_model_name,
    validate_provider_label,
)
from app.config import PROCESSED_DIR, RAW_DIR
from app.downloader.http import safe_get
from app.logging_config import logger
from app.manifest_utils import load_manifest_raw
from app.schemas import VisualQCInspectRequest, VisualQCKeyRequest
from app.secret_store import (
    SecretStoreUnavailable,
    delete_deepseek_api_key,
    delete_gemini_api_key,
    set_deepseek_api_key,
    set_gemini_api_key,
    delete_provider_api_key,
    delete_provider_config,
    get_provider_api_key,
    get_provider_config,
    list_provider_configs,
    provider_key_status,
    set_provider_api_key,
    set_provider_config,
)
from app.security import validate_chapter_id, validate_managed_path, validate_url
from app.visual_qc.batch_runner import RegionBatchRunner
from app.visual_qc.deepseek_region_client import (
    DEFAULT_DEEPSEEK_MODEL,
    OpenAICompatibleRegionQC,
)
from app.visual_qc.gemini import DEFAULT_GEMINI_MODEL, GeminiVisualQC, GeminiVisualQCTimeout
from app.visual_qc.jobs import VisualQCJobManager
from app.visual_qc.region_client import GeminiRegionQC
from app.visual_qc.openai_compatible import OpenAICompatibleVisualQC
from app.visual_qc.schemas import VisualQCChapterRequest
from app.visual_qc.service import ChapterQCService

router = APIRouter(prefix="/api/visual_qc", tags=["visual-qc"])
chapter_qc_jobs = VisualQCJobManager()
_chapter_qc_context: dict[str, dict] = {}


class DeepSeekKeyRequest(BaseModel):
    api_key: str

    @field_validator("api_key")
    @classmethod
    def _api_key_not_empty(cls, value: str) -> str:
        value = (value or "").strip()
        if not value:
            raise ValueError("DeepSeek API key is required")
        if len(value) > 4096:
            raise ValueError("DeepSeek API key is unexpectedly long")
        return value


def _file_revision(path: Path) -> tuple[int, int, int]:
    st = path.stat()
    return (st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def _page_paths(manifest: dict, page_index: int, chapter_id: str) -> tuple[Path, Path]:
    page = manifest["pages"][page_index]
    original_value = page.get("original")
    cleaned_value = page.get("clean")
    if not original_value:
        raise HTTPException(404, "Original page image not found")
    if not cleaned_value:
        raise HTTPException(409, "Page has not been cleaned yet")
    original = validate_managed_path(original_value, RAW_DIR / chapter_id)
    cleaned = validate_managed_path(cleaned_value, PROCESSED_DIR / chapter_id)
    return original, cleaned


def _redact_secret(text: object, secret: str | None) -> str:
    value = str(text)
    if secret:
        value = value.replace(secret, "[redacted]")
    return value


def _raise_job_capacity_error(exc: RuntimeError) -> None:
    detail = str(exc)
    if detail == "Visual QC is already running for this chapter":
        raise HTTPException(409, detail) from exc
    if detail in {"Too many active visual QC jobs", "Too many visual QC jobs are retained"}:
        raise HTTPException(429, detail) from exc
    raise HTTPException(500, "Could not start chapter visual QC") from exc


def _resolve_configured_provider(provider_id: str):
    normalized = normalize_provider_id(provider_id)
    cloud = cloud_job_provider(normalized)
    if cloud is not None:
        return cloud
    if normalized in PROVIDERS:
        return PROVIDERS[normalized]
    stored = get_provider_config(normalized)
    if not isinstance(stored, dict):
        raise ValueError(
            f"Custom AI provider is not configured: {normalized}"
        )
    provider = resolve_provider(
        normalized,
        label=stored.get("label"),
        protocol=stored.get("protocol"),
        api_base=stored.get("api_base"),
    )
    _validate_custom_remote(provider)
    return provider


def _validate_custom_remote(provider) -> None:
    if provider.builtin:
        return
    if provider.chat_url is None:
        raise ValueError(
            "Custom provider does not expose an OpenAI-compatible chat endpoint"
        )
    validate_url(provider.chat_url)
    validate_url(provider.models_url)


def _make_openai_service(client: OpenAICompatibleRegionQC) -> ChapterQCService:
    return ChapterQCService(
        RegionBatchRunner(client),
        chapter_qc_jobs,
        api_key_provider=lambda: get_provider_api_key(
            client.provider_id,
            provider_label=client.provider_label,
        ),
        model=client.model,
        provider=client.provider_id,
    )


def _make_chapter_service(provider, model: str | None, budget_usd: float):
    if not provider.supports_visual_qc:
        raise ValueError(f"{provider.label} is not available for visual QC")
    selected_model = validate_model_name(
        model,
        default=provider.default_qc_model,
    )
    if provider.protocol == "gemini":
        client = GeminiRegionQC(model=selected_model)
        service = ChapterQCService(
            RegionBatchRunner(client),
            chapter_qc_jobs,
            api_key_provider=lambda: get_provider_api_key(provider.id),
            model=selected_model,
            provider=provider.id,
        )
        return service, client
    client = OpenAICompatibleRegionQC(
        model=selected_model,
        budget_usd=budget_usd,
        provider_id=provider.id,
        provider_label=provider.label,
        chat_url=str(provider.chat_url),
        provider=provider,
    )
    return _make_openai_service(client), client

def _remember_job(job_id: str, *, provider, client=None) -> None:
    _chapter_qc_context[job_id] = {
        "provider": provider.id,
        "provider_label": provider.label,
        "provider_protocol": provider.protocol,
        "provider_api_base": provider.api_base,
        "model": client.model if client is not None else DEFAULT_GEMINI_MODEL,
        "client": client,
    }
    if len(_chapter_qc_context) > 64:
        for stale_id in list(_chapter_qc_context)[:-64]:
            _chapter_qc_context.pop(stale_id, None)


def _enrich_snapshot(snapshot: dict) -> dict:
    result = dict(snapshot)
    context = _chapter_qc_context.get(str(snapshot.get("job_id"))) or {}
    provider_id = str(context.get("provider") or "gemini")
    provider_label = str(context.get("provider_label") or provider_id)
    result["provider"] = provider_id
    result["provider_label"] = provider_label
    result["model"] = str(context.get("model") or DEFAULT_GEMINI_MODEL)
    client = context.get("client")
    if isinstance(client, OpenAICompatibleRegionQC):
        result["usage"] = client.usage_snapshot()
    return result


@router.get("/settings")
def visual_qc_settings() -> dict:
    providers = {}
    for provider in PROVIDERS.values():
        status = provider_key_status(provider.id)
        providers[provider.id] = {
            **status,
            "id": provider.id,
            "label": provider.label,
            "protocol": provider.protocol,
            "api_base": provider.api_base,
            "model": provider.default_qc_model,
            "translation_model": provider.default_translation_model,
            "builtin": True,
            "capabilities": provider.public_capabilities(),
        }

    registry_detail = None
    try:
        custom_configs = list_provider_configs()
    except SecretStoreUnavailable as exc:
        custom_configs = []
        registry_detail = str(exc)
    for config in custom_configs:
        try:
            provider = resolve_provider(
                str(config.get("id") or ""),
                label=config.get("label"),
                protocol=config.get("protocol"),
                api_base=config.get("api_base"),
            )
        except ValueError:
            continue
        providers[provider.id] = {
            **provider_key_status(
                provider.id,
                provider_label=provider.label,
            ),
            "id": provider.id,
            "label": provider.label,
            "protocol": provider.protocol,
            "api_base": provider.api_base,
            "model": "",
            "translation_model": "",
            "builtin": False,
            "capabilities": provider.public_capabilities(),
        }

    gemini = providers["gemini"]
    result = {
        **gemini,
        "model": DEFAULT_GEMINI_MODEL,
        "providers": providers,
        "custom_protocols": [
            {
                "id": "openai",
                "label": "OpenAI-compatible",
                "description": "HTTPS API root exposing /models and /chat/completions",
            }
        ],
    }
    if registry_detail:
        result["registry_detail"] = registry_detail
    return result


@router.post("/providers/{provider_id}/key")
def save_provider_key(provider_id: str, req: VisualQCKeyRequest) -> dict:
    try:
        normalized = normalize_provider_id(provider_id)
        builtin = PROVIDERS.get(normalized)
        label = validate_provider_label(
            req.provider_label,
            default=(builtin.label if builtin else normalized),
        )
        if builtin is None:
            provider = resolve_provider(
                normalized,
                label=label,
                protocol=req.provider_protocol,
                api_base=req.provider_api_base,
            )
            _validate_custom_remote(provider)
            set_provider_config(
                normalized,
                label=provider.label,
                protocol=provider.protocol,
                api_base=provider.api_base,
            )
        set_provider_api_key(
            normalized,
            req.api_key,
            provider_label=label,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except SecretStoreUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    return {
        "configured": True,
        "source": "os_secure_storage",
        "provider": normalized,
    }


@router.delete("/providers/{provider_id}/key")
def clear_provider_key(
    provider_id: str,
    provider_label: str | None = None,
    remove_config: bool = False,
) -> dict:
    try:
        normalized = normalize_provider_id(provider_id)
        builtin = PROVIDERS.get(normalized)
        if builtin is not None and any(
            (os.getenv(name) or "").strip() for name in builtin.env_names
        ):
            return {
                "configured": True,
                "source": "environment",
                "provider": normalized,
                "detail": "Environment-provided keys must be removed from the process environment.",
            }
        label = validate_provider_label(
            provider_label,
            default=(builtin.label if builtin else normalized),
        )
        delete_provider_api_key(normalized, provider_label=label)
        if remove_config:
            delete_provider_config(normalized)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except SecretStoreUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    return {
        "configured": False,
        "source": "none",
        "provider": normalized,
    }


@router.get("/providers/{provider_id}/models")
def list_provider_models(provider_id: str) -> dict:
    try:
        provider = _resolve_configured_provider(provider_id)
        api_key = get_provider_api_key(
            provider.id,
            provider_label=provider.label,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except SecretStoreUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    if not api_key:
        raise HTTPException(409, f"{provider.label} API key is not configured")

    headers = {"Accept": "application/json"}
    if provider.protocol == "gemini":
        headers["x-goog-api-key"] = api_key
    else:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        response = safe_get(
            provider.models_url,
            headers=headers,
            timeout=(10, 30),
            stream=False,
        )
    except (requests.RequestException, ValueError, HTTPException) as exc:
        raise HTTPException(
            502,
            f"Could not load {provider.label} models",
        ) from exc
    try:
        payload = response.json()
    except ValueError as exc:
        raise HTTPException(
            502,
            f"{provider.label} returned an invalid models response",
        ) from exc
    finally:
        response.close()

    rows = (
        payload.get("models")
        if provider.protocol == "gemini"
        else payload.get("data")
    )
    models = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        model_id = str(
            row.get("name")
            if provider.protocol == "gemini"
            else row.get("id") or ""
        )
        if model_id.startswith("models/"):
            model_id = model_id[7:]
        if model_id:
            models.append(model_id)
    return {
        "provider": provider.id,
        "provider_label": provider.label,
        "models": sorted(set(models))[:500],
    }


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
    if (os.getenv("GEMINI_API_KEY") or "").strip() or (os.getenv("GOOGLE_API_KEY") or "").strip():
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


@router.post(
    "/deepseek/key",
    responses={
        400: {"description": "Invalid DeepSeek API key"},
        503: {"description": "OS secure storage is unavailable"},
    },
)
def save_deepseek_visual_qc_key(req: DeepSeekKeyRequest) -> dict:
    try:
        set_deepseek_api_key(req.api_key)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except SecretStoreUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    return {
        "configured": True,
        "source": "os_secure_storage",
        "model": DEFAULT_DEEPSEEK_MODEL,
    }


@router.delete(
    "/deepseek/key",
    responses={503: {"description": "OS secure storage is unavailable"}},
)
def clear_deepseek_visual_qc_key() -> dict:
    if (os.getenv("DEEPSEEK_API_KEY") or "").strip():
        return {
            "configured": True,
            "source": "environment",
            "model": DEFAULT_DEEPSEEK_MODEL,
            "detail": "Environment-provided keys must be removed from the process environment.",
        }
    try:
        delete_deepseek_api_key()
    except SecretStoreUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    return {"configured": False, "source": "none", "model": DEFAULT_DEEPSEEK_MODEL}


async def _visual_qc_plan() -> None:
    await run_in_threadpool(cloud.require_feature, "visual_qc")


@router.post("/inspect", dependencies=[Depends(_visual_qc_plan)])
async def inspect_visual_qc(req: VisualQCInspectRequest) -> dict:
    validate_chapter_id(req.chapter_id)
    manifest = load_manifest_raw(req.chapter_id)
    pages = manifest.get("pages", [])
    if req.page_index < 0 or req.page_index >= len(pages):
        raise HTTPException(400, f"Invalid page_index: {req.page_index}")

    original_path, cleaned_path = _page_paths(manifest, req.page_index, req.chapter_id)
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
        provider = _resolve_configured_provider(req.provider)
        if not provider.supports_visual_qc:
            raise ValueError(f"{provider.label} is not available for visual QC")
        model = validate_model_name(req.model, default=provider.default_qc_model)
        api_key = get_provider_api_key(
            provider.id,
            provider_label=provider.label,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except SecretStoreUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    if not api_key:
        raise HTTPException(409, f"{provider.label} API key is not configured")

    client = (
        GeminiVisualQC(model=model)
        if provider.protocol == "gemini"
        else OpenAICompatibleVisualQC(
            provider_id=provider.id,
            provider_label=provider.label,
            chat_url=str(provider.chat_url),
            model=model,
            provider=provider,
        )
    )

    try:
        issues = await run_in_threadpool(client.inspect, original_path, cleaned_path, api_key)
    except ValueError as exc:
        raise HTTPException(400, _redact_secret(exc, api_key)) from exc
    except GeminiVisualQCTimeout as exc:
        detail = _redact_secret(exc, api_key)
        logger.warning(
            "Chapter {} page {} {} visual QC timed out: {}",
            req.chapter_id,
            req.page_index,
            provider.label,
            detail,
        )
        raise HTTPException(504, detail) from exc
    except Exception as exc:
        detail = _redact_secret(exc, api_key)
        logger.error(
            "Chapter {} page {} {} visual QC failed: {}",
            req.chapter_id,
            req.page_index,
            provider.label,
            detail,
        )
        raise HTTPException(502, detail) from exc

    try:
        latest_manifest = load_manifest_raw(req.chapter_id)
        latest_pages = latest_manifest.get("pages", [])
        if req.page_index >= len(latest_pages):
            raise HTTPException(409, "Page changed while AI was inspecting it; run AI QC again")
        latest_original, latest_cleaned = _page_paths(latest_manifest, req.page_index, req.chapter_id)
        if (
            latest_original != original_path
            or latest_cleaned != cleaned_path
            or not latest_original.is_file()
            or not latest_cleaned.is_file()
            or _file_revision(latest_original) != original_revision
            or _file_revision(latest_cleaned) != cleaned_revision
        ):
            raise HTTPException(409, "Page changed while AI was inspecting it; run AI QC again")
    except HTTPException:
        raise
    except OSError as exc:
        raise HTTPException(409, "Page changed while AI was inspecting it; run AI QC again") from exc

    return {
        "provider": provider.id,
        "model": client.model,
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


@router.post(
    "/chapter",
    dependencies=[Depends(_visual_qc_plan)],
    responses={
        400: {"description": "Invalid chapter QC request"},
        409: {"description": "Provider key missing or chapter QC already active"},
        429: {"description": "Visual QC job capacity reached"},
        500: {"description": "Chapter visual QC could not be started"},
        503: {"description": "OS secure storage is unavailable"},
    },
)
async def start_chapter_visual_qc(req: VisualQCChapterRequest) -> dict:
    validate_chapter_id(req.chapter_id)
    try:
        provider = _resolve_configured_provider(req.provider)
        service, client = _make_chapter_service(
            provider,
            req.model,
            req.budget_usd,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except SecretStoreUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    try:
        job = await service.start(req.chapter_id, concurrency=req.concurrency)
    except HTTPException:
        raise
    except SecretStoreUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    except ValueError as exc:
        status = 409 if "API key" in str(exc) else 400
        raise HTTPException(status, str(exc)) from exc
    except RuntimeError as exc:
        _raise_job_capacity_error(exc)
    except Exception as exc:
        logger.error("Chapter {} visual QC job failed to start: {}", req.chapter_id, exc)
        raise HTTPException(500, "Could not start chapter visual QC") from exc
    _remember_job(job.job_id, provider=provider, client=client)
    return _enrich_snapshot(chapter_qc_jobs.snapshot(job.job_id))


def _job_snapshot_or_404(job_id: str) -> dict:
    try:
        return _enrich_snapshot(chapter_qc_jobs.snapshot(job_id))
    except KeyError as exc:
        _chapter_qc_context.pop(job_id, None)
        raise HTTPException(404, "Visual QC job not found") from exc


@router.get("/chapter/{job_id}")
def chapter_visual_qc_status(job_id: str) -> dict:
    return _job_snapshot_or_404(job_id)


@router.post("/chapter/{job_id}/cancel")
def cancel_chapter_visual_qc(job_id: str) -> dict:
    snapshot = _job_snapshot_or_404(job_id)
    if snapshot["status"] not in {"completed", "cancelled"}:
        chapter_qc_jobs.cancel(job_id)
    return _enrich_snapshot(chapter_qc_jobs.snapshot(job_id))


@router.post(
    "/chapter/{job_id}/retry",
    dependencies=[Depends(_visual_qc_plan)],
    responses={
        400: {"description": "Invalid retry request"},
        404: {"description": "Visual QC job not found"},
        409: {"description": "Visual QC job cannot be retried in its current state"},
        429: {"description": "Visual QC job capacity reached"},
        500: {"description": "Visual QC retry could not be started"},
        503: {"description": "OS secure storage is unavailable"},
    },
)
async def retry_failed_chapter_visual_qc(job_id: str) -> dict:
    previous = _job_snapshot_or_404(job_id)
    if previous["status"] not in {"completed", "cancelled"}:
        raise HTTPException(409, "Visual QC job is still running")
    if int(previous.get("failed") or 0) <= 0:
        raise HTTPException(409, "Visual QC job has no failed regions to retry")

    context = _chapter_qc_context.get(job_id) or {"provider": "gemini"}
    provider_id = str(context.get("provider") or "gemini")
    client = context.get("client")
    if not isinstance(client, (OpenAICompatibleRegionQC, GeminiRegionQC)):
        raise HTTPException(409, "AI QC job context is no longer available")
    service = (
        _make_openai_service(client)
        if isinstance(client, OpenAICompatibleRegionQC)
        else ChapterQCService(
            RegionBatchRunner(client),
            chapter_qc_jobs,
            api_key_provider=lambda: get_provider_api_key(provider_id),
            model=client.model,
            provider=provider_id,
        )
    )
    try:
        job = await service.start(
            previous["chapter_id"],
            concurrency=int(previous.get("concurrency") or 2),
        )
    except HTTPException:
        raise
    except SecretStoreUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    except ValueError as exc:
        status = 409 if "API key" in str(exc) else 400
        raise HTTPException(status, str(exc)) from exc
    except RuntimeError as exc:
        _raise_job_capacity_error(exc)
    retry_provider = (
        _resolve_configured_provider(provider_id)
        if provider_id != "gemini"
        else PROVIDERS["gemini"]
    )
    _remember_job(job.job_id, provider=retry_provider, client=client)
    return _enrich_snapshot(chapter_qc_jobs.snapshot(job.job_id))
