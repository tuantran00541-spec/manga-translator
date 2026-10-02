from __future__ import annotations

import os

import requests

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator

from app.ai_providers import (
    PROVIDERS,
    cloud_job_provider,
    normalize_provider_id,
    resolve_provider,
    validate_provider_label,
)
from app.downloader.http import safe_get
from app.schemas import VisualQCKeyRequest
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
from app.security import validate_url
from app.visual_qc.deepseek_region_client import DEFAULT_DEEPSEEK_MODEL
from app.visual_qc.gemini import DEFAULT_GEMINI_MODEL

router = APIRouter(prefix="/api/visual_qc", tags=["visual-qc"])


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
