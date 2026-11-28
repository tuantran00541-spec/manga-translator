from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import replace

import requests

from fastapi import APIRouter, HTTPException

from app.ai_providers import (
    PROVIDERS,
    api_base_candidates,
    auth_headers,
    cloud_job_provider,
    normalize_api_base,
    normalize_provider_id,
    resolve_provider,
    validate_api_base,
    validate_provider_label,
)
from app.downloader.http import read_response_limited, safe_get
from app.schemas import KeyRotationSaveRequest, VisualQCKeyRequest
from app.secret_store import (
    MAX_KEYS,
    SecretStoreUnavailable,
    delete_provider_api_key,
    delete_provider_config,
    get_provider_api_key,
    get_provider_config,
    get_provider_key_hints,
    list_provider_configs,
    parse_key_lines,
    provider_key_status,
    set_provider_api_key,
    set_provider_api_keys,
    set_provider_config,
)
from app.security import validate_url
from app.visual_qc.gemini import DEFAULT_GEMINI_MODEL

router = APIRouter(prefix="/api/visual_qc", tags=["visual-qc"])

_MODELS_CACHE_TTL_SECONDS = 300.0
_MODELS_CACHE_MAX_ENTRIES = 64
_MAX_MODELS_RESPONSE_BYTES = 4 * 1024 * 1024
_MODELS_CACHE_LOCK = threading.Lock()
_MODELS_CACHE: dict[tuple[str, str, str, str], tuple[float, dict]] = {}
_MODELS_CACHE_GENERATION = 0
_MODELS_CACHE_PROVIDER_GENERATIONS: dict[str, int] = {}


def clear_models_cache(provider_id: str | None = None) -> None:
    """Clear cached provider model lists, all entries or for one provider."""
    global _MODELS_CACHE_GENERATION
    with _MODELS_CACHE_LOCK:
        if not provider_id:
            _MODELS_CACHE.clear()
            _MODELS_CACHE_GENERATION += 1
            _MODELS_CACHE_PROVIDER_GENERATIONS.clear()
            return
        normalized = normalize_provider_id(provider_id)
        _MODELS_CACHE_PROVIDER_GENERATIONS[normalized] = (
            _MODELS_CACHE_PROVIDER_GENERATIONS.get(normalized, 0) + 1
        )
        for key in [k for k in _MODELS_CACHE if k[0] == normalized]:
            _MODELS_CACHE.pop(key, None)


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


def _read_json_limited(response) -> object:
    if hasattr(response, "iter_content"):
        raw = read_response_limited(response, limit_bytes=_MAX_MODELS_RESPONSE_BYTES)
        return json.loads(raw.decode("utf-8"))
    return response.json()


def _answers_models(provider, api_key: str) -> bool:
    headers = {"Accept": "application/json", **auth_headers(provider, api_key)}
    try:
        response = safe_get(provider.models_url, headers=headers, timeout=(5, 15), stream=True, max_redirects=0)
    except requests.HTTPError as exc:
        if exc.response is not None:
            try:
                status = exc.response.status_code
                return status in (401, 403) and "json" in (exc.response.headers.get("content-type") or "")
            finally:
                if hasattr(exc.response, "close"):
                    exc.response.close()
        return False
    except (requests.RequestException, ValueError, HTTPException):
        return False
    try:
        payload = _read_json_limited(response)
        return isinstance(payload, dict) and isinstance(payload.get("data"), list)
    except (ValueError, UnicodeDecodeError, AttributeError):
        return False
    finally:
        if hasattr(response, "close"):
            response.close()


def _discover_api_base(provider, api_key: str):
    candidates = api_base_candidates(provider.api_base)
    if len(candidates) == 1:
        return provider
    for base in candidates:
        trial = replace(provider, api_base=base)
        if _answers_models(trial, api_key):
            return trial
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
            provider = _discover_api_base(provider, req.api_key)
            set_provider_config(
                normalized,
                label=provider.label,
                protocol=provider.protocol,
                api_base=provider.api_base,
            )
        count = set_provider_api_key(
            normalized,
            req.api_key,
            provider_label=label,
            append=req.append,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except SecretStoreUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    clear_models_cache(normalized)
    result = {
        "configured": True,
        "source": "os_secure_storage",
        "provider": normalized,
        "count": count,
    }
    if builtin is None:
        result["api_base"] = provider.api_base
    return result


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
    clear_models_cache(normalized)
    return {
        "configured": False,
        "source": "none",
        "provider": normalized,
    }


@router.get("/providers/{provider_id}/models")
def list_provider_models(provider_id: str, refresh: bool = False) -> dict:
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

    key_digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]
    cache_key = (provider.id, provider.models_url or "", provider.protocol, key_digest)
    now = time.monotonic()
    with _MODELS_CACHE_LOCK:
        fetch_gen = (_MODELS_CACHE_GENERATION, _MODELS_CACHE_PROVIDER_GENERATIONS.get(provider.id, 0))
        if not refresh:
            cached = _MODELS_CACHE.get(cache_key)
            if cached and (now - cached[0]) < _MODELS_CACHE_TTL_SECONDS:
                entry = cached[1]
                return {
                    "provider": entry["provider"],
                    "provider_label": entry["provider_label"],
                    "models": list(entry["models"]),
                }

    headers = {"Accept": "application/json"}
    if provider.protocol == "gemini":
        headers["x-goog-api-key"] = api_key
    else:
        headers.update(auth_headers(provider, api_key))
    try:
        response = safe_get(
            provider.models_url,
            headers=headers,
            timeout=(10, 30),
            stream=True,
            max_redirects=0,
        )
    except (requests.RequestException, ValueError, HTTPException) as exc:
        raise HTTPException(
            502,
            f"Could not load {provider.label} models",
        ) from exc
    try:
        payload = _read_json_limited(response)
    except (ValueError, UnicodeDecodeError) as exc:
        raise HTTPException(
            502,
            f"{provider.label} returned an invalid models response",
        ) from exc
    finally:
        if hasattr(response, "close"):
            response.close()

    if not isinstance(payload, dict):
        raise HTTPException(
            502,
            f"{provider.label} returned an invalid models response",
        )

    rows = (
        payload.get("models")
        if provider.protocol == "gemini"
        else payload.get("data")
    )
    models = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        raw_id = row.get("name") if provider.protocol == "gemini" else row.get("id")
        model_id = str(raw_id).strip() if raw_id is not None else ""
        if model_id.startswith("models/"):
            model_id = model_id[7:]
        if model_id:
            models.append(model_id)
    result = {
        "provider": provider.id,
        "provider_label": provider.label,
        "models": sorted(set(models))[:500],
    }
    with _MODELS_CACHE_LOCK:
        current_gen = (_MODELS_CACHE_GENERATION, _MODELS_CACHE_PROVIDER_GENERATIONS.get(provider.id, 0))
        if current_gen == fetch_gen:
            if cache_key not in _MODELS_CACHE and len(_MODELS_CACHE) >= _MODELS_CACHE_MAX_ENTRIES:
                oldest_key = min(_MODELS_CACHE, key=lambda k: _MODELS_CACHE[k][0])
                _MODELS_CACHE.pop(oldest_key, None)
            _MODELS_CACHE[cache_key] = (
                time.monotonic(),
                {
                    "provider": result["provider"],
                    "provider_label": result["provider_label"],
                    "models": tuple(result["models"]),
                },
            )
    return result


def _base_url_variants(raw_url: str) -> set[str]:
    cleaned = (raw_url or "").strip().rstrip("/").lower()
    if not cleaned:
        return set()
    variants = {cleaned}
    norm = normalize_api_base(cleaned).rstrip("/").lower()
    if norm:
        variants.add(norm)
    return variants


def _find_provider_by_base_url(base_url: str):
    target_variants = _base_url_variants(base_url)
    for provider in PROVIDERS.values():
        if target_variants & _base_url_variants(provider.api_base):
            return provider
    for cfg in list_provider_configs():
        if not isinstance(cfg, dict):
            continue
        if target_variants & _base_url_variants(str(cfg.get("api_base") or "")):
            try:
                return resolve_provider(
                    str(cfg.get("id") or ""),
                    label=cfg.get("label"),
                    protocol=cfg.get("protocol"),
                    api_base=cfg.get("api_base"),
                )
            except ValueError:
                continue
    return None


def _slug_provider_id_from_base(base_url: str) -> tuple[str, str]:
    import re
    from urllib.parse import urlparse

    parsed = urlparse(base_url)
    host = (parsed.hostname or "custom-provider").lower()
    base_slug = re.sub(r"[^a-z0-9]+", "-", host).strip("-")[:52] or "custom-provider"
    if base_slug in PROVIDERS:
        base_slug = f"{base_slug}-pool"
    existing_ids = {str(cfg.get("id") or "").lower() for cfg in list_provider_configs()} | set(PROVIDERS)
    candidate = base_slug
    idx = 2
    while candidate in existing_ids:
        candidate = f"{base_slug}-{idx}"
        idx += 1
    return candidate, host


@router.get("/key-rotation")
def get_key_rotation_pools() -> dict:
    pools: list[dict] = []
    seen: set[str] = set()

    for provider in PROVIDERS.values():
        status = provider_key_status(provider.id)
        hints = get_provider_key_hints(provider.id)
        if not status.get("configured") and not hints:
            continue
        seen.add(provider.id)
        pools.append(
            {
                "id": provider.id,
                "label": provider.label,
                "api_base": provider.api_base,
                "protocol": provider.protocol,
                "builtin": True,
                "configured": bool(status.get("configured")),
                "source": status.get("source", "none"),
                "count": len(hints) or int(status.get("count") or 0),
                "key_hints": hints,
            }
        )

    try:
        custom_configs = list_provider_configs()
    except SecretStoreUnavailable:
        custom_configs = []

    for cfg in custom_configs:
        pid = str(cfg.get("id") or "").strip().lower()
        if not pid or pid in seen:
            continue
        try:
            provider = resolve_provider(
                pid,
                label=cfg.get("label"),
                protocol=cfg.get("protocol"),
                api_base=cfg.get("api_base"),
            )
        except ValueError:
            continue
        status = provider_key_status(provider.id, provider_label=provider.label)
        hints = get_provider_key_hints(provider.id, provider_label=provider.label)
        seen.add(provider.id)
        pools.append(
            {
                "id": provider.id,
                "label": provider.label,
                "api_base": provider.api_base,
                "protocol": provider.protocol,
                "builtin": False,
                "configured": bool(status.get("configured")),
                "source": status.get("source", "none"),
                "count": len(hints) or int(status.get("count") or 0),
                "key_hints": hints,
            }
        )

    return {"pools": pools, "max_keys": MAX_KEYS}


@router.post("/key-rotation")
def save_key_rotation_pool(req: KeyRotationSaveRequest) -> dict:
    try:
        keys = parse_key_lines(req.keys_text)
        if not keys:
            raise ValueError("Không tìm thấy khóa API hợp lệ nào (mỗi khóa nằm trên một dòng).")
        normalized_base = validate_api_base(req.base_url)
        provider = _find_provider_by_base_url(normalized_base)
        created = False
        if provider is None:
            new_id, default_label = _slug_provider_id_from_base(normalized_base)
            label = validate_provider_label(req.provider_label, default=default_label)
            provider = resolve_provider(
                new_id,
                label=label,
                protocol="openai",
                api_base=normalized_base,
            )
            _validate_custom_remote(provider)
            set_provider_config(
                provider.id,
                label=provider.label,
                protocol=provider.protocol,
                api_base=provider.api_base,
            )
            created = True
        count = set_provider_api_keys(
            provider.id,
            keys,
            provider_label=provider.label,
            append=req.append,
        )
        hints = get_provider_key_hints(provider.id, provider_label=provider.label)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except SecretStoreUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc

    clear_models_cache(provider.id)
    return {
        "provider_id": provider.id,
        "label": provider.label,
        "api_base": provider.api_base,
        "count": count,
        "parsed_count": len(keys),
        "key_hints": hints,
        "created": created,
    }


@router.delete("/key-rotation/{provider_id}")
def clear_key_rotation_pool(provider_id: str) -> dict:
    try:
        normalized = normalize_provider_id(provider_id)
        builtin = PROVIDERS.get(normalized)
        cfg = get_provider_config(normalized)
        label = (
            builtin.label
            if builtin is not None
            else str((cfg or {}).get("label") or normalized)
        )
        delete_provider_api_key(normalized, provider_label=label)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except SecretStoreUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    clear_models_cache(normalized)
    return {"cleared": True, "provider_id": normalized}
