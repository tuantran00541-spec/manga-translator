from __future__ import annotations

import ipaddress
import json
import os
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from app import api_proxy
from app.ai_providers import (
    PROVIDERS,
    AIProvider,
    auth_headers,
    normalize_provider_id,
    resolve_provider,
)
from app.parameters import TRANSLATION_CONNECT_TIMEOUT_SECONDS, TRANSLATION_READ_TIMEOUT_SECONDS
from app.secret_store import (
    SecretStoreUnavailable,
    get_provider_api_key,
    get_provider_config,
    list_provider_configs,
)
from app.security import validate_url

router = APIRouter(prefix="/api/proxy", tags=["api-proxy"])

GEMINI_OPENAI_CHAT_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"


def _is_loopback_host(host: str) -> bool:
    if host == "testclient":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host in ("localhost", "127.0.0.1", "::1")


def _require_loopback(request: Request) -> None:
    host = request.client.host if request.client else ""
    if not _is_loopback_host(host):
        raise HTTPException(403, "API proxy is local only.")


def _resolve_provider_only(provider_id: str) -> AIProvider:
    normalized = normalize_provider_id(provider_id)
    if normalized in PROVIDERS:
        return PROVIDERS[normalized]
    stored = get_provider_config(normalized)
    if stored is None:
        raise ValueError(f"Unknown AI provider: {provider_id}")
    return resolve_provider(
        normalized,
        label=stored.get("label"),
        protocol=stored.get("protocol"),
        api_base=stored.get("api_base"),
    )


def _configured_providers() -> list[AIProvider]:
    found: list[AIProvider] = []
    for pid, provider in PROVIDERS.items():
        try:
            if get_provider_api_key(pid, provider_label=provider.label):
                found.append(provider)
        except Exception:
            continue
    for cfg in list_provider_configs() or []:
        if not isinstance(cfg, dict):
            continue
        pid = str(cfg.get("id") or "").strip()
        if not pid:
            continue
        try:
            provider = resolve_provider(
                pid,
                label=cfg.get("label"),
                protocol=cfg.get("protocol"),
                api_base=cfg.get("api_base"),
            )
            if get_provider_api_key(provider.id, provider_label=provider.label):
                found.append(provider)
        except Exception:
            continue
    return found


def _resolve_proxy_target(request: Request, body: dict[str, Any]) -> tuple[AIProvider | None, str, str, str]:
    """Resolve ``(provider, chat_url, api_key, model)`` for incoming local proxy calls."""
    explicit_base = (
        request.headers.get("X-Target-Base-URL")
        or str(body.get("target_base_url") or body.get("base_url") or "")
    ).strip()
    auth_header = (request.headers.get("Authorization") or "").strip()
    bearer_key = auth_header[7:].strip() if auth_header.lower().startswith("bearer ") else ""
    model = str(body.get("model") or "").strip()
    header_provider = (
        request.headers.get("X-Target-Provider")
        or request.query_params.get("provider")
        or str(body.get("provider") or "")
    ).strip()

    if explicit_base:
        validate_url(explicit_base)
        provider = resolve_provider(
            header_provider or "proxy-target",
            label=header_provider or "Proxy Target",
            protocol="openai",
            api_base=explicit_base,
        )
        key = bearer_key or (get_provider_api_key(provider.id, provider_label=provider.label) or "")
        if not key:
            raise HTTPException(409, "Missing upstream API key for proxy target.")
        return provider, str(provider.chat_url), key, model

    provider_id = header_provider
    if not provider_id and ":" in model:
        candidate_id, rest_model = model.split(":", 1)
        try:
            _resolve_provider_only(candidate_id)
            provider_id = candidate_id
            model = rest_model.strip()
        except ValueError:
            pass

    if provider_id:
        try:
            provider = _resolve_provider_only(provider_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
    else:
        configured = _configured_providers()
        openai_first = next((p for p in configured if p.protocol == "openai"), None)
        provider = openai_first or (configured[0] if configured else None)
        if provider is None:
            raise HTTPException(409, "Chưa có nhà cung cấp AI nào được cấu hình khóa API.")

    try:
        key = get_provider_api_key(provider.id, provider_label=provider.label) or bearer_key
    except SecretStoreUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    if not key:
        raise HTTPException(409, f"Chưa cấu hình khóa API cho {provider.label}.")

    target_url = GEMINI_OPENAI_CHAT_URL if provider.protocol == "gemini" else str(provider.chat_url or "")
    if not target_url:
        raise HTTPException(400, f"{provider.label} không có chat completions endpoint.")
    if not model:
        model = provider.default_translation_model or provider.default_qc_model or ""
    return provider, target_url, key, model


@router.get("/status")
def proxy_status(request: Request, limit: int = 80) -> dict:
    _require_loopback(request)
    snap = api_proxy.proxy_snapshot(limit=limit)
    port = os.getenv("PORT", "8000")
    snap["local_endpoint"] = f"http://127.0.0.1:{port}/api/proxy/v1"
    return snap


@router.post("/logs/clear")
def proxy_clear_logs(request: Request) -> dict:
    _require_loopback(request)
    api_proxy.clear_proxy_logs(include_learned=True)
    return {"cleared": True}


@router.post("/unwrap")
async def proxy_unwrap_target(request: Request) -> dict:
    _require_loopback(request)
    try:
        body = await request.json()
    except ValueError as exc:
        raise HTTPException(400, "Invalid JSON request body") from exc
    if not isinstance(body, dict):
        raise HTTPException(400, "Invalid unwrap payload")
    removed = api_proxy.unwrap_target(str(body.get("api_base") or ""), str(body.get("model") or ""))
    return {"unwrapped": removed}


@router.get("/v1/models")
def proxy_list_models(request: Request) -> dict:
    _require_loopback(request)
    rows = []
    seen: set[str] = set()
    for provider in _configured_providers():
        candidates = [c for c in (provider.default_qc_model, provider.default_translation_model) if c]
        if not candidates:
            candidates = ["default"]
        for candidate in candidates:
            mid = f"{provider.id}:{candidate}"
            if mid not in seen:
                seen.add(mid)
                rows.append({"id": mid, "object": "model", "owned_by": provider.id})
    return {"object": "list", "data": rows}


def _sse_from_completion(body: dict) -> StreamingResponse:
    """Wrap a rescued non-stream chat completion as an OpenAI-compatible SSE stream."""
    choice = (body.get("choices") or [{}])[0] if isinstance(body.get("choices"), list) else {}
    msg = choice.get("message") or {}
    chunk: dict[str, Any] = {
        "id": body.get("id") or "chatcmpl-proxy-rescued",
        "object": "chat.completion.chunk",
        "created": body.get("created") or 0,
        "model": body.get("model") or "",
        "choices": [{
            "index": 0,
            "delta": {k: v for k, v in msg.items() if v is not None},
            "finish_reason": choice.get("finish_reason") or "stop",
        }],
    }
    if isinstance(body.get("usage"), dict):
        chunk["usage"] = body["usage"]

    def _gen():
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8")
        yield b"data: [DONE]\n\n"

    return StreamingResponse(_gen(), media_type="text/event-stream")


@router.post("/v1/chat/completions")
async def proxy_chat_completions(request: Request):
    _require_loopback(request)
    try:
        body = await request.json()
    except ValueError as exc:
        raise HTTPException(400, "Invalid JSON request body") from exc
    if not isinstance(body, dict):
        raise HTTPException(400, "Request body must be a JSON object")

    provider, target_url, api_key, model = await run_in_threadpool(_resolve_proxy_target, request, body)
    upstream_payload = dict(body)
    if model:
        upstream_payload["model"] = model
    want_stream = bool(upstream_payload.get("stream"))
    asked_json = isinstance(body.get("response_format"), dict)

    headers = {
        "Authorization": f"Bearer {api_key}",
        **(auth_headers(provider, api_key) if provider and provider.protocol != "gemini" else {}),
        "Content-Type": "application/json",
    }

    try:
        response, final_payload, meta = await run_in_threadpool(
            lambda: api_proxy.proxy_post(
                url=target_url,
                headers=headers,
                payload=upstream_payload,
                provider=provider,
                api_key=api_key,
                caller="local_proxy",
                timeout=(TRANSLATION_CONNECT_TIMEOUT_SECONDS, TRANSLATION_READ_TIMEOUT_SECONDS),
                stream=want_stream,
                is_gemini=False,
                allow_tool_strip=True,
                allow_local=provider.builtin,
            )
        )
    except Exception as exc:
        raise HTTPException(502, f"Proxy upstream error: {exc}") from exc

    if not getattr(response, "ok", False):
        status_code = int(getattr(response, "status_code", 502) or 502)
        try:
            err_body = response.json()
        except ValueError:
            err_body = {"error": {"message": getattr(response, "text", "")[:600], "proxy_meta": meta}}
        if hasattr(response, "close"):
            response.close()
        return JSONResponse(err_body, status_code=status_code)

    upstream_streamed = bool(final_payload.get("stream"))
    if want_stream and upstream_streamed:
        def _iter_stream():
            try:
                for chunk in response.iter_content(chunk_size=None):
                    if chunk:
                        yield chunk
            finally:
                if hasattr(response, "close"):
                    response.close()

        return StreamingResponse(_iter_stream(), media_type="text/event-stream")

    try:
        data = response.json()
    except ValueError as exc:
        text = getattr(response, "text", "")
        if hasattr(response, "close"):
            response.close()
        repaired = api_proxy.repair_json_object(text)
        if repaired is not None:
            data = repaired
        else:
            raise HTTPException(502, "Upstream returned invalid JSON") from exc
    else:
        if hasattr(response, "close"):
            response.close()

    # Best-effort repair of JSON message content when caller requested json_object.
    if asked_json and isinstance(data, dict):
        choices = data.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            msg = choices[0].get("message")
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                raw_text = msg["content"]
                try:
                    json.loads(raw_text)
                except ValueError:
                    repaired = api_proxy.repair_json_object(raw_text)
                    if repaired is not None:
                        msg["content"] = json.dumps(repaired, ensure_ascii=False)
                        api_proxy.record_external_rescue(
                            caller="local_proxy",
                            provider=provider.label if provider else "",
                            model=model,
                            status="rescued",
                            error_detail="JSON completion content malformed",
                            error_category="malformed_json",
                            rescue_action="Tự sửa JSON lỗi trong message.content",
                            rescue_type="json_repair",
                        )

    if want_stream and not upstream_streamed and isinstance(data, dict):
        return _sse_from_completion(data)
    return JSONResponse(data, status_code=200)
