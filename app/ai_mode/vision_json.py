from __future__ import annotations

import copy
import json
from dataclasses import dataclass

import numpy as np
import requests
from fastapi import HTTPException

from app import api_proxy
from app.ai_providers import CLOUD_PROVIDER_ID, PROVIDERS, AIProvider, auth_headers, is_deepseek_v41_model
from app.parameters import TRANSLATION_CONNECT_TIMEOUT_SECONDS, TRANSLATION_READ_TIMEOUT_SECONDS
from app.security import validate_url
from app.translation.cost import usage_cost_usd
from app.visual_qc.deepseek_region_client import _extract_output_text, _safe_error_detail
from app.visual_qc.gemini import _encode_for_gemini
from app.visual_qc.gemini_interactions import (
    GEMINI_INTERACTIONS_URL,
    extract_output_text as gemini_output_text,
    safe_error_detail as gemini_safe_error,
)


@dataclass(frozen=True)
class VisionJSONResult:
    data: dict
    usage: dict
    estimated_cost_usd: float | None


def parse_json_object(content: str) -> dict:
    source = str(content or "").strip()
    if source.startswith("```"):
        source = source.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        data = json.loads(source)
    except (TypeError, ValueError):
        start = source.find("{")
        try:
            data = json.JSONDecoder().raw_decode(source, start)[0] if start >= 0 else None
        except ValueError:
            data = api_proxy.repair_json_object(source)
            if data is not None:
                api_proxy.record_external_rescue(
                    caller="ai_mode",
                    provider="",
                    model="",
                    status="rescued",
                    error_detail="JSON output malformed or truncated",
                    error_category="malformed_json",
                    rescue_action="Tự sửa JSON lỗi / bị cắt từ phản hồi model",
                    rescue_type="json_repair",
                )
        if data is None:
            raise RuntimeError("Vision model returned invalid JSON")
    if not isinstance(data, dict):
        raise RuntimeError("Vision model must return a JSON object")
    return data


def _merge_json_dicts(parts: list[dict]) -> dict:
    merged: dict = {}
    for part in parts:
        if not isinstance(part, dict):
            continue
        for key, value in part.items():
            if isinstance(value, list):
                copied = copy.deepcopy(value)
                if isinstance(merged.get(key), list):
                    merged[key].extend(copied)
                else:
                    merged[key] = copied
            elif isinstance(value, dict):
                existing = merged.get(key)
                merged[key] = _merge_json_dicts([existing, value]) if isinstance(existing, dict) else copy.deepcopy(value)
            elif key not in merged or merged[key] in (None, "", []):
                merged[key] = copy.deepcopy(value)
    return merged


def sum_usage_dicts(into: dict, src: dict | None) -> None:
    if not isinstance(src, dict):
        return
    for key, value in src.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            existing = into.get(key, 0)
            if isinstance(existing, (int, float)) and not isinstance(existing, bool):
                into[key] = existing + value
        elif isinstance(value, dict):
            sub = into.get(key)
            if not isinstance(sub, dict):
                sub = {}
                into[key] = sub
            sum_usage_dicts(sub, value)


def _merge_vision_results(parts: list[VisionJSONResult]) -> VisionJSONResult:
    usage: dict = {}
    costs: list[float] = []
    for part in parts:
        sum_usage_dicts(usage, part.usage)
        if part.estimated_cost_usd is not None:
            costs.append(part.estimated_cost_usd)
    return VisionJSONResult(
        _merge_json_dicts([part.data for part in parts]),
        usage,
        round(sum(costs), 6) if costs else None,
    )


def request_vision_json(
    provider: AIProvider,
    model: str,
    api_key: str,
    prompt: str,
    images: list[tuple[str, np.ndarray]],
    *,
    schema: dict | None = None,
    max_tokens: int = 2048,
    reasoning_effort: str | None = None,
    stage: str | None = None,
) -> VisionJSONResult:
    key = str(api_key or "").strip()
    if provider.id != CLOUD_PROVIDER_ID and key:
        try:
            from app.secret_store import get_provider_api_key, get_provider_api_keys

            pool = get_provider_api_keys(provider.id, provider_label=provider.label)
            if len(pool) > 1 and key in pool:
                rotated = get_provider_api_key(provider.id, provider_label=provider.label)
                if rotated:
                    key = rotated.strip()
        except Exception:
            pass
    if not key:
        raise ValueError(f"{provider.label} API key is not configured")
    wrapped_cap = api_proxy.get_wrapped_image_cap(str(provider.api_base or "").rstrip("/"), model)
    if wrapped_cap and len(images) > wrapped_cap:
        chunks = [images[idx : idx + wrapped_cap] for idx in range(0, len(images), wrapped_cap)]
        api_proxy.record_external_rescue(
            caller=f"ai_mode:{stage}" if stage else "ai_mode",
            provider=provider.label,
            model=model,
            status="rescued",
            error_category="auto_wrapped",
            rescue_action=f"Bọc sẵn — tự chia {len(images)} ảnh thành {len(chunks)} đợt (≤{wrapped_cap} ảnh)",
            rescue_type="too_many_images",
        )
        return _merge_vision_results([
            request_vision_json(
                provider, model, key, prompt, chunk,
                schema=schema, max_tokens=max_tokens, reasoning_effort=reasoning_effort, stage=stage,
            )
            for chunk in chunks
        ])
    try:
        encoded = [(label, _encode_for_gemini(image)) for label, image in images]
        if provider.protocol == "gemini":
            return _gemini(model, key, prompt, encoded, schema=schema, max_tokens=max_tokens)
        return _openai(
            provider,
            model,
            key,
            prompt,
            encoded,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            stage=stage,
        )
    except api_proxy.ImageLimitExceeded as exc:
        if len(images) <= 1:
            raise RuntimeError(f"{provider.label} rejected image payload: {exc.detail}") from exc
        cap = exc.limit if exc.limit and 1 <= exc.limit < len(images) else max(1, len(images) // 2)
        chunks = [images[idx : idx + cap] for idx in range(0, len(images), cap)]
        results = [
            request_vision_json(
                provider,
                model,
                key,
                prompt,
                chunk,
                schema=schema,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
                stage=stage,
            )
            for chunk in chunks
        ]
        return _merge_vision_results(results)


def stage_headers(provider: AIProvider, stage: str | None) -> dict:
    return {"X-MT-Stage": stage} if stage and provider.id == CLOUD_PROVIDER_ID else {}


def _openai(provider, model, api_key, prompt, encoded, *, max_tokens, reasoning_effort=None, stage=None) -> VisionJSONResult:
    url = str(provider.chat_url or "")
    if not url:
        raise ValueError("Provider does not have chat completions")
    if not provider.builtin:
        validate_url(url)
    content: list[dict] = [{"type": "text", "text": prompt}]
    for label, data in encoded:
        content.append({"type": "text", "text": label})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}})
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "response_format": {"type": "json_object"},
        "max_tokens": max_tokens,
        "stream": False,
    }
    payload.update(provider.chat_completion_extras(model))
    if reasoning_effort and (provider.id == CLOUD_PROVIDER_ID or is_deepseek_v41_model(model)):
        payload["reasoning_effort"] = reasoning_effort
    response = None
    try:
        try:
            response, _, _ = api_proxy.proxy_post(
                url=url,
                headers={**auth_headers(provider, api_key), "Content-Type": "application/json",
                         **stage_headers(provider, stage)},
                payload=payload,
                provider=provider,
                api_key=api_key,
                caller=f"ai_mode:{stage}" if stage else "ai_mode",
                timeout=(TRANSLATION_CONNECT_TIMEOUT_SECONDS, TRANSLATION_READ_TIMEOUT_SECONDS),
                stream=False,
                is_gemini=False,
                raise_on_image_split=len(encoded) > 1,
                allow_image_strip=False,
                allow_local=provider.builtin,
            )
        except (requests.RequestException, HTTPException) as exc:
            raise RuntimeError(f"{provider.label} vision request failed") from exc
        status = int(getattr(response, "status_code", 0) or 0)
        if 300 <= status < 400:
            raise RuntimeError(f"{provider.label} redirected the image request")
        if not api_proxy.is_success_response(response):
            raise RuntimeError(
                f"{provider.label} HTTP {status}: {_safe_error_detail(response, api_key)}"
            )
        try:
            body = response.json()
            answer = _extract_output_text(body)
        except (TypeError, KeyError, ValueError) as exc:
            raise RuntimeError(f"{provider.label} returned no text") from exc
        usage = body.get("usage") if isinstance(body, dict) and isinstance(body.get("usage"), dict) else {}
        cost = usage_cost_usd(usage) if provider.tracks_cost else None
        return VisionJSONResult(parse_json_object(answer), usage, cost)
    finally:
        if response is not None and hasattr(response, "close"):
            response.close()


def _thinking_level(model: str) -> str:
    return "minimal" if str(model or "").lower().startswith("gemma-") else "low"


def _gemini(model, api_key, prompt, encoded, *, schema, max_tokens) -> VisionJSONResult:
    inputs: list[dict] = [{"type": "text", "text": prompt}]
    for label, data in encoded:
        inputs.append({"type": "text", "text": label})
        inputs.append({"type": "image", "data": data, "mime_type": "image/jpeg"})
    response_format = {"type": "text", "mime_type": "application/json"}
    if schema is not None:
        response_format["schema"] = schema
    payload = {
        "model": model,
        "store": False,
        "input": inputs,
        "response_format": response_format,
        "generation_config": {"thinking_level": _thinking_level(model), "max_output_tokens": max_tokens},
    }
    response = None
    try:
        try:
            response, _, _ = api_proxy.proxy_post(
                url=GEMINI_INTERACTIONS_URL,
                headers={"Content-Type": "application/json", "x-goog-api-key": api_key, "Api-Revision": "2026-05-20"},
                payload=payload,
                provider=PROVIDERS["gemini"],
                api_key=api_key,
                caller="ai_mode:gemini",
                timeout=(TRANSLATION_CONNECT_TIMEOUT_SECONDS, TRANSLATION_READ_TIMEOUT_SECONDS),
                stream=False,
                is_gemini=True,
                raise_on_image_split=len(encoded) > 1,
                allow_image_strip=False,
                allow_local=True,
            )
        except (requests.RequestException, HTTPException) as exc:
            raise RuntimeError("Gemini vision request failed") from exc
        status = int(getattr(response, "status_code", 0) or 0)
        if 300 <= status < 400:
            raise RuntimeError("Gemini redirected the image request")
        if not api_proxy.is_success_response(response):
            raise RuntimeError(f"Gemini HTTP {status}: {gemini_safe_error(response, api_key)}")
        try:
            body = response.json()
            answer = gemini_output_text(body)
        except (TypeError, KeyError, ValueError) as exc:
            raise RuntimeError("Gemini returned no text") from exc
        usage = body.get("usage") if isinstance(body, dict) and isinstance(body.get("usage"), dict) else {}
        return VisionJSONResult(parse_json_object(answer), usage, None)
    finally:
        if response is not None and hasattr(response, "close"):
            response.close()
