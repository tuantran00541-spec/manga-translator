"""Send labelled images plus a prompt to a vision provider and parse JSON back.

Reuses the provider contracts, image encoding and error redaction already used
by visual QC and vision translation, for both the Gemini interactions API and
OpenAI-compatible chat completions.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import requests

from app.ai_providers import AIProvider
from app.parameters import TRANSLATION_CONNECT_TIMEOUT_SECONDS, TRANSLATION_READ_TIMEOUT_SECONDS
from app.security import validate_url
from app.translation.deepseek import _usage_cost_usd
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
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Vision model returned invalid JSON") from exc
    if not isinstance(data, dict):
        raise RuntimeError("Vision model must return a JSON object")
    return data


def request_vision_json(
    provider: AIProvider,
    model: str,
    api_key: str,
    prompt: str,
    images: list[tuple[str, np.ndarray]],
    *,
    schema: dict | None = None,
    max_tokens: int = 2048,
) -> VisionJSONResult:
    """Return the model's JSON object for ``prompt`` and labelled ``images``."""
    key = str(api_key or "").strip()
    if not key:
        raise ValueError(f"{provider.label} API key is not configured")
    encoded = [(label, _encode_for_gemini(image)) for label, image in images]
    if provider.protocol == "gemini":
        return _gemini(model, key, prompt, encoded, schema=schema, max_tokens=max_tokens)
    return _openai(provider, model, key, prompt, encoded, max_tokens=max_tokens)


def _openai(provider, model, api_key, prompt, encoded, *, max_tokens) -> VisionJSONResult:
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
    payload.update(provider.chat_completion_extras())
    try:
        response = requests.post(
            url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=(TRANSLATION_CONNECT_TIMEOUT_SECONDS, TRANSLATION_READ_TIMEOUT_SECONDS),
            allow_redirects=False,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"{provider.label} vision request failed") from exc
    if 300 <= response.status_code < 400:
        raise RuntimeError(f"{provider.label} redirected the image request")
    if not response.ok:
        raise RuntimeError(
            f"{provider.label} HTTP {response.status_code}: {_safe_error_detail(response, api_key)}"
        )
    try:
        body = response.json()
        answer = _extract_output_text(body)
    except (TypeError, KeyError, ValueError) as exc:
        raise RuntimeError(f"{provider.label} returned no text") from exc
    usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
    cost = _usage_cost_usd(usage) if provider.tracks_cost else None
    return VisionJSONResult(parse_json_object(answer), usage, cost)


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
        "generation_config": {"thinking_level": "low", "max_output_tokens": max_tokens},
    }
    try:
        response = requests.post(
            GEMINI_INTERACTIONS_URL,
            headers={"Content-Type": "application/json", "x-goog-api-key": api_key, "Api-Revision": "2026-05-20"},
            json=payload,
            timeout=(TRANSLATION_CONNECT_TIMEOUT_SECONDS, TRANSLATION_READ_TIMEOUT_SECONDS),
            allow_redirects=False,
        )
    except requests.RequestException as exc:
        raise RuntimeError("Gemini vision request failed") from exc
    if not response.ok:
        raise RuntimeError(f"Gemini HTTP {response.status_code}: {gemini_safe_error(response)}")
    try:
        body = response.json()
        answer = gemini_output_text(body)
    except (TypeError, KeyError, ValueError) as exc:
        raise RuntimeError("Gemini returned no text") from exc
    usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
    return VisionJSONResult(parse_json_object(answer), usage, None)
