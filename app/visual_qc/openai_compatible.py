from __future__ import annotations

import json
from pathlib import Path

import requests

from app.ai_providers import AIProvider, PROVIDERS, get_provider
from app.security import validate_url
from app.visual_qc.deepseek_region_client import _extract_output_text, _safe_error_detail
from app.visual_qc.gemini import (
    DEFAULT_CONNECT_TIMEOUT_SECONDS,
    DEFAULT_TIMEOUT_SECONDS,
    _PROMPT,
    _encode_for_gemini,
    _parse_issues,
    _read_image,
)


def _resolve_provider_id(provider_id: str | None, chat_url: str | None) -> str:
    if provider_id:
        return provider_id
    normalized = (chat_url or "").rstrip("/")
    for provider in PROVIDERS.values():
        if provider.chat_url and provider.chat_url.rstrip("/") == normalized:
            return provider.id
    raise ValueError("OpenAI-compatible visual QC provider could not be resolved")


class OpenAICompatibleVisualQC:
    def __init__(
        self,
        *,
        provider_id: str | None = None,
        provider_label: str | None = None,
        chat_url: str | None = None,
        model: str,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        provider: AIProvider | None = None,
    ):
        provider = provider or get_provider(
            _resolve_provider_id(provider_id, chat_url)
        )
        if provider.protocol != "openai" or not provider.supports_visual_qc:
            raise ValueError(f"{provider.label} is not available for OpenAI-compatible visual QC")
        if not provider.chat_url:
            raise ValueError(f"{provider.label} does not expose a chat-completions endpoint")
        if chat_url and chat_url.rstrip("/") != provider.chat_url.rstrip("/"):
            raise ValueError(f"{provider.label} visual QC endpoint does not match the provider contract")
        self.provider_id = provider.id
        self.provider_label = provider_label or provider.label
        self.chat_url = provider.chat_url
        self.model = model
        self.timeout_seconds = timeout_seconds
        self._request_extras = provider.chat_completion_extras()
        self._validate_remote = not provider.builtin

    def inspect(self, original_path: Path, cleaned_path: Path, api_key: str):
        secret = (api_key or "").strip()
        if not secret:
            raise ValueError(f"{self.provider_label} API key is not configured")
        original = _read_image(original_path)
        cleaned = _read_image(cleaned_path)
        if original.shape[:2] != cleaned.shape[:2]:
            raise ValueError("Original and cleaned images must have the same dimensions")
        original_b64 = _encode_for_gemini(original)
        cleaned_b64 = _encode_for_gemini(cleaned)
        payload = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": _PROMPT + "\nReturn JSON only."},
                    {"type": "text", "text": "ORIGINAL image:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{original_b64}"}},
                    {"type": "text", "text": "CLEANED image to inspect:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{cleaned_b64}"}},
                ],
            }],
            "response_format": {"type": "json_object"},
            "max_tokens": 1200,
            "stream": False,
        }
        payload.update(self._request_extras)
        try:
            if self._validate_remote:
                validate_url(self.chat_url)
            response = requests.post(
                self.chat_url,
                headers={"Authorization": f"Bearer {secret}", "Content-Type": "application/json"},
                json=payload,
                timeout=(DEFAULT_CONNECT_TIMEOUT_SECONDS, self.timeout_seconds),
                allow_redirects=False,
            )
            if 300 <= response.status_code < 400:
                raise RuntimeError(
                    f"{self.provider_label} API redirect was refused"
                )
        except requests.Timeout as exc:
            raise RuntimeError(f"{self.provider_label} visual QC timed out") from exc
        except requests.RequestException as exc:
            raise RuntimeError(f"{self.provider_label} request failed") from exc
        if not response.ok:
            raise RuntimeError(
                f"{self.provider_label} API returned HTTP {response.status_code}: "
                f"{_safe_error_detail(response, secret)}"
            )
        try:
            body = response.json()
            parsed = json.loads(_extract_output_text(body))
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"{self.provider_label} returned an invalid structured response") from exc
        h, w = cleaned.shape[:2]
        return _parse_issues(parsed, w, h)
