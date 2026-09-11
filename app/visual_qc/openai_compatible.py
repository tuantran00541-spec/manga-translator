from __future__ import annotations

import json
from pathlib import Path

import requests

from app.visual_qc.deepseek_region_client import _extract_output_text, _safe_error_detail
from app.visual_qc.gemini import (
    DEFAULT_CONNECT_TIMEOUT_SECONDS,
    DEFAULT_TIMEOUT_SECONDS,
    _PROMPT,
    _encode_for_gemini,
    _parse_issues,
    _read_image,
)


class OpenAICompatibleVisualQC:
    def __init__(
        self,
        *,
        provider_label: str,
        chat_url: str,
        model: str,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    ):
        self.provider_label = provider_label
        self.chat_url = chat_url
        self.model = model
        self.timeout_seconds = timeout_seconds

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
        try:
            response = requests.post(
                self.chat_url,
                headers={"Authorization": f"Bearer {secret}", "Content-Type": "application/json"},
                json=payload,
                timeout=(DEFAULT_CONNECT_TIMEOUT_SECONDS, self.timeout_seconds),
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
