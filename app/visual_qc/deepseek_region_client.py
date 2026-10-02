"""Response helpers shared by the OpenAI-compatible vision clients."""
from __future__ import annotations

import os

import requests

DEFAULT_DEEPSEEK_MODEL = os.getenv(
    "DEEPSEEK_VISUAL_QC_MODEL",
    "deepseek-v4-flash-vision-exp",
)


def _redact_secret(text: object, secret: str) -> str:
    value = str(text)
    return value.replace(secret, "[redacted]") if secret else value


def _safe_error_detail(response: requests.Response, secret: str) -> str:
    detail = ""
    try:
        payload = response.json()
        if isinstance(payload, dict):
            err = payload.get("error")
            if isinstance(err, dict) and err.get("message"):
                detail = str(err["message"])
            elif payload.get("message"):
                detail = str(payload["message"])
    except ValueError:
        pass
    if not detail:
        detail = (response.text or "").strip() or "unknown error"
    return _redact_secret(detail, secret)[:500]


def _extract_output_text(body: dict) -> str:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("No completion choice found")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise ValueError("Invalid completion choice")
    if choice.get("finish_reason") == "length":
        raise ValueError("Structured response was truncated")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("No completion message found")
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        parts = [
            str(item.get("text"))
            for item in content
            if isinstance(item, dict) and item.get("text")
        ]
        text = "".join(parts).strip()
        if text:
            return text
    raise ValueError("No text output found")
