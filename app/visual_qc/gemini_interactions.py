from __future__ import annotations

import os

import requests

GEMINI_INTERACTIONS_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_VISUAL_QC_MODEL", "gemini-3.7-flash")
DEFAULT_CONNECT_TIMEOUT_SECONDS = 10
try:
    DEFAULT_TIMEOUT_SECONDS = max(15, min(300, int(os.getenv("GEMINI_VISUAL_QC_TIMEOUT_SECONDS", "120"))))
except ValueError:
    DEFAULT_TIMEOUT_SECONDS = 120


def extract_output_text(body: dict) -> str:
    for step in reversed(body.get("steps") or []):
        if step.get("type") != "model_output":
            continue
        for content in step.get("content") or []:
            if content.get("type") == "text" and isinstance(content.get("text"), str):
                return content["text"]
    raise ValueError("No text model output found")


def safe_error_detail(response: requests.Response) -> str:
    try:
        payload = response.json()
        if isinstance(payload, dict):
            err = payload.get("error")
            if isinstance(err, dict) and err.get("message"):
                return str(err["message"])[:500]
            if payload.get("message"):
                return str(payload["message"])[:500]
    except ValueError:
        pass
    text = (response.text or "").strip()
    return text[:500] if text else "unknown error"
