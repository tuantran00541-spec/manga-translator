"""One chat call with native tool calls to an OpenAI-compatible provider."""
from __future__ import annotations

import json
import time
import uuid

import requests

from app.ai_providers import AIProvider, auth_headers
from app.parameters import TRANSLATION_CONNECT_TIMEOUT_SECONDS
from app.security import validate_url

READ_TIMEOUT = 600
RETRIES = 4
MAX_WAIT_S = 60


class ProviderError(RuntimeError):
    """The provider refused the call or kept failing after the retries."""


def chat_url(provider: AIProvider) -> str:
    if provider.protocol == "gemini":
        # Gemini serves the same chat format under its OpenAI-compatible path.
        return f"{provider.api_base.rstrip('/')}/openai/chat/completions"
    return str(provider.chat_url)


def _wait_s(response: requests.Response | None, attempt: int) -> float:
    """The wait a provider asked for in Retry-After, else a doubling backoff."""
    try:
        return min(MAX_WAIT_S, float(response.headers.get("Retry-After")))
    except (AttributeError, TypeError, ValueError):
        return min(MAX_WAIT_S, 2.0 ** (attempt + 1))


def _calls(message: dict) -> list[dict]:
    """The tool calls of a reply, each with its arguments parsed or an error saying why not."""
    calls = []
    for raw in message.get("tool_calls") or []:
        function = raw.get("function") or {}
        call = {"id": raw.get("id") or uuid.uuid4().hex[:12], "name": function.get("name") or "", "args": {}}
        try:
            args = json.loads(function.get("arguments") or "{}")
            call["args"] = args if isinstance(args, dict) else {}
        except ValueError:
            call["error"] = "The arguments were not valid JSON; send the call again with one JSON object."
        calls.append(call)
    return calls


def chat(provider: AIProvider, model: str, api_key: str, messages: list[dict], tools: list[dict], effort: str = "") -> dict:
    """One reply as {content, calls, usage}; retries timeouts, rate limits and 5xx."""
    url = validate_url(chat_url(provider))
    body = {"model": model, "messages": messages, "tools": [{"type": "function", "function": t} for t in tools]}
    if effort:
        body["reasoning_effort"] = effort
    headers = {**auth_headers(provider, api_key), "Content-Type": "application/json"}
    for attempt in range(RETRIES + 1):
        response = None
        try:
            response = requests.post(url, headers=headers, json=body, timeout=(TRANSLATION_CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT))
        except (requests.Timeout, requests.ConnectionError) as exc:
            if attempt == RETRIES:
                raise ProviderError(f"{provider.label} did not answer: {exc.__class__.__name__}") from exc
        if response is not None:
            if response.status_code == 429 or response.status_code >= 500:
                if attempt == RETRIES:
                    raise ProviderError(f"{provider.label} HTTP {response.status_code}: {response.text[:300]}")
            elif response.status_code >= 400:
                raise ProviderError(f"{provider.label} HTTP {response.status_code}: {response.text[:300]}")
            else:
                try:
                    data = response.json()
                    message = data["choices"][0]["message"]
                except (ValueError, KeyError, IndexError, TypeError) as exc:
                    raise ProviderError(f"{provider.label} sent a reply without a message: {response.text[:300]}") from exc
                return {"content": message.get("content") or "", "calls": _calls(message), "usage": data.get("usage") or {}}
        time.sleep(_wait_s(response, attempt))
    raise ProviderError(f"{provider.label} kept failing")
