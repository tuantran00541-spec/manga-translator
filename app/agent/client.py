"""One chat call to any configured provider, with native tool calls or a plain-text fallback."""
from __future__ import annotations

import json
import re
import time
import uuid

import requests

from app.ai_providers import AIProvider
from app.parameters import TRANSLATION_CONNECT_TIMEOUT_SECONDS
from app.security import validate_url
from app.visual_qc.deepseek_region_client import _safe_error_detail

READ_TIMEOUT = 600
RATE_LIMIT_RETRIES = 4
# When a provider says "slow down", every session using it waits, not just the one that was told.
_COOLDOWN: dict[str, float] = {}
# A block may lack its closing tag when the model stops early or opens the next call.
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|(?=<tool_call>)|\Z)", re.S)
PARAMETER_RE = re.compile(r"<parameter=(\w+)>\s*(.*?)\s*</parameter>", re.S)
TEXT_TOOLS_GUIDE = """You call tools by writing, anywhere in your reply, one block per call:
<tool_call>{"name": "read_file", "arguments": {"path": "app/main.py"}}</tool_call>
Results come back in <tool_result> blocks. Stop after your tool calls and wait for the results. Tools:
"""


class ToolsUnsupported(RuntimeError):
    """The provider refused the native tools field, so the session falls back to text tool calls."""


def chat_url(provider: AIProvider) -> str:
    if provider.protocol == "gemini":
        # Gemini serves the same chat format under its OpenAI-compatible path.
        return f"{provider.api_base.rstrip('/')}/openai/chat/completions"
    return str(provider.chat_url)


def native_tools(specs: list[dict]) -> list[dict]:
    return [{"type": "function", "function": spec} for spec in specs]


def text_tools_prompt(specs: list[dict]) -> str:
    return TEXT_TOOLS_GUIDE + "\n".join(
        f"- {s['name']}: {s['description']} Arguments: {json.dumps(s['parameters'].get('properties', {}))}"
        for s in specs
    )


def _read_call(raw: str) -> tuple[str, dict] | None:
    """A call from JSON, from Qwen's <function=...><parameter=...> form, or from the last JSON object in a mangled block."""
    try:
        data = json.loads(raw, strict=False)
        args = data.get("arguments", data.get("args", {}))
        return str(data.get("name") or ""), args if isinstance(args, dict) else {}
    except (ValueError, AttributeError):
        pass
    named = re.search(r"function=(\w+)", raw)
    if named:
        return named.group(1), {k: v for k, v in PARAMETER_RE.findall(raw)}
    for start in reversed([m.start() for m in re.finditer(r"\{", raw)]):
        try:
            data, _ = json.JSONDecoder(strict=False).raw_decode(raw[start:])
        except ValueError:
            continue
        if isinstance(data, dict) and data.get("name"):
            args = data.get("arguments", data.get("args", {}))
            return str(data["name"]), args if isinstance(args, dict) else {}
    return None


def parse_text_calls(text: str) -> tuple[str, list[dict]]:
    """Tool calls written as <tool_call> blocks, and the reply with those blocks taken out."""
    calls = []
    for raw in TOOL_CALL_RE.findall(text or ""):
        found = _read_call(raw)
        if found is None:
            calls.append({"id": uuid.uuid4().hex[:12], "name": "", "args": {}, "error": f"Unreadable tool call: {raw[:200]}. Write one valid JSON object: escape newlines as \\n and quotes as \\\", and keep arguments short."})
        else:
            calls.append({"id": uuid.uuid4().hex[:12], "name": found[0], "args": found[1]})
    return TOOL_CALL_RE.sub("", text or "").strip(), calls


def render(history: list[dict], system: str, text_mode: bool, specs: list[dict], *, reasoning: bool = False) -> list[dict]:
    """The neutral history as chat messages, with tool calls native or written as text."""
    messages = [{"role": "system", "content": system + ("\n\n" + text_tools_prompt(specs) if text_mode and specs else "")}]
    last_user = max((i for i, item in enumerate(history) if item["role"] == "user"), default=-1)
    for index, item in enumerate(history):
        role = item["role"]
        if role == "user":
            messages.append({"role": "user", "content": item["content"]})
        elif role == "assistant":
            calls = item.get("calls") or []
            if text_mode:
                blocks = "".join(f"\n<tool_call>{json.dumps({'name': c['name'], 'arguments': c['args']})}</tool_call>" for c in calls)
                messages.append({"role": "assistant", "content": (item.get("content") or "") + blocks})
            else:
                message = {"role": "assistant", "content": item.get("content") or ""}
                # DeepSeek's thinking mode needs the reasoning of the current turn's tool steps sent back.
                if reasoning and item.get("reasoning") and index > last_user:
                    message["reasoning_content"] = item["reasoning"]
                if calls:
                    message["tool_calls"] = [{"id": c["id"], "type": "function",
                                              "function": {"name": c["name"], "arguments": json.dumps(c["args"])}} for c in calls]
                messages.append(message)
        elif role == "tool":
            if not text_mode:
                messages.append({"role": "tool", "tool_call_id": item["id"], "content": item["content"]})
                continue
            block = f"<tool_result name=\"{item['name']}\">\n{item['content']}\n</tool_result>"
            if messages[-1]["role"] == "user" and messages[-1]["content"].startswith("<tool_result"):
                messages[-1]["content"] += "\n" + block
            else:
                messages.append({"role": "user", "content": block})
    return messages


def usage_of(raw: dict | None) -> dict:
    """Token counts from any OpenAI-style usage block, including how much of the prompt came from the provider's cache."""
    raw = raw or {}
    cached = ((raw.get("prompt_tokens_details") or {}).get("cached_tokens") or raw.get("prompt_cache_hit_tokens")
              or raw.get("cache_read_input_tokens") or 0)
    return {"prompt_tokens": int(raw.get("prompt_tokens") or 0), "completion_tokens": int(raw.get("completion_tokens") or 0),
            "cached_tokens": int(cached or 0)}


def _post(provider: AIProvider, api_key: str, payload: dict, stream: bool) -> requests.Response:
    """The request, with the shared 429 cooldown and retries."""
    url = chat_url(provider)
    validate_url(url)
    for attempt in range(RATE_LIMIT_RETRIES + 1):
        pause = _COOLDOWN.get(provider.id, 0.0) - time.time() if attempt == 0 else 0
        if pause > 0:
            time.sleep(min(pause, 60.0))
        try:
            response = requests.post(url, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                                     json=payload, timeout=(TRANSLATION_CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT),
                                     allow_redirects=False, stream=stream)
        except requests.RequestException as exc:
            raise RuntimeError(f"{provider.label} request failed: {type(exc).__name__}") from exc
        if response.status_code != 429 or attempt == RATE_LIMIT_RETRIES:
            return response
        # Free tiers allow a few requests a minute; wait as told, or longer each time.
        try:
            wait = float(response.headers.get("Retry-After") or 0)
        except ValueError:
            wait = 0.0
        wait = min(60.0, wait or 6.0 * 2 ** attempt)
        _COOLDOWN[provider.id] = max(_COOLDOWN.get(provider.id, 0.0), time.time() + wait)
        response.close()
        time.sleep(wait)
    return response


def _read_stream(response: requests.Response, on_delta) -> tuple[dict, dict, bool]:
    """Assemble the streamed message; on_delta(live) is called as it grows and returns True to stop early."""
    text, reasoning, usage, calls, stopped = "", "", {}, {}, False
    for raw in response.iter_lines(decode_unicode=True):
        if not raw or not raw.startswith("data:"):
            continue
        data = raw[5:].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except ValueError:
            continue
        usage = chunk.get("usage") or usage
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            text += delta.get("content") or ""
            reasoning += delta.get("reasoning_content") or delta.get("reasoning") or ""
            for part in delta.get("tool_calls") or []:
                slot = calls.setdefault(part.get("index", len(calls)), {"id": "", "name": "", "arguments": ""})
                slot["id"] = part.get("id") or slot["id"]
                function = part.get("function") or {}
                slot["name"] += function.get("name") or ""
                slot["arguments"] += function.get("arguments") or ""
        if on_delta({"text": text, "reasoning": reasoning, "tools": [c["name"] for c in calls.values() if c["name"]]}):
            stopped = True
            break
    message = {"content": text, "reasoning_content": reasoning,
               "tool_calls": [{"id": c["id"], "function": {"name": c["name"], "arguments": c["arguments"]}} for _, c in sorted(calls.items())]}
    return message, usage, stopped


def _build(message: dict, usage: dict) -> dict:
    calls = []
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        try:
            args = json.loads(function.get("arguments") or "{}", strict=False)
        except ValueError:
            args = None
        row = {"id": str(call.get("id") or uuid.uuid4().hex[:12]), "name": str(function.get("name") or ""),
               "args": args if isinstance(args, dict) else {}}
        if not isinstance(args, dict):
            row["error"] = "Tool arguments were not a JSON object"
        calls.append(row)
    text = message.get("content") or ""
    if isinstance(text, list):
        text = "".join(part.get("text", "") for part in text if isinstance(part, dict))
    # Models without native tools, and some with them, write their calls into the text.
    if "<tool_call>" in text:
        text, written = parse_text_calls(text)
        if not calls:
            calls = written
        else:
            # A native call that arrived with empty arguments takes them from the same call written in the text.
            spare = [w for w in written if w["args"] and not w.get("error")]
            for call in calls:
                match = next((w for w in spare if w["name"] == call["name"]), None)
                if match and not call["args"]:
                    call["args"] = match["args"]
                    call.pop("error", None)
                    spare.remove(match)
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    return {"text": str(text).strip(), "calls": calls, "reasoning": str(reasoning), "usage": usage_of(usage)}


def complete(provider: AIProvider, api_key: str, model: str, messages: list[dict], *, tools: list[dict] | None, on_delta=None) -> dict:
    """One assistant turn: its text, tool calls, reasoning and token usage; streamed when on_delta is given."""
    payload = {"model": model, "messages": messages, "stream": bool(on_delta)}
    payload.update(provider.chat_completion_extras())
    if tools:
        payload["tools"] = native_tools(tools)
    if on_delta:
        payload["stream_options"] = {"include_usage": True}
    response = _post(provider, api_key, payload, bool(on_delta))
    if on_delta and response.status_code == 400 and "stream_options" in response.text.lower():
        payload.pop("stream_options")
        response.close()
        response = _post(provider, api_key, payload, True)
    if 300 <= response.status_code < 400:
        raise RuntimeError(f"{provider.label} redirected the request")
    if not response.ok:
        detail = _safe_error_detail(response, api_key)
        if tools and 400 <= response.status_code < 500 and "tool" in detail.lower():
            raise ToolsUnsupported(detail)
        raise RuntimeError(f"{provider.label} HTTP {response.status_code}: {detail}")
    if on_delta:
        try:
            message, usage, _ = _read_stream(response, on_delta)
        except requests.RequestException as exc:
            raise RuntimeError(f"{provider.label} stream broke: {type(exc).__name__}") from exc
        finally:
            response.close()
        return _build(message, usage)
    try:
        body = response.json()
        message = body["choices"][0]["message"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"{provider.label} returned no message") from exc
    return _build(message, body.get("usage") or {})
