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


def render(history: list[dict], system: str, text_mode: bool, specs: list[dict]) -> list[dict]:
    """The neutral history as chat messages, with tool calls native or written as text."""
    messages = [{"role": "system", "content": system + ("\n\n" + text_tools_prompt(specs) if text_mode and specs else "")}]
    for item in history:
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


def complete(provider: AIProvider, api_key: str, model: str, messages: list[dict], *, tools: list[dict] | None) -> dict:
    """One assistant turn: its text, tool calls, reasoning and token usage."""
    url = chat_url(provider)
    validate_url(url)
    payload = {"model": model, "messages": messages, "stream": False}
    payload.update(provider.chat_completion_extras())
    if tools:
        payload["tools"] = native_tools(tools)
    for attempt in range(RATE_LIMIT_RETRIES + 1):
        try:
            response = requests.post(url, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                                     json=payload, timeout=(TRANSLATION_CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT),
                                     allow_redirects=False)
        except requests.RequestException as exc:
            raise RuntimeError(f"{provider.label} request failed: {type(exc).__name__}") from exc
        if response.status_code != 429 or attempt == RATE_LIMIT_RETRIES:
            break
        # Free tiers allow a few requests a minute; wait as told, or longer each time.
        try:
            wait = float(response.headers.get("Retry-After") or 0)
        except ValueError:
            wait = 0.0
        time.sleep(min(60.0, wait or 6.0 * 2 ** attempt))
    if 300 <= response.status_code < 400:
        raise RuntimeError(f"{provider.label} redirected the request")
    if not response.ok:
        detail = _safe_error_detail(response, api_key)
        if tools and 400 <= response.status_code < 500 and "tool" in detail.lower():
            raise ToolsUnsupported(detail)
        raise RuntimeError(f"{provider.label} HTTP {response.status_code}: {detail}")
    try:
        body = response.json()
        message = body["choices"][0]["message"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"{provider.label} returned no message") from exc
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
    return {"text": str(text).strip(), "calls": calls, "reasoning": str(reasoning), "usage": body.get("usage") or {}}
