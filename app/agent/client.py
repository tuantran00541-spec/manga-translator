from __future__ import annotations

import hashlib
import json
import random
import re
import threading
import time
import uuid
from contextvars import ContextVar
from email.utils import parsedate_to_datetime

import requests
from fastapi import HTTPException

from app import api_proxy
from app.ai_providers import AIProvider, auth_headers
from app.downloader.http import public_request
from app.logging_config import logger
from app.parameters import TRANSLATION_CONNECT_TIMEOUT_SECONDS
from app.security import validate_url
from app.visual_qc.deepseek_region_client import _safe_error_detail

READ_TIMEOUT = 600
STREAM_IDLE_TIMEOUT = 120
NETWORK_RETRIES = 10
OVERFLOW_PATTERNS = [re.compile(p, re.I) for p in (
    r"prompt (?:is )?too long", r"prompt exceeds max length", r"request_too_large", r"input is too long for requested model", r"exceeds the context window",
    r"exceeds (?:the )?(?:model'?s )?maximum context length", r"input token count.*exceeds the maximum", r"maximum prompt length is \d+",
    r"reduce the length of the messages", r"maximum context length is \d+ tokens", r"exceeds (?:the )?maximum allowed input length", r"is longer than the model'?s context length",
    r"exceeds the limit of \d+", r"exceeds the available context size", r"greater than the context length", r"context window exceeds limit", r"exceeded model token limit",
    r"too large for model with \d+ maximum context length", r"configured context size is", r"model_context_window_exceeded", r"prompt too long; exceeded",
    r"range of input length should be", r"context[_ ]length[_ ]exceeded", r"too many tokens", r"token limit exceeded", r"context.{0,20}(?:length|window)", r"maximum context")]
NOT_OVERFLOW = re.compile(r"rate limit|too many requests|throttl", re.I)
WINDOW_PATTERNS = [re.compile(p, re.I) for p in (
    r"maximum context length (?:is|of)? ?\(?([\d,]{4,})", r"maximum prompt length is ([\d,]{4,})", r"context (?:window|length|size|limit) (?:of|is) \(?([\d,]{4,})",
    r"context (?:length|size) \(([\d,]{4,})", r"maximum number of tokens allowed \(([\d,]{4,})", r"maximum allowed input length of ([\d,]{4,})",
    r"limit of ([\d,]{4,})", r"> ([\d,]{4,}) maximum", r"configured context size is ([\d,]{4,})")]
RATE_LIMIT_RETRIES = 4
_CLIENT_LOCK = threading.Lock()
_MAX_CLIENT_ENTRIES = 256
_COOLDOWN: dict[str, float] = {}
_PACE: dict[str, list[float]] = {}
PACE_HOLD_S = 900
PACE_RE = re.compile(r"(\d+) requests? (?:every|per|each) (\d+)? ?(second|sec|minute|min|hour)", re.I)
AFTER_RE = re.compile(r"try again (?:after|at) (\d{1,2}):(\d{2})(?::(\d{2}))?", re.I)
IN_RE = re.compile(r"(?:try again|retry) in (\d+(?:\.\d+)?) ?(ms|milliseconds?|s|secs?|seconds?|m|mins?|minutes?)\b", re.I)
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|(?=<tool_call>)|\Z)", re.S)
PARAMETER_RE = re.compile(r"<parameter=(\w+)>\s*(.*?)\s*</parameter>", re.S)
INVOKE_RE = re.compile(r"<invoke\b[^>]*?\bname\s*=\s*[\"']?([\w.-]+)[\"']?[^>]*>(.*?)(?:</invoke>|\Z)", re.S | re.I)
INVOKE_PARAM_RE = re.compile(r"<parameter\b[^>]*?\bname\s*=\s*[\"']?(\w+)[\"']?[^>]*>(.*?)</parameter>", re.S | re.I)
INVOKE_WRAP_RE = re.compile(r"</?(?:[\w-]+:)?(?:function_calls|tool_calls?)>", re.I)
TEXT_TOOLS_GUIDE = """To call a tool, write one block per call anywhere in your reply:
<tool_call>{"name": "read_file", "arguments": {"path": "app/main.py"}}</tool_call>
The results come back in <tool_result> blocks; end your reply after the calls and wait for them. If a call is rejected or a tool reports an error, read the error, fix the call once, and if it fails again say what failed instead of repeating the same call. Tools:
"""


CUT_OFF = ("Your reply reached the output limit and this call was cut off before its end. Send it in smaller parts: start the file "
           "small, then extend it with edit_file or a shell append (cat >> file <<'EOF').")


class TransientError(RuntimeError):
    pass


class KeyLimited(RuntimeError):
    pass


class CancelledError(RuntimeError):
    pass


ASK_ON_LIMIT: ContextVar[bool] = ContextVar("ask_on_limit", default=False)
REQUEST_EXTRA: ContextVar[dict] = ContextVar("request_extra", default={})
CANCEL_EVENT: ContextVar[threading.Event | None] = ContextVar("cancel_event", default=None)
CANCEL_HOOKS: ContextVar[list | None] = ContextVar("cancel_hooks", default=None)
_REFUSED_EXTRA: set[tuple[str, str]] = set()


def _is_cancelled() -> bool:
    ev = CANCEL_EVENT.get()
    return ev is not None and ev.is_set()


def _check_cancelled() -> None:
    if _is_cancelled():
        raise CancelledError("Đã dừng.")


def _sleep_interruptible(seconds: float) -> None:
    if seconds <= 0:
        return
    ev = CANCEL_EVENT.get()
    if ev is None:
        time.sleep(seconds)
        return
    if ev.wait(seconds):
        raise CancelledError("Đã dừng.")


def _force_close_response(resp: requests.Response | None) -> None:
    if resp is None:
        return
    try:
        raw = getattr(resp, "raw", None)
        fp = getattr(raw, "_fp", None)
        raw_sock = getattr(fp, "raw", None)
        sock = getattr(raw_sock, "_sock", None) or getattr(fp, "fp", None)
        for obj in (sock, raw_sock, fp, raw):
            if obj is None:
                continue
            try:
                if hasattr(obj, "shutdown"):
                    obj.shutdown(2)
            except Exception:
                pass
            try:
                obj.close()
            except Exception:
                pass
    except Exception:
        pass
    try:
        resp.close()
    except Exception:
        pass


def _register_cancel_hook(fn) -> None:
    hooks = CANCEL_HOOKS.get()
    if hooks is not None:
        hooks.append(fn)


def _unregister_cancel_hook(fn) -> None:
    hooks = CANCEL_HOOKS.get()
    if hooks is not None:
        try:
            hooks.remove(fn)
        except ValueError:
            pass


class ToolsUnsupported(RuntimeError):
    pass


def chat_url(provider: AIProvider) -> str:
    if provider.protocol == "gemini":
        return f"{provider.api_base.rstrip('/')}/openai/chat/completions"
    with _CLIENT_LOCK:
        return _FOUND_PATH.get(provider.api_base.rstrip("/"), str(provider.chat_url))


_FOUND_PATH: dict[str, str] = {}


def chat_candidates(provider: AIProvider) -> list[str]:
    base = provider.api_base.rstrip("/")
    root = re.sub(r"/v\d+[a-z0-9]*$", "", base)
    tails = [f"{base}/chat/completions", f"{base}/v1/chat/completions", f"{root}/v1/chat/completions", f"{root}/chat/completions",
             f"{root}/api/v1/chat/completions", f"{root}/openai/v1/chat/completions", f"{root}/v1beta/openai/chat/completions",
             f"{root}/compatible-mode/v1/chat/completions"]
    return list(dict.fromkeys(tails))


def native_tools(specs: list[dict]) -> list[dict]:
    return [{"type": "function", "function": spec} for spec in specs]


def text_tools_prompt(specs: list[dict]) -> str:
    return TEXT_TOOLS_GUIDE + "\n".join(
        f"- {s['name']}: {s['description']} Arguments: {json.dumps((s.get('parameters') or {}).get('properties', {}))}"
        for s in specs
    )


UNESCAPES = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "/": "/"}


def _lenient_args(raw: str, params: list[str], required: list[str]) -> dict | None:
    start = raw.find('"arguments"')
    body = raw[raw.find("{", start) + 1:] if start >= 0 else raw
    body = re.sub(r'\s*\}\s*\}?\s*$', "", body.rstrip())
    marks = []
    for name in params:
        found = re.search(r'(?:^|[,{])\s*"' + re.escape(name) + r'"\s*:\s*', body)
        if found:
            marks.append((found.start(), found.end(), name))
    marks.sort()
    args: dict = {}
    for index, (_, end, name) in enumerate(marks):
        value = body[end:marks[index + 1][0] if index + 1 < len(marks) else len(body)].strip()
        value = re.sub(r"\s*,\s*$", "", value)
        try:
            args[name] = json.loads(value, strict=False)
            continue
        except ValueError:
            pass
        if len(value) > 1 and value[0] == '"' and value[-1] == '"':
            value = value[1:-1]
        args[name] = re.sub(r"\\(.)", lambda m: UNESCAPES.get(m.group(1), m.group(0)), value)
    return args if args and all(name in args for name in required) else None


def _read_call(raw: str, schemas: dict | None = None) -> tuple[str, dict] | None:
    try:
        data = json.loads(raw, strict=False)
        args = data.get("arguments", data.get("args", {}))
        return str(data.get("name") or ""), args if isinstance(args, dict) else {}
    except (ValueError, AttributeError):
        pass
    named = re.search(r"function=(\w+)", raw)
    if named:
        return named.group(1), {k: v for k, v in PARAMETER_RE.findall(raw)}
    called = re.search(r'"name"\s*:\s*"(\w+)"', raw)
    if called and schemas and called.group(1) in schemas:
        args = _lenient_args(raw, *schemas[called.group(1)])
        if args is not None:
            return called.group(1), args
    for start in reversed([m.start() for m in re.finditer(r"\{", raw)]):
        try:
            data, _ = json.JSONDecoder(strict=False).raw_decode(raw[start:])
        except ValueError:
            continue
        if isinstance(data, dict) and data.get("name"):
            args = data.get("arguments", data.get("args", {}))
            return str(data["name"]), args if isinstance(args, dict) else {}
    return None


def schemas_of(tools: list[dict] | None) -> dict:
    return {t["name"]: (list((t.get("parameters") or {}).get("properties") or {}), list((t.get("parameters") or {}).get("required") or []))
            for t in tools or []}


def parse_text_calls(text: str, tools: list[dict] | None = None) -> tuple[str, list[dict]]:
    calls = []
    schemas = schemas_of(tools)
    for raw in TOOL_CALL_RE.findall(text or ""):
        if INVOKE_RE.search(raw):
            calls += _invokes(raw)
            continue
        found = _read_call(raw, schemas)
        if found is None:
            calls.append({"id": uuid.uuid4().hex[:12], "name": "", "args": {}, "error": f"Unreadable tool call: {raw[:200]}{' … ' + raw[-150:] if len(raw) > 400 else ''}. Write one valid JSON object: escape newlines as \\n and quotes as \\\", and keep arguments short."})
        else:
            calls.append({"id": uuid.uuid4().hex[:12], "name": found[0], "args": found[1]})
    text = TOOL_CALL_RE.sub("", text or "")
    if INVOKE_RE.search(text):
        calls += _invokes(text)
        text = INVOKE_WRAP_RE.sub("", INVOKE_RE.sub("", text))
    return text.strip(), calls


def _invokes(raw: str) -> list[dict]:
    def value(text: str):
        text = text.strip()
        if text[:1] in ("{", "["):
            try:
                return json.loads(text, strict=False)
            except ValueError:
                pass
        return text
    return [{"id": uuid.uuid4().hex[:12], "name": name, "args": {k: value(v) for k, v in INVOKE_PARAM_RE.findall(body)}}
            for name, body in INVOKE_RE.findall(raw)]


def render(history: list[dict], system: str, text_mode: bool, specs: list[dict], *, reasoning: bool | str = False) -> list[dict]:
    messages = [{"role": "system", "content": system + ("\n\n" + text_tools_prompt(specs) if text_mode and specs else "")}]
    last_user = max((i for i, item in enumerate(history) if item["role"] == "user"), default=-1)
    for index, item in enumerate(history):
        role = item["role"]
        if role == "user":
            images = item.get("images") or []
            if images and not text_mode:
                parts = [{"type": "text", "text": item["content"]}] + [{"type": "image_url", "image_url": {"url": url}} for url in images]
                messages.append({"role": "user", "content": parts})
            else:
                messages.append({"role": "user", "content": item["content"]})
        elif role == "assistant":
            calls = item.get("calls") or []
            if text_mode:
                blocks = "".join(f"\n<tool_call>{json.dumps({'name': c['name'], 'arguments': c['args']})}</tool_call>" for c in calls)
                messages.append({"role": "assistant", "content": (item.get("content") or "") + blocks})
            else:
                message = {"role": "assistant", "content": item.get("content") or ""}
                if reasoning == "all":
                    message["reasoning_content"] = item.get("reasoning") or ""
                elif reasoning and item.get("reasoning") and index > last_user:
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
    raw = raw or {}
    cached = ((raw.get("prompt_tokens_details") or {}).get("cached_tokens") or raw.get("prompt_cache_hit_tokens")
              or raw.get("cache_read_input_tokens") or 0)
    return {"prompt_tokens": int(raw.get("prompt_tokens") or 0), "completion_tokens": int(raw.get("completion_tokens") or 0),
            "cached_tokens": int(cached or 0)}


def is_overflow(message: str) -> bool:
    return any(p.search(message) for p in OVERFLOW_PATTERNS) and not NOT_OVERFLOW.search(message)


def context_window(message: str) -> int:
    for pattern in WINDOW_PATTERNS:
        found = pattern.search(message)
        if found:
            return int(found.group(1).replace(",", ""))
    return 0


def _retry_after(response: requests.Response) -> float:
    try:
        millis = response.headers.get("retry-after-ms")
        if millis:
            return float(millis) / 1000
        value = response.headers.get("Retry-After") or ""
        try:
            return float(value)
        except ValueError:
            return max(0.0, parsedate_to_datetime(value).timestamp() - time.time()) if value else 0.0
    except (TypeError, ValueError):
        return 0.0


def _hinted_wait(text: str) -> float:
    found = IN_RE.search(text)
    if found:
        unit = found.group(2).lower()
        return float(found.group(1)) * (0.001 if unit.startswith("ms") or unit.startswith("milli") else 60 if unit.startswith("m") else 1)
    found = AFTER_RE.search(text)
    if found:
        now = time.gmtime()
        target = int(found.group(1)) * 3600 + int(found.group(2)) * 60 + int(found.group(3) or 0)
        wait = (target - (now.tm_hour * 3600 + now.tm_min * 60 + now.tm_sec)) % 86400
        return float(wait) if wait <= 900 else 0.0
    return 0.0


def _pace_hint(text: str) -> float:
    found = PACE_RE.search(text)
    if not found or int(found.group(1)) == 0:
        return 0.0
    unit = found.group(3).lower()
    period = int(found.group(2) or 1) * (3600 if unit.startswith("h") else 60 if unit.startswith("m") else 1)
    return period / int(found.group(1))


def slot(provider: AIProvider, api_key: str) -> str:
    return f"{provider.id}:{hashlib.sha256((api_key or '').encode()).hexdigest()[:12]}"


def _bound_client_map(mapping: dict, key, limit: int = _MAX_CLIENT_ENTRIES) -> None:
    if key not in mapping:
        while len(mapping) >= limit:
            mapping.pop(next(iter(mapping)))


def _wait_for_pace(name: str) -> None:
    with _CLIENT_LOCK:
        pace = _PACE.get(name)
        if not pace:
            return
        interval, until, last = pace
        now = time.time()
        if now > until:
            _PACE.pop(name, None)
            return
        gap = max(0.0, last + interval - now)
        pace[2] = now + gap
    if gap > 0:
        _sleep_interruptible(gap)


RATE_LIMITS: dict[str, dict] = {}


def _note_rate_limit(provider: AIProvider, response: requests.Response, wait: float, api_key: str = "") -> None:
    try:
        raw_detail = response.text[:600]
    except Exception:
        raw_detail = ""
    detail = api_proxy.redact_secret(raw_detail, api_key)[:300]
    with _CLIENT_LOCK:
        _bound_client_map(RATE_LIMITS, provider.id)
        row = RATE_LIMITS.setdefault(provider.id, {"count": 0, "waited_s": 0.0, "detail": "", "first": time.time()})
        row["count"] += 1
        row["waited_s"] += wait
        row["detail"] = detail or row["detail"]
        row["last"] = time.time()
        count = row["count"]
    logger.info("{} answered 429 ({} so far, waiting {:.0f} s): {}", provider.label, count, wait, detail[:200])


def _post(provider: AIProvider, api_key: str, payload: dict, stream: bool) -> requests.Response:
    url0 = chat_url(provider)
    response = _post_to(url0, provider, api_key, payload, stream)
    base = provider.api_base.rstrip("/")
    with _CLIENT_LOCK:
        already_found = base in _FOUND_PATH
    if response.status_code not in (404, 405) or provider.builtin or provider.protocol != "openai" or already_found:
        return response
    tried = {url0}
    for url in chat_candidates(provider):
        if url in tried:
            continue
        tried.add(url)
        try:
            other = _post_to(url, provider, api_key, payload, stream)
        except (HTTPException, ValueError):
            continue
        if other.status_code not in (404, 405):
            response.close()
            with _CLIENT_LOCK:
                _bound_client_map(_FOUND_PATH, base)
                _FOUND_PATH[base] = url
            logger.info("{} answers chat at {}", provider.label, url)
            return other
        other.close()
    return response


def _cancellable_post(url: str, *, headers: dict, payload: dict, timeout: tuple[float, float], stream: bool) -> requests.Response:
    kwargs = dict(headers=headers, json=payload, timeout=timeout, allow_redirects=False, stream=stream)
    ev = CANCEL_EVENT.get()
    if ev is None:
        return public_request("POST", url, **kwargs)
    _check_cancelled()
    done = threading.Event()
    result: list[requests.Response | None] = [None]
    error: list[BaseException | None] = [None]

    def _run() -> None:
        try:
            resp = public_request("POST", url, **kwargs)
            if ev.is_set():
                _force_close_response(resp)
                return
            result[0] = resp
        except BaseException as exc:  # noqa: BLE001
            error[0] = exc
        finally:
            done.set()

    threading.Thread(target=_run, name="agent-http-post", daemon=True).start()
    while not done.wait(0.05):
        if ev.is_set():
            raise CancelledError("Đã dừng.")
    if ev.is_set():
        _force_close_response(result[0])
        raise CancelledError("Đã dừng.")
    if error[0] is not None:
        raise error[0]
    assert result[0] is not None
    return result[0]


def _post_to(url: str, provider: AIProvider, api_key: str, payload: dict, stream: bool) -> requests.Response:
    validate_url(url)
    key = slot(provider, api_key)
    for attempt in range(RATE_LIMIT_RETRIES + 1):
        _check_cancelled()
        with _CLIENT_LOCK:
            pause = _COOLDOWN.get(key, 0.0) - time.time() if attempt == 0 else 0
        if pause > 0:
            _sleep_interruptible(min(pause, 60.0))
        _wait_for_pace(key)
        _check_cancelled()
        try:
            response = _cancellable_post(
                url,
                headers={**auth_headers(provider, api_key), "Content-Type": "application/json"},
                payload=payload,
                timeout=(TRANSLATION_CONNECT_TIMEOUT_SECONDS, STREAM_IDLE_TIMEOUT if stream else READ_TIMEOUT),
                stream=stream,
            )
        except CancelledError:
            raise
        except requests.RequestException as exc:
            _check_cancelled()
            raise TransientError(f"{provider.label} request failed: {type(exc).__name__}") from exc
        if response.status_code != 429 or attempt == RATE_LIMIT_RETRIES:
            return response
        try:
            words = response.text[:500]
        except Exception:
            words = ""
        interval = _pace_hint(words)
        if ASK_ON_LIMIT.get() and not (interval or _retry_after(response) or _hinted_wait(words)):
            _note_rate_limit(provider, response, 0.0, api_key=api_key)
            return response
        if interval:
            with _CLIENT_LOCK:
                _bound_client_map(_PACE, key)
                _PACE[key] = [interval, time.time() + PACE_HOLD_S, time.time()]
                _bound_client_map(RATE_LIMITS, provider.id)
                RATE_LIMITS.setdefault(provider.id, {"count": 0, "waited_s": 0.0, "detail": "", "first": time.time()})["pace_s"] = interval
        wait = min(900.0 if interval else 60.0, _retry_after(response) or _hinted_wait(words) or interval or 6.0 * 2 ** attempt)
        _note_rate_limit(provider, response, wait, api_key=api_key)
        with _CLIENT_LOCK:
            _bound_client_map(_COOLDOWN, key)
            _COOLDOWN[key] = max(_COOLDOWN.get(key, 0.0), time.time() + wait)
        response.close()
        _sleep_interruptible(wait)
    return response


def _read_stream(response: requests.Response, on_delta) -> tuple[dict, dict, bool]:
    text, reasoning, usage, calls, stopped = "", "", {}, {}, False
    finish, other = "", {}
    response.encoding = "utf-8"
    cancel_hook = lambda: _force_close_response(response)
    _register_cancel_hook(cancel_hook)
    try:
        for raw in response.iter_lines(decode_unicode=True):
            if _is_cancelled():
                stopped = True
                break
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
                finish = choice.get("finish_reason") or finish
                for key, value in delta.items():
                    if key not in ("content", "reasoning_content", "reasoning", "tool_calls", "role") and isinstance(value, str) and value:
                        other[key] = other.get(key, "") + value
                text += delta.get("content") or ""
                reasoning += delta.get("reasoning_content") or delta.get("reasoning") or ""
                for part in delta.get("tool_calls") or []:
                    slot = calls.setdefault(part.get("index", len(calls)), {"id": "", "name": "", "arguments": ""})
                    slot["id"] = part.get("id") or slot["id"]
                    function = part.get("function") or {}
                    slot["name"] += function.get("name") or ""
                    slot["arguments"] += function.get("arguments") or ""
            arg_text = "".join(c["arguments"] for c in calls.values())
            if on_delta({"text": text, "reasoning": reasoning, "tools": [c["name"] for c in calls.values() if c["name"]],
                         "args": arg_text[-4000:], "arg_chars": len(arg_text)}):
                stopped = True
                break
    finally:
        _unregister_cancel_hook(cancel_hook)
    if other and not (text or reasoning or calls):
        reasoning = "".join(other.values())
    message = {"content": text, "reasoning_content": reasoning, "_debug": {"finish": finish, "fields": sorted(other)},
               "tool_calls": [{"id": c["id"], "function": {"name": c["name"], "arguments": c["arguments"]}} for _, c in sorted(calls.items())]}
    return message, usage, stopped


THINK = re.compile(r"<think>(.*?)</think>", re.S)


def split_thinking(text: str) -> tuple[str, str]:
    thought = "\n".join(part.strip() for part in THINK.findall(text))
    text = THINK.sub("", text)
    if "</think>" in text:
        before, _, text = text.partition("</think>")
        thought = (before.strip() + "\n" + thought).strip()
    return text.strip(), thought


def loads_args(raw: str):
    try:
        return json.loads(raw or "{}", strict=False)
    except ValueError:
        pass
    fixed = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", raw)
    fixed = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", fixed)
    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
    return json.loads(fixed, strict=False)


def _build(message: dict, usage: dict, tools: list[dict] | None = None) -> dict:
    calls = []
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        try:
            args = loads_args(function.get("arguments") or "{}")
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
    if "<tool_call>" in text or "<invoke" in text:
        text, written = parse_text_calls(text, tools)
        if not calls:
            calls = written
        else:
            spare = [w for w in written if w["args"] and not w.get("error")]
            for call in calls:
                match = next((w for w in spare if w["name"] == call["name"]), None)
                if match and not call["args"]:
                    call["args"] = match["args"]
                    call.pop("error", None)
                    spare.remove(match)
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    if "think>" in str(text):
        text, thought = split_thinking(str(text))
        reasoning = (str(reasoning) + "\n" + thought).strip()
    result = {"text": str(text).strip(), "calls": calls, "reasoning": str(reasoning), "usage": usage_of(usage)}
    finish = (message.get("_debug") or {}).get("finish")
    result["finish"] = finish or ""
    if finish == "length":
        for call in calls:
            if call.get("error"):
                call["error"] = CUT_OFF
    if not (result["text"] or calls) and message.get("_debug"):
        result["debug"] = message["_debug"]
    return result


OUTPUT_CAPS: dict[tuple[str, str], int] = {}


def output_cap(provider: AIProvider, model: str) -> int | None:
    with _CLIENT_LOCK:
        return OUTPUT_CAPS.get((provider.api_base, model))


def complete(provider: AIProvider, api_key: str, model: str, messages: list[dict], *, tools: list[dict] | None, on_delta=None,
             max_tokens: int | None = None) -> dict:
    for attempt in range(NETWORK_RETRIES + 1):
        _check_cancelled()
        try:
            return _complete_once(provider, api_key, model, messages, tools=tools, on_delta=on_delta, max_tokens=max_tokens)
        except TransientError as exc:
            if attempt == NETWORK_RETRIES:
                raise
            wait = min(90.0, 2.0 * 2 ** attempt) * random.uniform(0.75, 1.0)
            logger.warning("{} ({}); asking again in {:.0f}s", exc, provider.label, wait)
            _sleep_interruptible(wait)
    raise AssertionError("unreachable")


def _complete_once(provider: AIProvider, api_key: str, model: str, messages: list[dict], *, tools: list[dict] | None, on_delta=None,
                   max_tokens: int | None = None) -> dict:
    started = time.monotonic()
    stream_active = bool(on_delta)
    payload = {"model": model, "messages": messages, "stream": stream_active}
    payload.update(provider.chat_completion_extras())
    if tools:
        payload["tools"] = native_tools(tools)
    if stream_active:
        payload["stream_options"] = {"include_usage": True}
    cap = output_cap(provider, model)
    if max_tokens and cap != 0:
        payload["max_tokens"] = min(int(max_tokens), cap) if cap else int(max_tokens)
    refused = (provider.api_base, model)
    with _CLIENT_LOCK:
        is_refused = refused in _REFUSED_EXTRA
    extra = {} if is_refused else {k: v for k, v in REQUEST_EXTRA.get().items() if k not in ("model", "messages", "stream", "tools")}
    payload.update(extra)

    base_url_key = str(provider.api_base or "").rstrip("/")
    payload, stream_active, pre_wrapped, wrapped_chat = api_proxy.apply_wrapped_rules(
        payload,
        api_base=base_url_key,
        model=model,
        stream=stream_active,
        is_gemini=False,
        allow_image_strip=True,
    )
    if wrapped_chat and base_url_key:
        with _CLIENT_LOCK:
            _bound_client_map(_FOUND_PATH, base_url_key)
            _FOUND_PATH[base_url_key] = wrapped_chat

    errors_seen: list[dict] = []
    rescues_applied: list[str] = list(pre_wrapped)
    rescue_types: list[str] = ["auto_wrapped"] if pre_wrapped else []
    downsize_attempts = 0

    for attempt in range(1, 7):
        _check_cancelled()
        response = _post(provider, api_key, payload, stream_active)
        if extra and attempt < 6 and response.status_code in (400, 422) and any(k.lower() in response.text.lower() for k in extra):
            logger.warning("{} refused {}; asking without them from now on", provider.label, ", ".join(extra))
            with _CLIENT_LOCK:
                if len(_REFUSED_EXTRA) >= _MAX_CLIENT_ENTRIES:
                    _REFUSED_EXTRA.clear()
                _REFUSED_EXTRA.add(refused)
            removed_keys = list(extra)
            for key in removed_keys:
                payload.pop(key, None)
            extra = {}
            errors_seen.append({"attempt": attempt, "status": response.status_code, "category": "unsupported_field", "detail": _safe_error_detail(response, api_key)[:200]})
            rescues_applied.append(f"Gỡ extra refused: {', '.join(removed_keys)}")
            rescue_types.append("unsupported_field")
            response.close()
            continue

        if attempt < 6 and response.status_code == 400 and any(k in payload for k in ("max_tokens", "max_completion_tokens")) and re.search(r"max_(completion_)?tokens", response.text, re.I):
            fixed = api_proxy.adjust_token_limit(payload, response.text, is_gemini=False)
            if fixed is not None:
                payload, desc = fixed
                if "max_tokens" in payload and isinstance(payload["max_tokens"], int):
                    cap_val = payload["max_tokens"]
                elif "max_completion_tokens" in payload and isinstance(payload["max_completion_tokens"], int):
                    cap_val = payload["max_completion_tokens"]
                else:
                    cap_val = 0
                with _CLIENT_LOCK:
                    _bound_client_map(OUTPUT_CAPS, (provider.api_base, model))
                    OUTPUT_CAPS[(provider.api_base, model)] = cap_val
                errors_seen.append({"attempt": attempt, "status": response.status_code, "category": "token_limit", "detail": _safe_error_detail(response, api_key)[:200]})
                rescues_applied.append(desc)
                rescue_types.append("token_limit")
                response.close()
                continue

        if attempt < 6 and stream_active and response.status_code == 400 and "stream_options" in response.text.lower():
            payload.pop("stream_options", None)
            errors_seen.append({"attempt": attempt, "status": 400, "category": "unsupported_field", "detail": _safe_error_detail(response, api_key)[:200]})
            rescues_applied.append("Gỡ stream_options")
            rescue_types.append("unsupported_field")
            response.close()
            continue

        if 300 <= response.status_code < 400:
            response.close()
            api_proxy.record_event(
                caller="chaos_agent", provider=provider.label, model=model, status="failed", attempts=attempt,
                latency_ms=round((time.monotonic() - started) * 1000), errors_seen=errors_seen, rescues_applied=rescues_applied,
                detail=f"{provider.label} redirected the request",
            )
            raise RuntimeError(f"{provider.label} redirected the request")

        if not api_proxy.is_success_response(response):
            detail = _safe_error_detail(response, api_key)
            cat = api_proxy.classify_http_error(response.status_code, detail, payload, is_gemini=False)
            errors_seen.append({"attempt": attempt, "status": response.status_code, "category": cat, "detail": detail[:220]})

            if cat == "quota_or_limit" and (response.status_code == 402 or response.status_code == 429):
                api_proxy.record_event(
                    caller="chaos_agent", provider=provider.label, model=model, status="skipped_quota", attempts=attempt,
                    latency_ms=round((time.monotonic() - started) * 1000), errors_seen=errors_seen,
                    rescues_applied=[*rescues_applied, "Bỏ qua 429 / hết quota"], detail=detail,
                )
            elif attempt < 6:
                rescued_msg = ""
                wrap_kw: dict = {}
                if cat == "too_many_tools" and isinstance(payload.get("tools"), list) and len(payload["tools"]) > 1:
                    before = len(payload["tools"])
                    trimmed, target = api_proxy.trim_openai_tools(payload["tools"], detail)
                    payload["tools"] = trimmed
                    rescued_msg = f"Cắt tools {before} → {target}"
                    rescue_types.append("too_many_tools")
                    wrap_kw["tool_cap"] = target
                elif cat == "too_many_images":
                    before_images = api_proxy.count_payload_images(payload)
                    res = api_proxy.rescue_payload_images_limit(
                        payload, detail, is_gemini=False, allow_strip_all=True,
                    )
                    if res is not None:
                        payload, rescued_msg = res
                        rescue_types.append("too_many_images")
                        after_images = api_proxy.count_payload_images(payload)
                        img_cap = api_proxy.extract_limit_number(detail, r"images?", before_images)
                        wrap_kw["image_cap"] = 0 if after_images == 0 else (img_cap if img_cap is not None else after_images)
                elif cat == "payload_too_large":
                    downsize_attempts += 1
                    new_payload, did = api_proxy.downsize_payload_images(payload, attempt=downsize_attempts, is_gemini=False)
                    if did:
                        payload = new_payload
                        rescued_msg = f"Nén ảnh quá cỡ (lần {downsize_attempts})"
                        rescue_types.append("payload_too_large")
                elif cat == "stream_unsupported" and stream_active:
                    stream_active = False
                    payload["stream"] = False
                    payload.pop("stream_options", None)
                    rescued_msg = "Chuyển stream → non-stream"
                    rescue_types.append("stream_unsupported")
                    wrap_kw["force_non_stream"] = True
                elif cat == "system_role_unsupported":
                    new_payload, did = api_proxy.fold_system_messages(payload)
                    if did:
                        payload = new_payload
                        rescued_msg = "Gộp system prompt vào user message"
                        rescue_types.append("system_role_unsupported")
                        wrap_kw["fold_system"] = True
                elif cat == "token_or_context_limit":
                    new_payload, did = api_proxy.compact_payload_context(payload, is_gemini=False)
                    if did:
                        payload = new_payload
                        rescued_msg = "Nén lịch sử ngữ cảnh vượt giới hạn"
                        rescue_types.append("context_overflow")
                elif cat == "unsupported_field":
                    new_payload, removed = api_proxy.strip_rejected_fields(payload, detail, api_base=provider.api_base, model=model, is_gemini=False, persist=False)
                    if removed:
                        payload = new_payload
                        rescued_msg = f"Gỡ trường bị từ chối: {', '.join(removed)}"
                        rescue_types.append("unsupported_field")
                        wrap_kw["refused_fields"] = removed
                if rescued_msg:
                    rescues_applied.append(rescued_msg)
                    api_proxy.wrap_target(
                        base_url_key, model, url=chat_url(provider), provider_id=provider.id,
                        provider_label=provider.label, api_key=api_key, error_detail=detail,
                        rescue_note=rescued_msg, **wrap_kw,
                    )
                    response.close()
                    continue

            response.close()
            if tools and 400 <= response.status_code < 500 and "tool" in detail.lower() and not api_proxy.is_tools_limit_error(detail, len(payload.get("tools") or [])):
                api_proxy.record_event(
                    caller="chaos_agent", provider=provider.label, model=model, status="failed", attempts=attempt,
                    latency_ms=round((time.monotonic() - started) * 1000), errors_seen=errors_seen, rescues_applied=rescues_applied, detail=detail,
                )
                raise ToolsUnsupported(detail)
            if ASK_ON_LIMIT.get() and (response.status_code == 402 or response.status_code == 429 and not (
                    _pace_hint(detail) or _retry_after(response) or _hinted_wait(detail))):
                raise KeyLimited(f"{provider.label} HTTP {response.status_code}: {detail}")
            busy = response.status_code == 429 and (not re.search(r"quota|billing|insufficient|balance|credit", detail, re.I) or bool(_pace_hint(detail)))
            retry = response.headers.get("x-should-retry", "").lower()
            error = TransientError if (response.status_code >= 500 or response.status_code in (408, 409) or busy or retry == "true") and retry != "false" else RuntimeError
            if response.status_code not in (402, 429):
                api_proxy.record_event(
                    caller="chaos_agent", provider=provider.label, model=model, status="failed", attempts=attempt,
                    latency_ms=round((time.monotonic() - started) * 1000), errors_seen=errors_seen, rescues_applied=rescues_applied, detail=detail,
                )
            raise error(f"{provider.label} HTTP {response.status_code}: {detail}")

        if stream_active:
            try:
                message, usage, _ = _read_stream(response, on_delta)
            except Exception as exc:
                _check_cancelled()
                if not isinstance(exc, requests.RequestException):
                    raise
                errors_seen.append({"attempt": attempt, "status": 200, "category": "stream_broke", "detail": type(exc).__name__})
                if attempt < 6:
                    stream_active = False
                    payload["stream"] = False
                    payload.pop("stream_options", None)
                    rescues_applied.append("Stream đứt giữa chừng → chuyển non-stream")
                    rescue_types.append("stream_unsupported")
                    continue
                raise TransientError(f"{provider.label} stream broke: {type(exc).__name__}") from exc
            finally:
                response.close()
            _check_cancelled()
            api_proxy.record_event(
                caller="chaos_agent", provider=provider.label, model=model,
                status="rescued" if rescues_applied else "ok", attempts=attempt,
                latency_ms=round((time.monotonic() - started) * 1000),
                errors_seen=errors_seen, rescues_applied=rescues_applied, rescue_types=rescue_types,
            )
            return _build(message, usage, tools)

        cancel_hook = lambda: _force_close_response(response)
        _register_cancel_hook(cancel_hook)
        try:
            body = response.json()
            message = {**body["choices"][0]["message"], "_debug": {"finish": body["choices"][0].get("finish_reason") or "", "fields": sorted(body["choices"][0]["message"])}}
        except Exception as exc:
            _check_cancelled()
            if not isinstance(exc, (ValueError, KeyError, IndexError, TypeError)):
                raise
            api_proxy.record_event(
                caller="chaos_agent", provider=provider.label, model=model, status="failed", attempts=attempt,
                latency_ms=round((time.monotonic() - started) * 1000), errors_seen=errors_seen, rescues_applied=rescues_applied,
                detail=f"{provider.label} returned no message",
            )
            raise RuntimeError(f"{provider.label} returned no message") from exc
        finally:
            _unregister_cancel_hook(cancel_hook)
            response.close()
        api_proxy.record_event(
            caller="chaos_agent", provider=provider.label, model=model,
            status="rescued" if rescues_applied else "ok", attempts=attempt,
            latency_ms=round((time.monotonic() - started) * 1000),
            errors_seen=errors_seen, rescues_applied=rescues_applied, rescue_types=rescue_types,
        )
        return _build(message, body.get("usage") or {}, tools)
    raise RuntimeError(f"{provider.label} rescue loop exceeded")
