"""One chat call to any configured provider, with native tool calls or a plain-text fallback."""
from __future__ import annotations

import json
import random
import re
import time
import uuid
from email.utils import parsedate_to_datetime

import requests
from loguru import logger

from app.ai_providers import AIProvider, chat_url_candidates
from app.logging_config import logger
from app.parameters import TRANSLATION_CONNECT_TIMEOUT_SECONDS
from app.security import validate_url
from app.visual_qc.deepseek_region_client import _safe_error_detail

READ_TIMEOUT = 600
STREAM_IDLE_TIMEOUT = 120
NETWORK_RETRIES = 10
# How providers word a full context window (pi's list), and the words that look the same but are rate limits.
OVERFLOW_PATTERNS = [re.compile(p, re.I) for p in (
    r"prompt (?:is )?too long", r"prompt exceeds max length", r"request_too_large", r"input is too long for requested model", r"exceeds the context window",
    r"exceeds (?:the )?(?:model'?s )?maximum context length", r"input token count.*exceeds the maximum", r"maximum prompt length is \d+",
    r"reduce the length of the messages", r"maximum context length is \d+ tokens", r"exceeds (?:the )?maximum allowed input length", r"is longer than the model'?s context length",
    r"exceeds the limit of \d+", r"exceeds the available context size", r"greater than the context length", r"context window exceeds limit", r"exceeded model token limit",
    r"too large for model with \d+ maximum context length", r"configured context size is", r"model_context_window_exceeded", r"prompt too long; exceeded",
    r"range of input length should be", r"context[_ ]length[_ ]exceeded", r"too many tokens", r"token limit exceeded", r"context.{0,20}(?:length|window)", r"maximum context")]
NOT_OVERFLOW = re.compile(r"rate limit|too many requests|throttl", re.I)
# Each finds the model's window in the message; the first that matches wins.
WINDOW_PATTERNS = [re.compile(p, re.I) for p in (
    r"maximum context length (?:is|of)? ?\(?([\d,]{4,})", r"maximum prompt length is ([\d,]{4,})", r"context (?:window|length|size|limit) (?:of|is) \(?([\d,]{4,})",
    r"context (?:length|size) \(([\d,]{4,})", r"maximum number of tokens allowed \(([\d,]{4,})", r"maximum allowed input length of ([\d,]{4,})",
    r"limit of ([\d,]{4,})", r"> ([\d,]{4,}) maximum", r"configured context size is ([\d,]{4,})")]
RATE_LIMIT_RETRIES = 4
# When a provider says "slow down", every session using it waits, not just the one that was told.
_COOLDOWN: dict[str, float] = {}
# A pace a provider asked for in words ("1 request every 1 minutes"): provider id -> (seconds between requests, until when, last sent).
_PACE: dict[str, list[float]] = {}
PACE_HOLD_S = 900  # A pace lapses this long after the provider last asked for it.
PACE_RE = re.compile(r"(\d+) requests? (?:every|per|each) (\d+)? ?(second|sec|minute|min|hour)", re.I)
AFTER_RE = re.compile(r"try again (?:after|at) (\d{1,2}):(\d{2})(?::(\d{2}))?", re.I)
IN_RE = re.compile(r"(?:try again|retry) in (\d+(?:\.\d+)?) ?(ms|milliseconds?|s|secs?|seconds?|m|mins?|minutes?)\b", re.I)
# A block may lack its closing tag when the model stops early or opens the next call.
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|(?=<tool_call>)|\Z)", re.S)
PARAMETER_RE = re.compile(r"<parameter=(\w+)>\s*(.*?)\s*</parameter>", re.S)
TEXT_TOOLS_GUIDE = """You call tools by writing, anywhere in your reply, one block per call:
<tool_call>{"name": "read_file", "arguments": {"path": "app/main.py"}}</tool_call>
Results come back in <tool_result> blocks. Stop after your tool calls and wait for the results. Tools:
"""


CUT_OFF = ("Your reply hit the output length limit and this call was cut off before it ended. Send it in smaller pieces: "
           "create the file with a short write_file, then add the rest with edit_file or run_command (cat >> file <<'EOF').")


class TransientError(RuntimeError):
    """A failure worth asking again: a timeout, a dropped connection or a 5xx from the provider."""


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
        f"- {s['name']}: {s['description']} Arguments: {json.dumps((s.get('parameters') or {}).get('properties', {}))}"
        for s in specs
    )


UNESCAPES = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "/": "/"}


def _lenient_args(raw: str, params: list[str], required: list[str]) -> dict | None:
    """Arguments from a call whose JSON broke on long text: split at the known parameter names and unescape each value."""
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
    """Each tool's parameter names and required ones, for reading calls whose JSON is broken."""
    return {t["name"]: (list((t.get("parameters") or {}).get("properties") or {}), list((t.get("parameters") or {}).get("required") or []))
            for t in tools or []}


def parse_text_calls(text: str, tools: list[dict] | None = None) -> tuple[str, list[dict]]:
    """Tool calls written as <tool_call> blocks, and the reply with those blocks taken out."""
    calls = []
    schemas = schemas_of(tools)
    for raw in TOOL_CALL_RE.findall(text or ""):
        found = _read_call(raw, schemas)
        if found is None:
            calls.append({"id": uuid.uuid4().hex[:12], "name": "", "args": {}, "error": f"Unreadable tool call: {raw[:200]}{' … ' + raw[-150:] if len(raw) > 400 else ''}. Write one valid JSON object: escape newlines as \\n and quotes as \\\", and keep arguments short."})
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


def is_overflow(message: str) -> bool:
    return any(p.search(message) for p in OVERFLOW_PATTERNS) and not NOT_OVERFLOW.search(message)


def context_window(message: str) -> int:
    """The model's window in tokens when the error names it, else 0."""
    for pattern in WINDOW_PATTERNS:
        found = pattern.search(message)
        if found:
            return int(found.group(1).replace(",", ""))
    return 0


def _retry_after(response: requests.Response) -> float:
    """Seconds the server asked to wait: retry-after-ms, retry-after as seconds, or retry-after as a date."""
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
    """Seconds the provider's own words ask to wait ("try again in 20s", "try again after 05:18:33"), or 0."""
    found = IN_RE.search(text)
    if found:
        unit = found.group(2).lower()
        return float(found.group(1)) * (0.001 if unit.startswith("ms") or unit.startswith("milli") else 60 if unit.startswith("m") else 1)
    found = AFTER_RE.search(text)
    if found:
        now = time.gmtime()
        target = int(found.group(1)) * 3600 + int(found.group(2)) * 60 + int(found.group(3) or 0)
        wait = (target - (now.tm_hour * 3600 + now.tm_min * 60 + now.tm_sec)) % 86400
        # A clock time in another zone would read as hours away; only a near one is trusted.
        return float(wait) if wait <= 900 else 0.0
    return 0.0


def _pace_hint(text: str) -> float:
    """Seconds between requests when the provider names a pace, or 0."""
    found = PACE_RE.search(text)
    if not found or int(found.group(1)) == 0:
        return 0.0
    unit = found.group(3).lower()
    period = int(found.group(2) or 1) * (3600 if unit.startswith("h") else 60 if unit.startswith("m") else 1)
    return period / int(found.group(1))


def _wait_for_pace(provider_id: str) -> None:
    pace = _PACE.get(provider_id)
    if not pace:
        return
    interval, until, last = pace
    if time.time() > until:
        _PACE.pop(provider_id, None)
        return
    gap = last + interval - time.time()
    if gap > 0:
        time.sleep(gap)
    pace[2] = time.time()


# Rate-limit answers seen per provider: how many, how long we waited, and the provider's last words, so a session can report them.
RATE_LIMITS: dict[str, dict] = {}


def _note_rate_limit(provider: AIProvider, response: requests.Response, wait: float) -> None:
    try:
        detail = response.text[:300]
    except Exception:
        detail = ""
    row = RATE_LIMITS.setdefault(provider.id, {"count": 0, "waited_s": 0.0, "detail": "", "first": time.time()})
    row["count"] += 1
    row["waited_s"] += wait
    row["detail"] = detail or row["detail"]
    row["last"] = time.time()
    logger.info("{} answered 429 ({} so far, waiting {:.0f} s): {}", provider.label, row["count"], wait, detail[:200])


# Resolved chat endpoint per api_base: the first candidate that answered.
_RESOLVED_URL: dict[str, str] = {}


def _looks_like_wrong_endpoint(response: requests.Response) -> bool:
    """True when the response means 'this URL is not the chat endpoint'.

    404/405 are unambiguous. A 403 with a non-JSON body is a path-level block
    (e.g. Cloudflare in front of a gateway that only serves /v1/*); a 403
    with a JSON error body is the API refusing the key, which must NOT
    trigger a fallback.
    """
    if response.status_code in (404, 405):
        return True
    if response.status_code == 403:
        ctype = response.headers.get("content-type", "")
        if "json" not in ctype.lower():
            return True
    return False


def _resolved_chat_url(provider: AIProvider) -> str:
    """Pick the working chat endpoint for provider.api_base, probing once."""
    key = provider.api_base.strip().rstrip("/")
    hit = _RESOLVED_URL.get(key)
    if hit:
        return hit
    cands = chat_url_candidates(provider.api_base)
    if not cands:
        raise RuntimeError(f"{provider.label}: no chat endpoint for empty api_base")
    # Fast path: single candidate, no probing needed.
    if len(cands) == 1:
        _RESOLVED_URL[key] = cands[0]
        return cands[0]
    probe_headers = {"Content-Type": "application/json"}
    for url in cands:
        try:
            validate_url(url)
        except Exception:
            continue
        try:
            # Minimal probe: wrong model name still yields a JSON API error
            # (401/400) on the right endpoint, vs 404/403-HTML on the wrong one.
            resp = requests.post(
                url, headers=probe_headers,
                json={"model": "__probe__", "messages": []},
                timeout=(5, 10), allow_redirects=False, stream=False,
            )
        except requests.RequestException:
            continue
        try:
            if _looks_like_wrong_endpoint(resp):
                continue
            _RESOLVED_URL[key] = url
            return url
        finally:
            resp.close()
    # Nothing answered: fall back to the first candidate and let the normal
    # error path report what the server actually said.
    _RESOLVED_URL[key] = cands[0]
    return cands[0]


def _post(provider: AIProvider, api_key: str, payload: dict, stream: bool) -> requests.Response:
    """The request, with the shared 429 cooldown and retries.

    The chat endpoint is auto-resolved: chat_url_candidates() lists every
    likely URL for the configured api_base (bare host, /v1 base, or full
    endpoint) and the first one that answers wins. The winner is cached per
    api_base so only the very first call ever probes.
    """
    url = _resolved_chat_url(provider)
    validate_url(url)
    for attempt in range(RATE_LIMIT_RETRIES + 1):
        pause = _COOLDOWN.get(provider.id, 0.0) - time.time() if attempt == 0 else 0
        if pause > 0:
            time.sleep(min(pause, 60.0))
        _wait_for_pace(provider.id)
        try:
            response = requests.post(url, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                                     json=payload, timeout=(TRANSLATION_CONNECT_TIMEOUT_SECONDS, STREAM_IDLE_TIMEOUT if stream else READ_TIMEOUT),
                                     allow_redirects=False, stream=stream)
        except requests.RequestException as exc:
            raise TransientError(f"{provider.label} request failed: {type(exc).__name__}") from exc
        if response.status_code != 429 or attempt == RATE_LIMIT_RETRIES:
            return response
        # Free tiers allow a few requests a minute; wait as told (in a header or in words), or longer each time.
        try:
            words = response.text[:500]
        except Exception:
            words = ""
        interval = _pace_hint(words)
        if interval:
            # Once the provider names a slower pace, every later request keeps to it instead of failing a few times and giving up.
            _PACE[provider.id] = [interval, time.time() + PACE_HOLD_S, time.time()]
            RATE_LIMITS.setdefault(provider.id, {"count": 0, "waited_s": 0.0, "detail": "", "first": time.time()})["pace_s"] = interval
        wait = min(900.0 if interval else 60.0, _retry_after(response) or _hinted_wait(words) or interval or 6.0 * 2 ** attempt)
        _note_rate_limit(provider, response, wait)
        _COOLDOWN[provider.id] = max(_COOLDOWN.get(provider.id, 0.0), time.time() + wait)
        response.close()
        time.sleep(wait)
    return response


# M4: cap for streamed text/reasoning accumulation; a misbehaving provider streaming forever
# must not grow RAM without bound.
_STREAM_CAP = 4_000_000


def _read_stream(response: requests.Response, on_delta) -> tuple[dict, dict, bool]:
    """Assemble the streamed message; on_delta(live) is called as it grows and returns True to stop early."""
    # M4: the old code rebuilt "".join(all tool-call arguments) on EVERY chunk (O(n^2)) and let
    # text/reasoning grow without bound if a provider streams forever. Arguments are tracked
    # incrementally and the text buffers are capped.
    # M4: the old code rebuilt "".join(all tool-call arguments) on EVERY chunk (O(n^2)) and let
    # text/reasoning grow without bound if a provider streams forever. Arguments are tracked
    # incrementally and the text buffers are capped.
    text_parts, reasoning_parts = [], []
    text_len, reasoning_len = 0, 0
    arg_parts: dict[int, list[str]] = {}
    arg_len = 0
    usage, calls, stopped = {}, {}, False
    finish, other = "", {}
    response.encoding = "utf-8"  # An event stream without a charset is otherwise read as Latin-1 and Vietnamese text turns into mojibake.
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
            finish = choice.get("finish_reason") or finish
            for key, value in delta.items():
                if key not in ("content", "reasoning_content", "reasoning", "tool_calls", "role") and isinstance(value, str) and value:
                    other[key] = other.get(key, "") + value
            piece = delta.get("content") or ""
            if piece and text_len < _STREAM_CAP:
                take = min(len(piece), _STREAM_CAP - text_len)
                text_parts.append(piece[:take])
                text_len += take
            piece = delta.get("reasoning_content") or delta.get("reasoning") or ""
            if piece and reasoning_len < _STREAM_CAP:
                take = min(len(piece), _STREAM_CAP - reasoning_len)
                reasoning_parts.append(piece[:take])
                reasoning_len += take
            for part in delta.get("tool_calls") or []:
                slot = calls.setdefault(part.get("index", len(calls)), {"id": "", "name": "", "arguments": ""})
                slot["id"] = part.get("id") or slot["id"]
                function = part.get("function") or {}
                slot["name"] += function.get("name") or ""
                arg = function.get("arguments") or ""
                if arg:
                    arg_parts.setdefault(part.get("index", 0), []).append(arg)
                    arg_len += len(arg)
        # Only the tail is shown live; the full length is tracked without rebuilding the string.
        arg_tail = "".join(arg_parts.get(i, [""])[-1] for i in sorted(arg_parts))[-4000:] if arg_parts else ""
        if on_delta({"text": "".join(text_parts), "reasoning": "".join(reasoning_parts),
                     "tools": [c["name"] for c in calls.values() if c["name"]],
                     "args": arg_tail, "arg_chars": arg_len}):
            stopped = True
            break
    text, reasoning = "".join(text_parts), "".join(reasoning_parts)
    for idx, slot in calls.items():
        slot["arguments"] = "".join(arg_parts.get(idx, []))
    if other and not (text or reasoning or calls):
        # Some providers put the reply in a field of their own; keep it rather than lose it.
        reasoning = "".join(other.values())
    message = {"content": text, "reasoning_content": reasoning, "_debug": {"finish": finish, "fields": sorted(other)},
               "tool_calls": [{"id": c["id"], "function": {"name": c["name"], "arguments": c["arguments"]}} for _, c in sorted(calls.items())]}
    return message, usage, stopped


THINK = re.compile(r"<think>(.*?)</think>", re.S)


def split_thinking(text: str) -> tuple[str, str]:
    """Reasoning some models write into the reply itself (<think>…</think>, or only the closing tag) is moved out of it."""
    thought = "\n".join(part.strip() for part in THINK.findall(text))
    text = THINK.sub("", text)
    if "</think>" in text:
        before, _, text = text.partition("</think>")
        thought = (before.strip() + "\n" + thought).strip()
    return text.strip(), thought


def loads_args(raw: str):
    """Tool arguments as JSON, after the repairs models most often need (pi's repairJson): code fences, stray backslashes, trailing commas."""
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
    # Models without native tools, and some with them, write their calls into the text.
    if "<tool_call>" in text:
        text, written = parse_text_calls(text, tools)
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
    if "think>" in str(text):
        text, thought = split_thinking(str(text))
        reasoning = (str(reasoning) + "\n" + thought).strip()
    result = {"text": str(text).strip(), "calls": calls, "reasoning": str(reasoning), "usage": usage_of(usage)}
    finish = (message.get("_debug") or {}).get("finish")
    result["finish"] = finish or ""
    if finish == "length":
        # Fail-safe (Pi): a reply cut off by the length limit may have truncated tool call
        # arguments mid-JSON; executing them would run garbled commands. Mark every call
        # as failed so the model retries with smaller pieces instead.
        for call in calls:
            if not call.get("error"):
                call["error"] = CUT_OFF
            elif CUT_OFF not in call["error"]:
                call["error"] = CUT_OFF + " " + call["error"]
    if not (result["text"] or calls) and message.get("_debug"):
        result["debug"] = message["_debug"]
    return result


def complete(provider: AIProvider, api_key: str, model: str, messages: list[dict], *, tools: list[dict] | None, on_delta=None,
             max_tokens: int | None = None) -> dict:
    """One assistant turn: its text, tool calls, reasoning and token usage; streamed when on_delta is given. Transient failures are retried."""
    for attempt in range(NETWORK_RETRIES + 1):
        try:
            return _complete_once(provider, api_key, model, messages, tools=tools, on_delta=on_delta, max_tokens=max_tokens)
        except TransientError as exc:
            if attempt == NETWORK_RETRIES:
                raise
            wait = min(90.0, 2.0 * 2 ** attempt) * random.uniform(0.75, 1.0)
            logger.warning("{} ({}); asking again in {:.0f}s", exc, provider.label, wait)
            time.sleep(wait)
    raise AssertionError("unreachable")


def _complete_once(provider: AIProvider, api_key: str, model: str, messages: list[dict], *, tools: list[dict] | None, on_delta=None,
                   max_tokens: int | None = None) -> dict:
    payload = {"model": model, "messages": messages, "stream": bool(on_delta)}
    payload.update(provider.chat_completion_extras())
    if tools:
        payload["tools"] = native_tools(tools)
    if on_delta:
        payload["stream_options"] = {"include_usage": True}
    if max_tokens:
        payload["max_tokens"] = int(max_tokens)
    response = _post(provider, api_key, payload, bool(on_delta))
    if max_tokens and response.status_code == 400 and "max_tokens" in response.text.lower() or (
            max_tokens and response.status_code == 400 and "max_completion_tokens" in response.text.lower()):
        # A provider with a lower cap, or none to set, refuses the field; ask again without it.
        payload.pop("max_tokens")
        response.close()
        response = _post(provider, api_key, payload, bool(on_delta))
    if on_delta and response.status_code == 400 and "stream_options" in response.text.lower():
        payload.pop("stream_options")
        response.close()
        response = _post(provider, api_key, payload, True)
    if 300 <= response.status_code < 400:
        raise RuntimeError(f"{provider.label} redirected the request")
    if not response.ok:
        # M5: close the streamed error response before raising; the old code leaked the socket
        # on every non-OK streaming response (the 429 path closed it, this one did not).
        detail = _safe_error_detail(response, api_key)
        response.close()
        if tools and 400 <= response.status_code < 500 and "tool" in detail.lower():
            raise ToolsUnsupported(detail)
        # A busy model ("at capacity, retry in a few seconds") is waited out like a server error; a spent quota or balance is not.
        # A spent daily quota that still lets requests through at a slower pace is waited out too.
        busy = response.status_code == 429 and (not re.search(r"quota|billing|insufficient|balance|credit", detail, re.I) or bool(_pace_hint(detail)))
        retry = response.headers.get("x-should-retry", "").lower()
        error = TransientError if (response.status_code >= 500 or response.status_code in (408, 409) or busy or retry == "true") and retry != "false" else RuntimeError
        raise error(f"{provider.label} HTTP {response.status_code}: {detail}")
    if on_delta:
        try:
            message, usage, _ = _read_stream(response, on_delta)
        except requests.RequestException as exc:
            raise TransientError(f"{provider.label} stream broke: {type(exc).__name__}") from exc
        finally:
            response.close()
        return _build(message, usage, tools)
    try:
        body = response.json()
        message = {**body["choices"][0]["message"], "_debug": {"finish": body["choices"][0].get("finish_reason") or "", "fields": sorted(body["choices"][0]["message"])}}
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"{provider.label} returned no message") from exc
    return _build(message, body.get("usage") or {}, tools)
