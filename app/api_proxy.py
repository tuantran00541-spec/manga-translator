from __future__ import annotations

import base64
import copy
import json
import re
import threading
import time
import uuid
from collections import deque
from typing import Any, Callable
from urllib.parse import urlparse

import cv2
import numpy as np
import requests

from app.ai_providers import AIProvider, api_base_candidates
from app.downloader.http import public_request
from app.logging_config import logger
from app.security import validate_url

MAX_LOG_EVENTS = 300
MAX_RESCUES_PER_CALL = 5
MAX_LEARNED_ENTRIES = 256
TRANSIENT_STATUS_CODES = frozenset({408, 409, 500, 502, 503, 504})
_ORIGINAL_REQUESTS_POST = requests.post
_KEY_PARAM_RE = re.compile(r"([?&](?:api_key|key|access_token|token)=)[^&\s\"']+", re.I)
_BEARER_RE = re.compile(r"(\bBearer\s+)[A-Za-z0-9._~+/=-]{8,}", re.I)

QUOTA_RE = re.compile(
    r"\b(?:insufficient_quota|quota[_\s-]*exceeded|exceeded your current quota|"
    r"rate[_\s-]*limit(?:ed)?|rate_limit_exceeded|too many requests|"
    r"requests per min(?:ute)?|tokens per min(?:ute)?|"
    r"insufficient\s+(?:balance|credits|funds)|out of credits|billing|payment required)\b",
    re.I,
)
IMAGE_LIMIT_RE = re.compile(
    r"(?:too many images|maximum.*?\bimages?\b|max(?:imum)?[_\s-]*(?:number of\s*)?images?|"
    r"at most\s*\d+\s*images?|up to\s*\d+\s*images?|only supports?\s*(?:up to\s*)?\d+\s*images?|"
    r"exceeds?\s*(?:the\s*)?(?:maximum\s*)?(?:\d+\s*)?images?|image(?:s)?\s+(?:limit|count)|"
    r"vision is not supported|does not support images?|image input.*not supported)",
    re.I,
)
TOOL_LIMIT_RE = re.compile(
    r"(?:too many (?:tools|functions|function declarations)|"
    r"(?:tools|functions|function declarations)\s*(?:array|list)?\s*(?:must have|length|count|limit|max|exceed|too long|too many)|"
    r"(?:max(?:imum)?|at most|up to|no more than|<=)\s*\d+\s*(?:tools|functions|function declarations))",
    re.I,
)
TOOLS_UNSUPPORTED_RE = re.compile(
    r"(?:does not support\s*(?:tools|function)|(?:tools|function calling|tool use)\s*(?:is|are)?\s*(?:not supported|disabled|unsupported|not allowed)|"
    r"unknown field\s*['\"]?tools['\"]?|unsupported parameter[:\s'\"]*tools)",
    re.I,
)
PAYLOAD_LARGE_RE = re.compile(
    r"(?:request entity too large|payload too large|body too large|content too large|"
    r"image(?:s)?\s+(?:is\s+)?too (?:large|big)|size exceeds|maximum.*(?:mb|bytes|size)|exceeds.*mb)",
    re.I,
)
CONTEXT_OVERFLOW_RE = re.compile(
    r"(?:context_length_exceeded|maximum context length|prompt is too long|"
    r"input is too long|request too many tokens|token limit exceeded|exceeds the context window|"
    r"too many input tokens|maximum number of tokens)",
    re.I,
)
TOKEN_CAP_RE = re.compile(r"max_(?:completion_|output_)?tokens", re.I)
SYSTEM_ROLE_RE = re.compile(
    r"(?:system\s+(?:role|message|prompt)s?\s*(?:is|are)?\s*(?:not supported|unsupported|not allowed|invalid)|"
    r"developer\s+instruction.*not supported|invalid role[:\s'\"]*system)",
    re.I,
)

OPTIONAL_FIELDS = (
    "response_format",
    "reasoning_effort",
    "stream_options",
    "parallel_tool_calls",
    "temperature",
    "top_p",
    "presence_penalty",
    "frequency_penalty",
    "seed",
    "store",
    "modalities",
    "verbosity",
    "reasoning",
)

CORE_TOOL_ORDER = (
    "read_file",
    "edit_file",
    "write_file",
    "list_dir",
    "search",
    "glob",
    "run_command",
    "ask_user",
    "todo_write",
    "web_fetch",
    "web_search",
)

_LOCK = threading.Lock()
_EVENTS: deque[dict] = deque(maxlen=MAX_LOG_EVENTS)
_STATS: dict[str, int] = {
    "total_calls": 0,
    "ok_calls": 0,
    "rescued_calls": 0,
    "skipped_quota_calls": 0,
    "failed_calls": 0,
}
_RESCUE_COUNTS: dict[str, int] = {}
_REFUSED_FIELDS: dict[tuple[str, str], set[str]] = {}
_FOUND_CHAT_URLS: dict[str, str] = {}
_WRAPPED_TARGETS: dict[tuple[str, str], dict] = {}


def is_success_response(response) -> bool:
    """Return True strictly for 2xx HTTP status codes, falling back to `ok is True` for statusless mocks."""
    status = getattr(response, "status_code", None)
    if isinstance(status, int) and not isinstance(status, bool):
        return 200 <= status < 300
    return getattr(response, "ok", None) is True


def redact_secret(text: str, api_key: str = "") -> str:
    cleaned = str(text or "")
    secret = str(api_key or "").strip()
    if secret and len(secret) >= 4:
        cleaned = cleaned.replace(secret, "[redacted]")
    cleaned = _KEY_PARAM_RE.sub(r"\1[redacted]", cleaned)
    cleaned = _BEARER_RE.sub(r"\1[redacted]", cleaned)
    return cleaned


def _bound_dict(mapping: dict, key: Any, limit: int = MAX_LEARNED_ENTRIES) -> None:
    if key not in mapping:
        while len(mapping) >= limit:
            mapping.pop(next(iter(mapping)))


def _public_post(url: str, **kwargs) -> requests.Response:
    return public_request("POST", url, **kwargs)


def _mask_secret(api_key: str) -> str:
    cleaned = str(api_key or "").strip()
    if not cleaned:
        return ""
    if len(cleaned) <= 8:
        return f"{cleaned[:2]}…{cleaned[-1:]}" if len(cleaned) >= 4 else "[đã bọc]"
    return f"{cleaned[:5]}…{cleaned[-4:]}"


def _base_key(api_base: str, url: str = "") -> str:
    raw = str(api_base or "").strip().rstrip("/")
    if raw:
        return raw
    if not url:
        return ""
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}".rstrip("/") if parsed.scheme and parsed.netloc else url


def _format_wrap_rules(entry: dict) -> list[str]:
    rules: list[str] = []
    if entry.get("tool_cap"):
        rules.append(f"Tools ≤ {entry['tool_cap']}")
    if entry.get("image_cap") is not None:
        rules.append(f"Ảnh ≤ {entry['image_cap']}")
    if entry.get("token_cap"):
        rules.append(f"Max tokens ≤ {entry['token_cap']}")
    if entry.get("force_non_stream"):
        rules.append("Non-stream")
    if entry.get("fold_system"):
        rules.append("Gộp system → user")
    if entry.get("refused_fields"):
        rules.append(f"Bỏ: {', '.join(sorted(entry['refused_fields']))}")
    if entry.get("chat_url"):
        rules.append(f"Endpoint: {entry['chat_url']}")
    return rules


def wrap_target(
    api_base: str,
    model: str,
    *,
    url: str = "",
    provider_id: str = "",
    provider_label: str = "",
    api_key: str = "",
    chat_url: str = "",
    refused_fields: list[str] | None = None,
    tool_cap: int | None = None,
    image_cap: int | None = None,
    token_cap: int | None = None,
    force_non_stream: bool | None = None,
    fold_system: bool | None = None,
    error_detail: str = "",
    rescue_note: str = "",
) -> dict | None:
    """Register or update an auto-wrapped (base_url, model) target after an error occurs."""
    base = _base_key(api_base, url)
    model_key = str(model or "").strip()
    if not base or not model_key:
        return None
    key = (base, model_key)
    with _LOCK:
        entry = _WRAPPED_TARGETS.get(key)
        if entry is None:
            entry = {
                "api_base": base,
                "model": model_key,
                "provider_id": provider_id or "",
                "provider": provider_label or provider_id or base,
                "key_hint": _mask_secret(api_key),
                "chat_url": "",
                "refused_fields": set(),
                "tool_cap": None,
                "image_cap": None,
                "token_cap": None,
                "force_non_stream": False,
                "fold_system": False,
                "wrapped_at": round(time.time(), 3),
                "rescued_count": 0,
                "last_error": "",
                "last_rescue": "",
            }
            _bound_dict(_WRAPPED_TARGETS, key)
            _WRAPPED_TARGETS[key] = entry
        if api_key and not entry.get("key_hint"):
            entry["key_hint"] = _mask_secret(api_key)
        if provider_label:
            entry["provider"] = provider_label
        if provider_id:
            entry["provider_id"] = provider_id
        if chat_url:
            entry["chat_url"] = chat_url
        if refused_fields:
            entry["refused_fields"].update(refused_fields)
            _bound_dict(_REFUSED_FIELDS, key)
            _REFUSED_FIELDS.setdefault(key, set()).update(refused_fields)
        if tool_cap is not None and tool_cap >= 1:
            prev = entry.get("tool_cap")
            entry["tool_cap"] = min(prev, int(tool_cap)) if prev else int(tool_cap)
        if image_cap is not None and image_cap >= 0:
            prev = entry.get("image_cap")
            entry["image_cap"] = min(prev, int(image_cap)) if prev is not None else int(image_cap)
        if token_cap is not None and token_cap >= 128:
            prev = entry.get("token_cap")
            entry["token_cap"] = min(prev, int(token_cap)) if prev else int(token_cap)
        if force_non_stream is True:
            entry["force_non_stream"] = True
        if fold_system is True:
            entry["fold_system"] = True
        entry["rescued_count"] = int(entry.get("rescued_count") or 0) + 1
        if error_detail:
            entry["last_error"] = error_detail[:220]
        if rescue_note:
            entry["last_rescue"] = rescue_note[:220]
        return dict(entry)


def get_wrapped_image_cap(api_base: str, model: str) -> int | None:
    """Return learned image-count cap for a wrapped (api_base, model), if any."""
    base = str(api_base or "").strip().rstrip("/")
    model_key = str(model or "").strip()
    if not base or not model_key:
        return None
    with _LOCK:
        entry = _WRAPPED_TARGETS.get((base, model_key))
        cap = entry.get("image_cap") if entry else None
        return int(cap) if isinstance(cap, int) and cap >= 1 else None


def unwrap_target(api_base: str, model: str) -> bool:
    """Remove learned auto-wrap rules for one (api_base, model) pair."""
    base = _base_key(api_base)
    model_key = str(model or "").strip()
    with _LOCK:
        removed = _WRAPPED_TARGETS.pop((base, model_key), None) is not None
        refused_removed = _REFUSED_FIELDS.pop((base, model_key), None) is not None
        url_removed = False
        if base and not any(k[0] == base for k in _WRAPPED_TARGETS):
            url_removed = _FOUND_CHAT_URLS.pop(base, None) is not None
        return removed or refused_removed or url_removed


def apply_wrapped_rules(
    payload: dict,
    *,
    api_base: str,
    model: str,
    stream: bool = False,
    is_gemini: bool = False,
    apply_image_cap: bool = True,
    allow_image_strip: bool = False,
) -> tuple[dict, bool, list[str], str | None]:
    """Pre-apply learned auto-wrap fixes so wrapped models skip repeating past errors."""
    base = _base_key(api_base)
    model_key = str(model or "").strip()
    with _LOCK:
        entry = copy.deepcopy(_WRAPPED_TARGETS.get((base, model_key))) if base and model_key else None
        learned_fields = set(_REFUSED_FIELDS.get((base, model_key), ())) if base and model_key else set()

    if entry and entry.get("refused_fields"):
        learned_fields.update(entry["refused_fields"])

    updated = copy.deepcopy(payload)
    current_stream = bool(stream)
    applied: list[str] = []
    chat_override = entry.get("chat_url") if (entry and entry.get("chat_url") and not is_gemini) else None

    if learned_fields and not is_gemini:
        stripped = []
        for field in sorted(learned_fields):
            if field in updated:
                updated.pop(field, None)
                stripped.append(field)
                if field == "response_format":
                    _append_json_hint(updated, is_gemini=False)
        if stripped:
            applied.append(f"Bọc sẵn — bỏ trường: {', '.join(stripped)}")

    if entry:
        if entry.get("force_non_stream") and current_stream:
            current_stream = False
            if not is_gemini:
                updated["stream"] = False
                updated.pop("stream_options", None)
            applied.append("Bọc sẵn — chạy non-stream")
        if entry.get("fold_system") and not is_gemini:
            updated, did = fold_system_messages(updated)
            if did:
                applied.append("Bọc sẵn — gộp system prompt")
        tool_cap = entry.get("tool_cap")
        if tool_cap and isinstance(updated.get("tools"), list) and len(updated["tools"]) > tool_cap:
            before = len(updated["tools"])
            trimmed, _ = trim_openai_tools(updated["tools"], f"max {tool_cap} tools")
            updated["tools"] = trimmed
            applied.append(f"Bọc sẵn — tools {before} -> {len(trimmed)}")
        token_cap = entry.get("token_cap")
        if token_cap and not is_gemini:
            for k in ("max_tokens", "max_completion_tokens"):
                if isinstance(updated.get(k), int) and updated[k] > token_cap:
                    updated[k] = token_cap
                    applied.append(f"Bọc sẵn — {k} <= {token_cap}")
        img_cap = entry.get("image_cap")
        if (
            apply_image_cap
            and allow_image_strip
            and img_cap is not None
            and (img_cap >= 1 or allow_image_strip)
            and count_payload_images(updated, is_gemini=is_gemini) > img_cap
        ):
            res = rescue_payload_images_limit(
                updated,
                f"at most {img_cap} images",
                is_gemini=is_gemini,
                allow_strip_all=allow_image_strip,
            )
            if res is not None:
                updated, desc = res
                applied.append(f"Bọc sẵn — {desc}")

    return updated, current_stream, applied, chat_override


class ImageLimitExceeded(RuntimeError):
    """Raised internally when a multi-image call needs chunk splitting by caller."""

    def __init__(self, limit: int | None, count: int, detail: str):
        super().__init__(detail)
        self.limit = limit
        self.count = count
        self.detail = detail


def _safe_error_detail(response: requests.Response, api_key: str = "") -> str:
    try:
        text = str(getattr(response, "text", "") or "")
    except Exception:
        text = ""
    if not text:
        try:
            body = response.json()
            text = json.dumps(body, ensure_ascii=False)
        except Exception:
            text = ""
    cleaned = redact_secret(text, api_key).strip()[:600]
    return cleaned or f"HTTP {getattr(response, 'status_code', '?')}"


def _record_rescue_types(categories: list[str]) -> None:
    for cat in categories:
        _RESCUE_COUNTS[cat] = _RESCUE_COUNTS.get(cat, 0) + 1


def record_event(
    *,
    caller: str,
    provider: str,
    model: str,
    status: str,
    attempts: int = 1,
    latency_ms: int = 0,
    errors_seen: list[dict] | None = None,
    rescues_applied: list[str] | None = None,
    rescue_types: list[str] | None = None,
    detail: str = "",
) -> dict:
    """Record one model API call and any errors/rescues into the proxy monitor."""
    entry = {
        "id": uuid.uuid4().hex[:10],
        "time": round(time.time(), 3),
        "caller": caller or "app",
        "provider": provider or "",
        "model": model or "",
        "status": status,
        "attempts": max(1, int(attempts)),
        "latency_ms": max(0, int(latency_ms)),
        "errors_seen": list(errors_seen or []),
        "rescues_applied": list(rescues_applied or []),
        "detail": (detail or "")[:280],
    }
    with _LOCK:
        _EVENTS.appendleft(entry)
        _STATS["total_calls"] += 1
        if status == "ok":
            _STATS["ok_calls"] += 1
        elif status == "rescued":
            _STATS["rescued_calls"] += 1
        elif status == "skipped_quota":
            _STATS["skipped_quota_calls"] += 1
        else:
            _STATS["failed_calls"] += 1
        if status == "rescued" and rescue_types:
            _record_rescue_types(rescue_types)
    return entry


def record_external_rescue(
    *,
    caller: str,
    provider: str,
    model: str,
    status: str,
    error_detail: str = "",
    error_category: str = "",
    rescue_action: str = "",
    rescue_type: str = "",
    latency_ms: int = 0,
) -> dict:
    """Helper for callers that perform custom post-processing rescues (e.g. JSON repair)."""
    return record_event(
        caller=caller,
        provider=provider,
        model=model,
        status=status,
        attempts=1,
        latency_ms=latency_ms,
        errors_seen=([{"category": error_category or "error", "detail": error_detail[:240]}] if error_detail else []),
        rescues_applied=([rescue_action] if rescue_action else []),
        rescue_types=([rescue_type] if rescue_type else []),
        detail=error_detail,
    )


def proxy_snapshot(limit: int = 100) -> dict:
    """Return proxy statistics and recent captured error/rescue events for monitoring UI."""
    with _LOCK:
        wrapped = [
            {
                "api_base": row["api_base"],
                "model": row["model"],
                "provider_id": row["provider_id"],
                "provider": row["provider"],
                "key_hint": row["key_hint"],
                "rules": _format_wrap_rules(row),
                "wrapped_at": row["wrapped_at"],
                "rescued_count": row["rescued_count"],
                "last_error": row["last_error"],
                "last_rescue": row["last_rescue"],
            }
            for row in sorted(_WRAPPED_TARGETS.values(), key=lambda r: r.get("wrapped_at", 0), reverse=True)
        ]
        return {
            "enabled": True,
            "mode": "auto",
            "stats": dict(_STATS),
            "rescue_counts": dict(_RESCUE_COUNTS),
            "learned_refused_fields": len(_REFUSED_FIELDS),
            "learned_endpoints": len(_FOUND_CHAT_URLS),
            "wrapped_targets": wrapped,
            "events": list(_EVENTS)[: max(1, min(limit, MAX_LOG_EVENTS))],
        }


def clear_proxy_logs(include_learned: bool = False) -> None:
    """Clear recorded proxy events and counters."""
    with _LOCK:
        _EVENTS.clear()
        for key in _STATS:
            _STATS[key] = 0
        _RESCUE_COUNTS.clear()
    if include_learned:
        reset_learned_state()


def reset_learned_state() -> None:
    """Clear learned refused fields, wrapped targets, and discovered endpoint paths."""
    with _LOCK:
        _REFUSED_FIELDS.clear()
        _FOUND_CHAT_URLS.clear()
        _WRAPPED_TARGETS.clear()


def is_quota_or_rate_limit(status_code: int, detail: str) -> bool:
    """Return True for HTTP 429/402 or billing/quota errors that proxy must skip."""
    if status_code in (402, 429):
        return True
    if not detail:
        return False
    if IMAGE_LIMIT_RE.search(detail) or TOOL_LIMIT_RE.search(detail) or CONTEXT_OVERFLOW_RE.search(detail):
        return False
    return bool(QUOTA_RE.search(detail))


def extract_limit_number(detail: str, noun_pattern: str, ceiling: int) -> int | None:
    """Parse numeric cap near a noun (e.g. 'at most 4 images', 'max 64 tools')."""
    if not detail:
        return None
    patterns = (
        rf"(?:max(?:imum)?(?:\s+number\s+of)?|at most|up to|no more than|limit(?:ed)? to|<=|<)\s*[:=]?\s*(\d+)\s*(?:{noun_pattern})",
        rf"(?:{noun_pattern})\s*(?:count\s*)?(?:<=|<|max(?:imum)?|limit(?:ed)?(?:\s+to)?)\s*[:=]?\s*(\d+)",
        rf"\b(\d+)\s*(?:{noun_pattern})\b",
    )
    found: list[int] = []
    for pat in patterns:
        for match in re.finditer(pat, detail, re.I):
            val = int(match.group(1))
            if 0 <= val < ceiling:
                found.append(val)
    return min(found) if found else None


def is_tools_limit_error(detail: str, tool_count: int) -> bool:
    """Return True if an error indicates too many tools rather than zero tool support."""
    if tool_count <= 1 or not detail:
        return False
    if TOOL_LIMIT_RE.search(detail):
        return True
    if "tool" in detail.lower() or "function" in detail.lower():
        cap = extract_limit_number(detail, r"tools?|functions?|declarations?", tool_count)
        return cap is not None and cap >= 1
    return False


def trim_openai_tools(tools: list[dict], detail: str) -> tuple[list[dict], int]:
    """Reduce OpenAI tool list to fit provider cap while prioritizing core tools."""
    total = len(tools)
    cap = extract_limit_number(detail, r"tools?|functions?|declarations?", total)
    target = max(1, cap) if cap and 1 <= cap < total else max(1, total // 2)
    rank = {name: idx for idx, name in enumerate(CORE_TOOL_ORDER)}

    def _name(tool: dict) -> str:
        fn = tool.get("function") if isinstance(tool, dict) else None
        if isinstance(fn, dict) and fn.get("name"):
            return str(fn["name"])
        return str((tool or {}).get("name") or "")

    core = [t for t in tools if _name(t) in rank]
    core.sort(key=lambda t: rank[_name(t)])
    rest = [t for t in tools if _name(t) not in rank]
    trimmed = (core + rest)[:target]
    return trimmed, target


def count_payload_images(payload: dict, *, is_gemini: bool = False) -> int:
    """Count image attachments across OpenAI messages or Gemini inputs."""
    count = 0
    if is_gemini:
        for item in payload.get("input") or []:
            if isinstance(item, dict) and item.get("type") == "image":
                count += 1
        return count
    for msg in payload.get("messages") or []:
        content = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in ("image_url", "input_image", "image"):
                    count += 1
    return count


def rescue_payload_images_limit(
    payload: dict,
    detail: str,
    *,
    is_gemini: bool = False,
    allow_strip_all: bool = False,
) -> tuple[dict, str] | None:
    """Keep newest N images (or remove all if vision unsupported when explicitly allowed)."""
    total = count_payload_images(payload, is_gemini=is_gemini)
    if total == 0:
        return None
    vision_unsupported = bool(re.search(r"does not support images?|vision is not supported|no image|not support image", detail, re.I))
    cap = 0 if vision_unsupported else extract_limit_number(detail, r"images?", total)
    target = 0 if cap == 0 else (cap if cap and 1 <= cap < total else max(1, total // 2))
    if target >= total:
        target = max(0, total - 1)
    if target == 0 and not allow_strip_all:
        return None
    updated = copy.deepcopy(payload)
    drop_count = total - target
    note = "[Ảnh đã ẩn: model không hỗ trợ thị giác]" if target == 0 else f"[Ảnh cũ đã rút gọn để khớp giới hạn {target} ảnh/request]"
    dropped = 0
    if is_gemini:
        new_inputs = []
        for item in updated.get("input") or []:
            if isinstance(item, dict) and item.get("type") == "image" and dropped < drop_count:
                dropped += 1
                new_inputs.append({"type": "text", "text": note})
            else:
                new_inputs.append(item)
        updated["input"] = new_inputs
    else:
        for msg in updated.get("messages") or []:
            content = msg.get("content") if isinstance(msg, dict) else None
            if not isinstance(content, list):
                continue
            new_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") in ("image_url", "input_image", "image") and dropped < drop_count:
                    dropped += 1
                    new_parts.append({"type": "text", "text": note})
                else:
                    new_parts.append(part)
            msg["content"] = new_parts
    desc = f"Rút gọn ảnh trong request: {total} → {target}" if target > 0 else f"Gỡ {total} ảnh do model không hỗ trợ thị giác"
    return updated, desc


def _shrink_b64_image_with_mime(raw_b64: str, max_side: int, quality: int) -> tuple[str, str] | None:
    try:
        raw_bytes = base64.b64decode(raw_b64, validate=False)
        arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        h, w = img.shape[:2]
        scale = min(1.0, max(max_side / max(h, w), min(1.0, 256 / min(h, w))))
        if scale < 1.0:
            img = cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
        has_alpha = img.ndim == 3 and img.shape[2] == 4
        suffix = ".png" if has_alpha else ".jpg"
        options = [cv2.IMWRITE_PNG_COMPRESSION, 9] if has_alpha else [cv2.IMWRITE_JPEG_QUALITY, quality]
        ok, buf = cv2.imencode(suffix, img, options)
        if not ok:
            return None
        shrunk = base64.b64encode(buf.tobytes()).decode("ascii")
        return (shrunk, "image/png" if has_alpha else "image/jpeg") if len(shrunk) < len(raw_b64) else None
    except Exception:
        return None


def _shrink_b64_image(raw_b64: str, max_side: int, quality: int) -> str | None:
    result = _shrink_b64_image_with_mime(raw_b64, max_side, quality)
    return result[0] if result else None


def _image_parts(payload: dict, is_gemini: bool):
    if is_gemini:
        return [part for part in payload.get("input") or []
                if isinstance(part, dict) and part.get("type") == "image" and part.get("data")]
    return [part for msg in payload.get("messages") or [] if isinstance(msg, dict)
            for part in (msg.get("content") if isinstance(msg.get("content"), list) else [])
            if isinstance(part, dict) and part.get("type") == "image_url"]


def downsize_payload_images(payload: dict, attempt: int = 1, *, is_gemini: bool = False,
                            original_payload: dict | None = None) -> tuple[dict, bool]:
    """Recompress and resize embedded base64 images to recover from 413/size errors."""
    sides = (1024, 720, 512)
    qualities = (68, 50, 38)
    idx = min(max(0, attempt - 1), len(sides) - 1)
    max_side, quality = sides[idx], qualities[idx]
    updated = copy.deepcopy(payload)
    changed = False
    if original_payload is not None:
        original_parts = _image_parts(original_payload, is_gemini)
        current_parts = _image_parts(updated, is_gemini)
        if len(original_parts) == len(current_parts):
            for current, original in zip(current_parts, original_parts):
                if is_gemini:
                    current["data"] = original["data"]
                    current["mime_type"] = original.get("mime_type", "image/jpeg")
                else:
                    current["image_url"] = copy.deepcopy(original["image_url"])

    if is_gemini:
        for item in updated.get("input") or []:
            if isinstance(item, dict) and item.get("type") == "image" and item.get("data"):
                shrunk = _shrink_b64_image_with_mime(str(item["data"]), max_side, quality)
                if shrunk:
                    item["data"], item["mime_type"] = shrunk
                    changed = True
        return updated, changed

    for msg in updated.get("messages") or []:
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "image_url":
                continue
            url = str((part.get("image_url") or {}).get("url") or "")
            if not url.startswith("data:image/") or ";base64," not in url:
                continue
            _, _, b64_part = url.partition(";base64,")
            shrunk = _shrink_b64_image_with_mime(b64_part, max_side, quality)
            if shrunk:
                data, mime = shrunk
                part["image_url"]["url"] = f"data:{mime};base64,{data}"
                changed = True
    return updated, changed


def compact_payload_context(payload: dict, *, is_gemini: bool = False) -> tuple[dict, bool]:
    """Shorten older middle messages/tool outputs when context length overflows."""
    updated = copy.deepcopy(payload)
    if is_gemini:
        inputs = updated.get("input") or []
        changed = False
        for item in inputs[:-2]:
            if isinstance(item, dict) and item.get("type") == "text" and len(str(item.get("text") or "")) > 800:
                item["text"] = str(item["text"])[:600] + "\n...[đã rút gọn ngữ cảnh cũ]"
                changed = True
        return updated, changed

    msgs = updated.get("messages") or []
    if len(msgs) <= 2:
        changed = False
        for msg in msgs:
            if isinstance(msg, dict) and isinstance(msg.get("content"), str) and len(msg["content"]) > 4000:
                msg["content"] = msg["content"][:3200] + "\n...[đã rút gọn ngữ cảnh]"
                changed = True
        return updated, changed

    keep_head = 1 if msgs and msgs[0].get("role") in ("system", "developer") else 0
    keep_tail = min(4, max(1, len(msgs) - keep_head))
    middle_end = len(msgs) - keep_tail
    changed = False
    if middle_end > keep_head:
        for msg in msgs[keep_head:middle_end]:
            if isinstance(msg, dict) and isinstance(msg.get("content"), str) and len(msg["content"]) > 300:
                msg["content"] = msg["content"][:220] + "\n...[đã nén lịch sử cũ]"
                changed = True
        if not changed and middle_end - keep_head >= 2:
            cut = keep_head + max(1, (middle_end - keep_head) // 2)
            while cut < middle_end and isinstance(msgs[cut], dict) and msgs[cut].get("role") == "tool":
                cut += 1
            if cut < middle_end:
                updated["messages"] = msgs[:keep_head] + msgs[cut:]
                changed = True
    return updated, changed


def fold_system_messages(payload: dict) -> tuple[dict, bool]:
    """Merge system/developer messages into the first user message."""
    msgs = payload.get("messages") or []
    sys_texts = []
    rest = []
    for msg in msgs:
        if isinstance(msg, dict) and msg.get("role") in ("system", "developer"):
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                sys_texts.append(content.strip())
            continue
        rest.append(copy.deepcopy(msg))
    if not sys_texts:
        return payload, False
    prefix = "\n\n".join(sys_texts)
    updated = copy.deepcopy(payload)
    if rest and isinstance(rest[0], dict) and rest[0].get("role") == "user":
        first_content = rest[0].get("content")
        if isinstance(first_content, str):
            rest[0]["content"] = f"{prefix}\n\n{first_content}"
        elif isinstance(first_content, list):
            rest[0]["content"] = [{"type": "text", "text": prefix}, *first_content]
    else:
        rest.insert(0, {"role": "user", "content": prefix})
    updated["messages"] = rest
    return updated, True


def _append_json_hint(payload: dict, *, is_gemini: bool = False) -> None:
    hint = "\nReturn strictly a valid JSON object without markdown prose."
    if is_gemini:
        inputs = payload.get("input") or []
        if inputs and isinstance(inputs[-1], dict) and inputs[-1].get("type") == "text":
            if "json" not in str(inputs[-1].get("text") or "").lower():
                inputs[-1]["text"] = f"{inputs[-1]['text']}{hint}"
        return
    msgs = payload.get("messages") or []
    for msg in msgs:
        if isinstance(msg, dict) and isinstance(msg.get("content"), str):
            if "json object" not in msg["content"].lower():
                msg["content"] = f"{msg['content']}{hint}"
            return


def strip_rejected_fields(
    payload: dict,
    detail: str,
    *,
    api_base: str = "",
    model: str = "",
    is_gemini: bool = False,
    persist: bool = True,
) -> tuple[dict, list[str]]:
    """Remove optional payload parameters refused by the model/gateway."""
    updated = copy.deepcopy(payload)
    removed: list[str] = []
    lower = detail.lower()
    if is_gemini:
        gen = updated.get("generation_config")
        if isinstance(gen, dict):
            for key in ("thinking_level", "thinking_budget"):
                if key in gen and (key in lower or "thinking" in lower):
                    gen.pop(key, None)
                    removed.append(f"generation_config.{key}")
        if "response_format" in updated and ("response_format" in lower or "schema" in lower):
            updated.pop("response_format", None)
            removed.append("response_format")
            _append_json_hint(updated, is_gemini=True)
        return updated, removed

    for key in OPTIONAL_FIELDS:
        if key in updated and (key in lower or key.replace("_", " ") in lower):
            updated.pop(key, None)
            removed.append(key)
            if key == "response_format":
                _append_json_hint(updated, is_gemini=False)
    if not removed and ("extra" in lower or "unknown" in lower or "unsupported" in lower or "not permitted" in lower):
        for key in OPTIONAL_FIELDS:
            if key in updated:
                updated.pop(key, None)
                removed.append(key)
                if key == "response_format":
                    _append_json_hint(updated, is_gemini=False)
    if persist and removed and api_base and model:
        with _LOCK:
            _bound_dict(_REFUSED_FIELDS, (api_base, model))
            _REFUSED_FIELDS.setdefault((api_base, model), set()).update(removed)
    return updated, removed


def adjust_token_limit(payload: dict, detail: str, *, is_gemini: bool = False) -> tuple[dict, str] | None:
    """Adapt max_tokens / max_completion_tokens / max_output_tokens on token limit errors."""
    updated = copy.deepcopy(payload)
    lower = detail.lower()
    if is_gemini:
        gen = updated.get("generation_config")
        if isinstance(gen, dict) and isinstance(gen.get("max_output_tokens"), int):
            asked = gen["max_output_tokens"]
            named = [int(n.replace(",", "")) for n in re.findall(r"\d[\d,]*", detail)]
            fallback = min(asked // 2, 16384) if asked > 32768 else asked // 2
            lower_val = max((n for n in named if 128 <= n < asked), default=fallback)
            if lower_val < 256:
                gen.pop("max_output_tokens", None)
                return updated, f"Bỏ giới hạn max_output_tokens ({asked})"
            gen["max_output_tokens"] = lower_val
            return updated, f"Giảm max_output_tokens {asked} -> {lower_val}"
        return None

    if "max_tokens" in updated and "max_completion_tokens" in lower and "use" in lower:
        val = updated.pop("max_tokens")
        updated["max_completion_tokens"] = val
        return updated, "Đổi max_tokens -> max_completion_tokens"

    for key in ("max_tokens", "max_completion_tokens"):
        val = updated.get(key)
        if not isinstance(val, int):
            continue
        named = [int(n.replace(",", "")) for n in re.findall(r"\d[\d,]*", detail)]
        fallback = min(val // 2, 16384) if val > 32768 else val // 2
        lower_val = max((n for n in named if 256 <= n < val), default=fallback)
        if lower_val < 1024:
            updated.pop(key, None)
            return updated, f"Bỏ trường {key} ({val})"
        updated[key] = lower_val
        return updated, f"Giảm {key} {val} -> {lower_val}"
    return None


def strip_tools_fallback(payload: dict) -> tuple[dict, bool]:
    """Remove native tools from OpenAI payload and summarize tool specs in system text."""
    tools = payload.get("tools")
    if not tools:
        return payload, False
    updated = copy.deepcopy(payload)
    removed = updated.pop("tools") or []
    updated.pop("tool_choice", None)
    updated.pop("parallel_tool_calls", None)
    names = []
    for item in removed[:30]:
        fn = item.get("function") if isinstance(item, dict) else None
        name = (fn or {}).get("name") or (item or {}).get("name")
        if name:
            names.append(str(name))
    if names:
        note = f"[Lưu ý hệ thống: Model không hỗ trợ native tools; hãy trả lời trực tiếp bằng văn bản/JSON. Công cụ liên quan: {', '.join(names)}]"
        msgs = updated.get("messages") or []
        if msgs and isinstance(msgs[0], dict) and msgs[0].get("role") == "system" and isinstance(msgs[0].get("content"), str):
            msgs[0]["content"] = f"{msgs[0]['content']}\n\n{note}"
        else:
            updated["messages"] = [{"role": "system", "content": note}, *msgs]
    return updated, True


def repair_json_text(raw: str) -> str:
    """Repair common LLM JSON output defects (think blocks, fences, escapes, commas, truncation)."""
    text = str(raw or "").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    if text.startswith("```"):
        text = re.sub(r"^\s*```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?\s*```\s*$", "", text).strip()
    text = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", text)
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text


def repair_json_object(raw: str) -> dict | None:
    """Best-effort repair and parse of a JSON object string; returns None if unrecoverable."""
    text = repair_json_text(raw)
    start = text.find("{")
    if start < 0:
        return None
    candidate = text[start:]
    for attempt_text in (candidate, re.sub(r",\s*([}\]])", r"\1", candidate)):
        try:
            obj, _ = json.JSONDecoder().raw_decode(attempt_text)
            if isinstance(obj, dict):
                return obj
        except ValueError:
            pass
    # Attempt to close unbalanced string / brackets when output was truncated cleanly.
    stack: list[str] = []
    in_str = False
    escaped = False
    for ch in candidate:
        if escaped:
            escaped = False
            continue
        if in_str:
            if ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in ("{", "["):
            stack.append("}" if ch == "{" else "]")
        elif ch in ("}", "]") and stack and stack[-1] == ch:
            stack.pop()
    if stack:
        closed = candidate.rstrip()
        if in_str:
            closed += '"'
        closed = re.sub(r",\s*$", "", closed)
        closed += "".join(reversed(stack))
        closed = re.sub(r",\s*([}\]])", r"\1", closed)
        try:
            obj = json.loads(closed)
            if isinstance(obj, dict):
                return obj
        except ValueError:
            pass
    return None


def classify_http_error(status_code: int, detail: str, payload: dict, *, is_gemini: bool = False) -> str:
    """Classify an HTTP error response into an actionable rescue category."""
    if is_quota_or_rate_limit(status_code, detail):
        return "quota_or_limit"
    if status_code in (401, 403):
        return "auth"
    if status_code == 413 or PAYLOAD_LARGE_RE.search(detail):
        return "payload_too_large"
    if status_code in TRANSIENT_STATUS_CODES:
        return "transient_5xx"
    if status_code in (404, 405):
        if re.search(r"model.*not found|does not exist|no such model", detail, re.I):
            return "model_not_found"
        return "endpoint_404_405"
    if 400 <= status_code < 500:
        tools = payload.get("tools") or []
        if tools and is_tools_limit_error(detail, len(tools)):
            return "too_many_tools"
        if tools and TOOLS_UNSUPPORTED_RE.search(detail):
            return "tools_unsupported"
        if count_payload_images(payload, is_gemini=is_gemini) > 0 and IMAGE_LIMIT_RE.search(detail):
            return "too_many_images"
        if payload.get("stream") and re.search(r"\bstream\b", detail, re.I) and "stream_options" not in detail.lower():
            return "stream_unsupported"
        if SYSTEM_ROLE_RE.search(detail):
            return "system_role_unsupported"
        if CONTEXT_OVERFLOW_RE.search(detail) or TOKEN_CAP_RE.search(detail):
            return "token_or_context_limit"
        if any(k in detail.lower() or k.replace("_", " ") in detail.lower() for k in (*OPTIONAL_FIELDS, "thinking")):
            return "unsupported_field"
        return "other_client_error"
    return "other"


def _next_chat_url(provider: AIProvider | None, tried_urls: set[str], *, allow_local: bool = False) -> str | None:
    if provider is None or provider.protocol != "openai":
        return None
    base = str(provider.api_base or "").rstrip("/")
    if not base:
        return None
    raw_candidates = [
        *(f"{b.rstrip('/')}/chat/completions" for b in api_base_candidates(base)),
        f"{base}/chat/completions",
        f"{base}/v1/chat/completions",
    ]
    seen: set[str] = set()
    for candidate in raw_candidates:
        if not candidate or candidate in seen or candidate in tried_urls:
            continue
        seen.add(candidate)
        if not allow_local:
            try:
                validate_url(candidate)
            except Exception:
                continue
        return candidate
    return None


def proxy_post(
    *,
    url: str,
    headers: dict,
    payload: dict,
    provider: AIProvider | None = None,
    api_key: str = "",
    caller: str = "app",
    timeout=(15, 180),
    stream: bool = False,
    is_gemini: bool = False,
    allow_tool_strip: bool = True,
    raise_on_image_split: bool = False,
    allow_image_strip: bool = False,
    post_fn: Callable | None = None,
    allow_local: bool = False,
) -> tuple[requests.Response, dict, dict]:
    """Send a model API request with full error listening, 429/quota skipping, and auto-rescue.

    Returns ``(response, final_payload, meta)`` where ``meta`` holds latency, errors seen, and rescues.
    """
    if not allow_local:
        validate_url(url)
    sender = post_fn or (
        requests.post
        if (allow_local or requests.post is not _ORIGINAL_REQUESTS_POST)
        else _public_post
    )
    started = time.monotonic()
    provider_id = provider.id if provider else ""
    provider_label = provider.label if provider else (provider_id or urlparse(url).netloc or url)
    api_base = _base_key(str(provider.api_base or "") if provider else "", url)
    model_name = str(payload.get("model") or "")

    current_payload, current_stream, pre_applied, wrapped_chat = apply_wrapped_rules(
        payload,
        api_base=api_base,
        model=model_name,
        stream=stream,
        is_gemini=is_gemini,
        apply_image_cap=not raise_on_image_split,
        allow_image_strip=allow_image_strip,
    )
    cached_url = None
    if not wrapped_chat and api_base and not is_gemini:
        with _LOCK:
            candidate = _FOUND_CHAT_URLS.get(api_base)
        if candidate:
            if allow_local:
                cached_url = candidate
            else:
                try:
                    validate_url(candidate)
                    cached_url = candidate
                except Exception:
                    cached_url = None
    current_url = wrapped_chat or cached_url or url

    errors_seen: list[dict] = []
    rescues_applied: list[str] = list(pre_applied)
    rescue_types: list[str] = ["auto_wrapped"] if pre_applied else []

    tried_urls = {current_url}
    downsize_attempts = 0
    transient_attempts = 0
    pending_wraps: list[dict] = []
    pending_refused: set[str] = set()
    pending_found_url: str | None = None

    for attempt in range(1, MAX_RESCUES_PER_CALL + 2):
        response = None
        try:
            post_kwargs = {
                "headers": headers,
                "json": current_payload,
                "timeout": timeout,
                "allow_redirects": False,
            }
            if current_stream:
                post_kwargs["stream"] = True
            response = sender(current_url, **post_kwargs)
        except requests.RequestException as exc:
            detail = f"{provider_label} {type(exc).__name__}: {redact_secret(str(exc), api_key)[:180]}"
            errors_seen.append({"attempt": attempt, "status": 0, "category": "network_error", "detail": detail})
            if attempt <= MAX_RESCUES_PER_CALL and transient_attempts < 2:
                transient_attempts += 1
                if current_stream:
                    current_stream = False
                    if not is_gemini:
                        current_payload["stream"] = False
                        current_payload.pop("stream_options", None)
                    rescues_applied.append("Tắt stream và thử lại sau lỗi kết nối")
                else:
                    rescues_applied.append(f"Thử lại lần {transient_attempts} sau {type(exc).__name__}")
                rescue_types.append("transient_network")
                time.sleep(min(2.0, 0.6 * transient_attempts))
                continue
            record_event(
                caller=caller, provider=provider_label, model=model_name,
                status="failed", attempts=attempt, latency_ms=round((time.monotonic() - started) * 1000),
                errors_seen=errors_seen, rescues_applied=rescues_applied, rescue_types=rescue_types, detail=detail,
            )
            raise

        raw_status = getattr(response, "status_code", None)
        status_code = raw_status if (isinstance(raw_status, int) and not isinstance(raw_status, bool)) else 0
        if 300 <= status_code < 400:
            detail = f"{provider_label} redirected HTTP {status_code}"
            errors_seen.append({"attempt": attempt, "status": status_code, "category": "redirect", "detail": detail})
            record_event(
                caller=caller, provider=provider_label, model=model_name,
                status="failed", attempts=attempt, latency_ms=round((time.monotonic() - started) * 1000),
                errors_seen=errors_seen, rescues_applied=rescues_applied, rescue_types=rescue_types, detail=detail,
            )
            return response, current_payload, {"errors_seen": errors_seen, "rescues_applied": rescues_applied}

        if is_success_response(response):
            for wkw in pending_wraps:
                err_det = wkw.pop("_error_detail", "")
                res_note = wkw.pop("_rescue_note", "")
                wrap_target(
                    api_base,
                    model_name,
                    url=current_url,
                    provider_id=provider_id,
                    provider_label=provider_label,
                    api_key=api_key,
                    error_detail=err_det,
                    rescue_note=res_note,
                    **wkw,
                )
            if pending_refused and api_base and model_name:
                with _LOCK:
                    _bound_dict(_REFUSED_FIELDS, (api_base, model_name))
                    _REFUSED_FIELDS.setdefault((api_base, model_name), set()).update(pending_refused)
            if pending_found_url and api_base and not is_gemini:
                ok_url = allow_local
                if not ok_url:
                    try:
                        validate_url(pending_found_url)
                        ok_url = True
                    except Exception:
                        ok_url = False
                if ok_url:
                    with _LOCK:
                        _bound_dict(_FOUND_CHAT_URLS, api_base)
                        _FOUND_CHAT_URLS[api_base] = pending_found_url
            final_status = "rescued" if (rescues_applied and len(errors_seen) > 0) or pre_applied else "ok"
            meta = {"errors_seen": errors_seen, "rescues_applied": rescues_applied, "attempts": attempt}
            record_event(
                caller=caller, provider=provider_label, model=model_name,
                status=final_status, attempts=attempt, latency_ms=round((time.monotonic() - started) * 1000),
                errors_seen=errors_seen, rescues_applied=rescues_applied, rescue_types=rescue_types,
            )
            return response, current_payload, meta

        detail = _safe_error_detail(response, api_key)
        category = classify_http_error(status_code, detail, current_payload, is_gemini=is_gemini)
        errors_seen.append({"attempt": attempt, "status": status_code, "category": category, "detail": detail[:240]})

        if category == "quota_or_limit":
            rescues_applied.append("Bỏ qua 429 / hết quota")
            record_event(
                caller=caller, provider=provider_label, model=model_name,
                status="skipped_quota", attempts=attempt, latency_ms=round((time.monotonic() - started) * 1000),
                errors_seen=errors_seen, rescues_applied=rescues_applied, detail=detail,
            )
            return response, current_payload, {"errors_seen": errors_seen, "rescues_applied": rescues_applied, "skipped_quota": True}

        if attempt > MAX_RESCUES_PER_CALL or category == "auth":
            record_event(
                caller=caller, provider=provider_label, model=model_name,
                status="failed", attempts=attempt, latency_ms=round((time.monotonic() - started) * 1000),
                errors_seen=errors_seen, rescues_applied=rescues_applied, rescue_types=rescue_types, detail=detail,
            )
            return response, current_payload, {"errors_seen": errors_seen, "rescues_applied": rescues_applied}

        rescued_desc = ""
        wrap_kwargs: dict = {}
        if category == "too_many_tools" and isinstance(current_payload.get("tools"), list) and len(current_payload["tools"]) > 1:
            before = len(current_payload["tools"])
            trimmed, target = trim_openai_tools(current_payload["tools"], detail)
            current_payload["tools"] = trimmed
            rescued_desc = f"Cắt tools {before} → {target}"
            rescue_types.append("too_many_tools")
            wrap_kwargs["tool_cap"] = target
        elif category == "tools_unsupported" and allow_tool_strip:
            before = len(current_payload.get("tools") or [])
            current_payload, did = strip_tools_fallback(current_payload)
            if did:
                rescued_desc = f"Chuyển {before} tools sang mô tả văn bản"
                rescue_types.append("tools_unsupported")
        elif category == "too_many_images":
            img_count = count_payload_images(current_payload, is_gemini=is_gemini)
            cap = extract_limit_number(detail, r"images?", img_count)
            effective_cap = cap if cap is not None and 0 <= cap < img_count else max(1, img_count // 2)
            if allow_image_strip:
                wrap_kwargs["image_cap"] = effective_cap
            if raise_on_image_split and img_count > 1 and (cap is None or cap >= 1):
                if hasattr(response, "close"):
                    response.close()
                note = f"Chia nhỏ batch {img_count} ảnh (cap={cap or 'nửa'})"
                if cap is not None:
                    wrap_target(
                        api_base, model_name, url=current_url, provider_id=provider_id,
                        provider_label=provider_label, api_key=api_key, image_cap=cap,
                        error_detail=detail, rescue_note=note,
                    )
                rescues_applied.append(note)
                rescue_types.append("too_many_images")
                record_event(
                    caller=caller, provider=provider_label, model=model_name,
                    status="rescued", attempts=attempt, latency_ms=round((time.monotonic() - started) * 1000),
                    errors_seen=errors_seen, rescues_applied=rescues_applied, rescue_types=rescue_types, detail=detail,
                )
                raise ImageLimitExceeded(cap, img_count, detail)
            res = rescue_payload_images_limit(
                current_payload,
                detail,
                is_gemini=is_gemini,
                allow_strip_all=True,
            ) if allow_image_strip else None
            if res is not None:
                current_payload, rescued_desc = res
                rescue_types.append("too_many_images")
        elif category == "payload_too_large":
            downsize_attempts += 1
            img_count = count_payload_images(current_payload, is_gemini=is_gemini)
            if raise_on_image_split and img_count > 1 and downsize_attempts > 1:
                if hasattr(response, "close"):
                    response.close()
                note = f"Chia batch {img_count} ảnh do lỗi dung lượng 413"
                split_cap = max(1, img_count // 2)
                rescues_applied.append(note)
                rescue_types.append("payload_too_large")
                record_event(
                    caller=caller, provider=provider_label, model=model_name,
                    status="rescued", attempts=attempt, latency_ms=round((time.monotonic() - started) * 1000),
                    errors_seen=errors_seen, rescues_applied=rescues_applied, rescue_types=rescue_types, detail=detail,
                )
                raise ImageLimitExceeded(split_cap, img_count, detail)
            new_payload, did = downsize_payload_images(current_payload, attempt=downsize_attempts, is_gemini=is_gemini,
                                                      original_payload=payload)
            if did:
                current_payload = new_payload
                rescued_desc = f"Nén & thu nhỏ ảnh lần {downsize_attempts} (HTTP {status_code})"
                rescue_types.append("payload_too_large")
        elif category == "stream_unsupported" and current_stream:
            current_stream = False
            if not is_gemini:
                current_payload["stream"] = False
                current_payload.pop("stream_options", None)
            rescued_desc = "Chuyển stream -> non-stream"
            rescue_types.append("stream_unsupported")
            wrap_kwargs["force_non_stream"] = True
        elif category == "system_role_unsupported" and not is_gemini:
            new_payload, did = fold_system_messages(current_payload)
            if did:
                current_payload = new_payload
                rescued_desc = "Gộp system prompt vào user message"
                rescue_types.append("system_role_unsupported")
                wrap_kwargs["fold_system"] = True
        elif category == "token_or_context_limit":
            is_overflow = bool(CONTEXT_OVERFLOW_RE.search(detail))
            tok_fix = None if is_overflow else adjust_token_limit(current_payload, detail, is_gemini=is_gemini)
            if tok_fix is not None:
                current_payload, rescued_desc = tok_fix
                rescue_types.append("token_limit")
                cap_val = current_payload.get("max_tokens") or current_payload.get("max_completion_tokens")
                if isinstance(cap_val, int):
                    wrap_kwargs["token_cap"] = cap_val
            else:
                new_payload, did = compact_payload_context(current_payload, is_gemini=is_gemini)
                if did:
                    current_payload = new_payload
                    rescued_desc = "Nén lịch sử ngữ cảnh vượt giới hạn"
                    rescue_types.append("context_overflow")
        elif category == "unsupported_field":
            new_payload, removed = strip_rejected_fields(
                current_payload, detail, api_base=api_base, model=model_name, is_gemini=is_gemini, persist=False
            )
            if removed:
                current_payload = new_payload
                rescued_desc = f"Gỡ trường bị từ chối: {', '.join(removed)}"
                rescue_types.append("unsupported_field")
                wrap_kwargs["refused_fields"] = removed
                pending_refused.update(removed)
        elif category == "endpoint_404_405" and not is_gemini:
            next_url = _next_chat_url(provider, tried_urls, allow_local=allow_local)
            if next_url:
                tried_urls.add(next_url)
                current_url = next_url
                pending_found_url = next_url
                rescued_desc = f"Đổi endpoint chat -> {next_url}"
                rescue_types.append("endpoint_404_405")
                wrap_kwargs["chat_url"] = next_url
        elif category == "model_not_found" and str(current_payload.get("model") or "").startswith("models/"):
            current_payload["model"] = str(current_payload["model"])[7:]
            rescued_desc = "Chuẩn hóa tên model (bỏ tiền tố models/)"
            rescue_types.append("model_not_found")
        elif category == "transient_5xx" and transient_attempts < 2:
            transient_attempts += 1
            time.sleep(min(2.0, 0.7 * transient_attempts))
            rescued_desc = f"Thử lại lỗi HTTP {status_code} (lần {transient_attempts})"
            rescue_types.append("transient_5xx")
        if rescued_desc:
            rescues_applied.append(rescued_desc)
            if wrap_kwargs:
                pending_wraps.append({
                    "_error_detail": detail,
                    "_rescue_note": rescued_desc,
                    **wrap_kwargs,
                })
            logger.info("API Proxy rescue [{} {}]: {} -> {}", provider_label, model_name, detail[:120], rescued_desc)
            if hasattr(response, "close"):
                response.close()
            continue

        # No applicable rescue remained for this response.
        record_event(
            caller=caller, provider=provider_label, model=model_name,
            status="failed", attempts=attempt, latency_ms=round((time.monotonic() - started) * 1000),
            errors_seen=errors_seen, rescues_applied=rescues_applied, rescue_types=rescue_types, detail=detail,
        )
        return response, current_payload, {"errors_seen": errors_seen, "rescues_applied": rescues_applied}

    raise RuntimeError("Proxy rescue loop exhausted unexpectedly")
