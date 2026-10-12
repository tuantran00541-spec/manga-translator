"""Model roles: named models on any configured provider that chains can call, each with a daily budget and a fallback."""
from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path

import requests

ROLE_NAME = re.compile(r"[a-z][a-z0-9_-]{0,30}")
USAGE = Path(".manga-agent") / "role-usage.json"
_LOCK = threading.Lock()


def parse(raw) -> dict[str, dict]:
    """The valid roles of a profile's model_roles: a model plus an app provider id, or a base_url with the env variable holding its key."""
    roles = {}
    for name, cfg in (raw.items() if isinstance(raw, dict) else []):
        if not ROLE_NAME.fullmatch(str(name)) or not isinstance(cfg, dict) or not isinstance(cfg.get("model"), str) or not cfg["model"].strip():
            continue
        row = {"model": cfg["model"].strip()[:200], "note": str(cfg.get("note") or "")[:200], "fallback": str(cfg.get("fallback") or "")}
        if isinstance(cfg.get("provider"), str) and cfg["provider"].strip():
            row["provider"] = cfg["provider"].strip()
        elif isinstance(cfg.get("base_url"), str) and isinstance(cfg.get("key_env"), str):
            row["base_url"], row["key_env"] = cfg["base_url"].strip(), cfg["key_env"].strip()
        else:
            continue
        for key in ("daily_requests", "daily_tokens"):
            row[key] = cfg[key] if isinstance(cfg.get(key), int) and cfg[key] > 0 else 0
        roles[str(name)] = row
    return roles


def _read(home: Path) -> dict:
    try:
        data = json.loads((home / USAGE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def today(home: Path) -> dict[str, dict]:
    """Requests and tokens each role used today (local date)."""
    data = _read(home)
    return data.get("days", {}).get(time.strftime("%Y-%m-%d"), {}) if isinstance(data.get("days"), dict) else {}


def within(home: Path, name: str, cfg: dict) -> bool:
    """Whether a role still has today's request and token budget."""
    used = today(home).get(name, {})
    return not ((cfg["daily_requests"] and used.get("requests", 0) >= cfg["daily_requests"]) or
                (cfg["daily_tokens"] and used.get("tokens", 0) >= cfg["daily_tokens"]))


def add(home: Path, name: str, requests_made: int, tokens: int) -> None:
    """Count a role's use toward today's budget; only the last week of days is kept."""
    with _LOCK:
        data = _read(home)
        days = data.get("days") if isinstance(data.get("days"), dict) else {}
        day = days.setdefault(time.strftime("%Y-%m-%d"), {})
        row = day.setdefault(name, {"requests": 0, "tokens": 0})
        row["requests"] += requests_made
        row["tokens"] += tokens
        data["days"] = {k: days[k] for k in sorted(days)[-7:]}
        path = home / USAGE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")


def price(row: dict) -> dict:
    """Dollars per million tokens from a /models row, in whichever form the provider gives them."""
    found = {}
    pricing = row.get("pricing") if isinstance(row.get("pricing"), dict) else {}
    # Per-token prices, as MiniRouter (input/output) and OpenRouter (prompt/completion) give them.
    for ours, names in (("in", ("input", "prompt")), ("out", ("output", "completion")), ("cached", ("input_cache_read",))):
        value = next((pricing[n] for n in names if pricing.get(n) not in (None, "")), None)
        try:
            if value is not None:
                found[ours] = round(float(value) * 1e6, 4)
        except (TypeError, ValueError):
            pass
    for ours, theirs in (("in", "input_cost_per_million_tokens_usd"), ("out", "output_cost_per_million_tokens_usd"),
                         ("cached", "cached_input_cost_per_million_tokens_usd")):
        if isinstance(row.get(theirs), (int, float)):
            found[ours] = float(row[theirs])
    return found


def catalog(base_url: str, headers: dict, query: str = "", limit: int = 40) -> list[dict]:
    """A provider's models matching query, with price per million tokens, context and whether they read images, cheapest first."""
    response = requests.get(f"{base_url.rstrip('/')}/models", headers={"Accept": "application/json", **headers}, timeout=30)
    response.raise_for_status()
    data = response.json()
    rows = data.get("data") if isinstance(data, dict) else data
    found = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict) or not row.get("id") or (row.get("type") not in (None, "language", "chat")):
            continue
        model = str(row["id"])
        if query and query.lower() not in f"{model} {row.get('name') or ''}".lower():
            continue
        inputs = (row.get("modalities") or {}).get("input") if isinstance(row.get("modalities"), dict) else None
        found.append({"id": model, "price": price(row), "context": row.get("context_window") or row.get("context_length") or row.get("context_window_tokens"),
                      "vision": bool(inputs and "image" in inputs) or "vision" in (row.get("tags") or [])})
    found.sort(key=lambda r: (r["price"].get("in", 1e9) + r["price"].get("out", 1e9) / 4, r["id"]))
    return found[:limit]


def env_key(cfg: dict) -> str:
    return os.environ.get(cfg.get("key_env") or "", "")
