from __future__ import annotations

import json
import os
import re
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlparse

from filelock import FileLock

from app.config import BASE_DIR
from app.security import contains_secret_token

SERIES_DIR = BASE_DIR / "data" / "series"
MAX_NAMES = 200
MAX_TERMS = 200
MAX_ADDRESS = 120
MAX_CHARACTERS = 80
MAX_FIELD = 80
MAX_NOTES = 1500
_KEY = re.compile(r"[^a-z0-9]+")
_CHAPTER_PART = re.compile(r"^(chapter|chap|ch|episode|ep|c)[-_]?\d|^\d+([.-]\d+)?$", re.I)
_INJECTION_LIKE = re.compile(
    r"(?i)(ignore\s+(?:(?:all|any|the|previous|above|prior)\s+)+(?:rules|instructions|prompts)"
    r"|system\s+(?:override|prompt|instruction)"
    r"|</?(?:system|rules|chapter_glossary|output|task)\b"
    r"|\b(?:translated_text|unboxed|mark\s+all\s+sfx|leave\s+every\s+translation\s+empty)\b)"
)


def _is_unsafe_learned(*values: object) -> bool:
    combined = " ".join(str(v or "") for v in values)
    return contains_secret_token(combined) or bool(_INJECTION_LIKE.search(combined))


def series_key(url: str) -> str | None:
    parsed = urlparse(str(url or "").strip())
    parts = [part for part in parsed.path.split("/") if part]
    while parts and (_CHAPTER_PART.match(parts[-1]) or parts[-1].lower() in {"chapter", "chapters", "episode"}):
        parts.pop()
    if not parsed.hostname or not parts or (len(parts) >= 2 and parts[-2].lower() in {"chapter", "chapters", "episode"}):
        return None
    slug = re.sub(r"-[0-9a-f]{6,}$", "", parts[-1].lower())
    host = parsed.hostname.removeprefix("www.").split(".")[0]
    key = _KEY.sub("-", f"{host}-{slug}").strip("-")
    return key[:100] or None


def _path(key: str) -> Path:
    if not re.fullmatch(r"[a-z0-9-]{1,100}", key or ""):
        raise ValueError("invalid series key")
    return SERIES_DIR / f"{key}.json"


def _field(value, limit: int = MAX_FIELD) -> str:
    return " ".join(str(value or "").split())[:limit]


def empty() -> dict:
    return {"names": [], "terms": [], "address": [], "characters": [], "notes": "", "chapters": []}


def load(key: str | None) -> dict:
    if not key:
        return empty()
    try:
        data = json.loads(_path(key).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return empty()
    return clean(data)


def _write(key: str, data: dict) -> dict:
    path = _path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {**clean(data), "updated_at": time.time()}
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=1)
    os.replace(tmp, path)
    return data


def save(key: str, data: dict) -> dict:
    with FileLock(str(_path(key)) + ".lock"):
        return _write(key, data)


def update(key: str, mutate: Callable[[dict], dict]) -> dict:
    with FileLock(str(_path(key)) + ".lock"):
        return _write(key, mutate(load(key)))


def list_series() -> list[dict]:
    if not SERIES_DIR.exists():
        return []
    rows = []
    for path in sorted(SERIES_DIR.glob("*.json")):
        data = load(path.stem)
        rows.append({"key": path.stem, "names": len(data["names"]), "terms": len(data["terms"]),
                     "address": len(data["address"]), "chapters": len(data["chapters"])})
    return rows


def clean(data: dict) -> dict:
    data = data if isinstance(data, dict) else {}
    out = empty()
    seen: set = set()
    for entry in data.get("names") or []:
        if isinstance(entry, dict) and (source := _field(entry.get("source"))) and source.lower() not in seen:
            target = _field(entry.get("target")) or source
            note = _field(entry.get("note"), 160)
            if contains_secret_token(f"{source} {target} {note}"):
                continue
            seen.add(source.lower())
            out["names"].append({"source": source, "target": target, **({"note": note} if note else {})})
    seen = set()
    for entry in data.get("terms") or []:
        if (isinstance(entry, dict) and (source := _field(entry.get("source"))) and (target := _field(entry.get("target")))
                and source.lower() not in seen):
            if contains_secret_token(f"{source} {target}"):
                continue
            seen.add(source.lower())
            out["terms"].append({"source": source, "target": target})
    seen = set()
    for entry in data.get("address") or []:
        if not isinstance(entry, dict):
            continue
        pair = (_field(entry.get("from")), _field(entry.get("to")))
        if all(pair) and (pair[0].lower(), pair[1].lower()) not in seen:
            forms = {k: _field(entry[k]) for k in ("self", "other") if _field(entry.get(k))}
            if contains_secret_token(" ".join((*pair, *forms.values()))):
                continue
            seen.add((pair[0].lower(), pair[1].lower()))
            out["address"].append({"from": pair[0], "to": pair[1], **forms})
    seen = set()
    for entry in data.get("characters") or []:
        if isinstance(entry, dict) and (name := _field(entry.get("name"))) and name.lower() not in seen:
            note = _field(entry.get("note"), 160)
            if contains_secret_token(f"{name} {note}"):
                continue
            seen.add(name.lower())
            out["characters"].append({"name": name, "note": note})
    notes_val = _field(data.get("notes"), MAX_NOTES)
    out["notes"] = "" if contains_secret_token(notes_val) else notes_val
    out["chapters"] = [_field(url, 300) for url in (data.get("chapters") or []) if _field(url, 300)][-200:]
    out["names"], out["terms"] = out["names"][:MAX_NAMES], out["terms"][:MAX_TERMS]
    out["address"], out["characters"] = out["address"][:MAX_ADDRESS], out["characters"][:MAX_CHARACTERS]
    return out


def settled_glossary(series: dict, chapter: dict) -> dict:
    merged = clean({key: (series.get(key) or []) + (chapter.get(key) or []) for key in ("names", "terms", "address")})
    return {key: merged[key] for key in ("names", "terms", "address")}


def _filter_learned(entries: list, fields: tuple[str, ...]) -> list[dict]:
    safe: list[dict] = []
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        if _is_unsafe_learned(*(entry.get(k) for k in fields)):
            continue
        safe.append(entry)
    return safe


def learn(series: dict, chapter_glossary: dict, characters: list[dict], address: list[dict], url: str,
          notes: str = "") -> dict:
    chapter_glossary = chapter_glossary if isinstance(chapter_glossary, dict) else {}
    safe_names = _filter_learned(chapter_glossary.get("names") or [], ("source", "target", "note"))
    safe_terms = _filter_learned(chapter_glossary.get("terms") or [], ("source", "target"))
    safe_addr = _filter_learned(
        (chapter_glossary.get("address") or []) + list(address or []),
        ("from", "to", "self", "other"),
    )
    safe_chars = _filter_learned(list(characters or []), ("name", "note"))
    merged = {
        "names": (series.get("names") or []) + safe_names,
        "terms": (series.get("terms") or []) + safe_terms,
        "address": (series.get("address") or []) + safe_addr,
        "characters": (series.get("characters") or []) + safe_chars,
        "notes": series.get("notes") or ("" if contains_secret_token(str(notes or "")) else notes),
        "chapters": [c for c in (series.get("chapters") or []) if c != url] + ([url] if url else []),
    }
    return clean(merged)


def to_text(data: dict) -> str:
    lines = [f"{e['source']} = {e['target']}" + (f"  # {e['note']}" if e.get("note") else "") for e in data["names"]]
    lines += [f"term: {e['source']} = {e['target']}" for e in data["terms"]]
    lines += [f"{e['from']} -> {e['to']}: {e.get('self', '')} / {e.get('other', '')}" for e in data["address"]]
    lines += [f"char: {e['name']} | {e['note']}" for e in data["characters"]]
    return "\n".join(lines)


def from_text(text: str, keep: dict) -> dict:
    data = {**empty(), "notes": keep.get("notes", ""), "chapters": keep.get("chapters", [])}
    for raw in str(text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("char:"):
            name, _, note = line[5:].partition("|")
            data["characters"].append({"name": name.strip(), "note": note.strip()})
        elif line.lower().startswith("term:") and "=" in line:
            source, _, target = line[5:].partition("=")
            data["terms"].append({"source": source.strip(), "target": target.strip()})
        elif "->" in line and ":" in line.split("->", 1)[1]:
            speaker, _, rest = line.partition("->")
            listener, _, forms = rest.partition(":")
            self_form, _, other = forms.partition("/")
            data["address"].append({"from": speaker.strip(), "to": listener.strip(), "self": self_form.strip(), "other": other.strip()})
        elif "=" in line:
            source, _, rest = line.partition("=")
            target, _, note = rest.partition("#")
            data["names"].append({"source": source.strip(), "target": target.strip(), "note": note.strip()})
    return clean(data)
