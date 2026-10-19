"""Names, terms, forms of address and characters kept per series, so each new chapter starts from what earlier ones settled."""
from __future__ import annotations

import json
import os
import re
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse

from filelock import FileLock

from app.config import BASE_DIR

SERIES_DIR = BASE_DIR / "data" / "series"
MAX_NAMES = 200
MAX_TERMS = 200
MAX_ADDRESS = 120
MAX_CHARACTERS = 80
MAX_FIELD = 80
MAX_NOTES = 1500
_KEY = re.compile(r"[^a-z0-9]+")
# The last path parts that name one chapter rather than the series.
_CHAPTER_PART = re.compile(r"^(chapter|chap|ch|episode|ep|c)[-_]?\d|^\d+([.-]\d+)?$", re.I)


def series_key(url: str) -> str | None:
    """A file-safe key for the series a chapter URL belongs to, or None when the URL names no series."""
    parsed = urlparse(str(url or "").strip())
    parts = [part for part in parsed.path.split("/") if part]
    while parts and (_CHAPTER_PART.match(parts[-1]) or parts[-1].lower() in {"chapter", "chapters", "episode"}):
        parts.pop()
    # A URL like /chapter/<id> names a chapter without its series.
    if not parsed.hostname or not parts or (len(parts) >= 2 and parts[-2].lower() in {"chapter", "chapters", "episode"}):
        return None
    # A trailing id on the slug ("solo-swordmaster-3ec3b16f") changes between site mirrors, so it is dropped.
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


def save(key: str, data: dict) -> dict:
    path = _path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {**clean(data), "updated_at": time.time()}
    with FileLock(str(path) + ".lock"):
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
    return data


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
    """Only well-formed, bounded entries, one per source name, term or speaker pair."""
    data = data if isinstance(data, dict) else {}
    out = empty()
    seen: set = set()
    for entry in data.get("names") or []:
        if isinstance(entry, dict) and (source := _field(entry.get("source"))) and source.lower() not in seen:
            seen.add(source.lower())
            out["names"].append({"source": source, "target": _field(entry.get("target")) or source,
                                 **({"note": _field(entry["note"], 160)} if _field(entry.get("note"), 160) else {})})
    seen = set()
    for entry in data.get("terms") or []:
        if (isinstance(entry, dict) and (source := _field(entry.get("source"))) and (target := _field(entry.get("target")))
                and source.lower() not in seen):
            seen.add(source.lower())
            out["terms"].append({"source": source, "target": target})
    seen = set()
    for entry in data.get("address") or []:
        if not isinstance(entry, dict):
            continue
        pair = (_field(entry.get("from")), _field(entry.get("to")))
        if all(pair) and (pair[0].lower(), pair[1].lower()) not in seen:
            seen.add((pair[0].lower(), pair[1].lower()))
            out["address"].append({"from": pair[0], "to": pair[1],
                                   **{k: _field(entry[k]) for k in ("self", "other") if _field(entry.get(k))}})
    seen = set()
    for entry in data.get("characters") or []:
        if isinstance(entry, dict) and (name := _field(entry.get("name"))) and name.lower() not in seen:
            seen.add(name.lower())
            out["characters"].append({"name": name, "note": _field(entry.get("note"), 160)})
    out["notes"] = _field(data.get("notes"), MAX_NOTES)
    out["chapters"] = [_field(url, 300) for url in (data.get("chapters") or []) if _field(url, 300)][-200:]
    out["names"], out["terms"] = out["names"][:MAX_NAMES], out["terms"][:MAX_TERMS]
    out["address"], out["characters"] = out["address"][:MAX_ADDRESS], out["characters"][:MAX_CHARACTERS]
    return out


def settled_glossary(series: dict, chapter: dict) -> dict:
    """The chapter's glossary under the series one: what the series settled wins, the chapter adds what is new."""
    merged = clean({key: (series.get(key) or []) + (chapter.get(key) or []) for key in ("names", "terms", "address")})
    return {key: merged[key] for key in ("names", "terms", "address")}


def learn(series: dict, chapter_glossary: dict, characters: list[dict], address: list[dict], url: str,
          notes: str = "") -> dict:
    """The series memory after one more chapter: earlier entries stay as they are, new ones are added."""
    merged = {key: (series.get(key) or []) + (chapter_glossary.get(key) or []) for key in ("names", "terms", "address")}
    merged["address"] = merged["address"] + list(address)
    merged["characters"] = (series.get("characters") or []) + list(characters)
    merged["notes"] = series.get("notes") or notes
    merged["chapters"] = [c for c in (series.get("chapters") or []) if c != url] + ([url] if url else [])
    return clean(merged)


# -- the plain-text form the user edits ------------------------------------------------
def to_text(data: dict) -> str:
    """One entry per line: 'Name = Tên', 'term: source = target', 'A -> B: self / other', 'char: Name | note'."""
    lines = [f"{e['source']} = {e['target']}" + (f"  # {e['note']}" if e.get("note") else "") for e in data["names"]]
    lines += [f"term: {e['source']} = {e['target']}" for e in data["terms"]]
    lines += [f"{e['from']} -> {e['to']}: {e.get('self', '')} / {e.get('other', '')}" for e in data["address"]]
    lines += [f"char: {e['name']} | {e['note']}" for e in data["characters"]]
    return "\n".join(lines)


def from_text(text: str, keep: dict) -> dict:
    """The memory the edited text describes; notes and the chapter list are kept from ``keep``."""
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
