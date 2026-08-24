from __future__ import annotations

import copy

from app.visual_qc.regions import qc_cache_identity, qc_cache_matches

CACHE_FIELD = "visual_qc_cache"


def _normalize_file_revision(value: object) -> tuple[int, int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        return tuple(int(v) for v in value)
    except (TypeError, ValueError):
        return None


def store_page_qc_cache(page: dict, *, model: str, mode: str, clean_file_revision: tuple[int, int, int], result: dict) -> None:
    revision = _normalize_file_revision(clean_file_revision)
    if revision is None:
        raise ValueError("clean_file_revision must be a 3-value file identity")
    cache = page.setdefault(CACHE_FIELD, {})
    if not isinstance(cache, dict):
        cache = {}
        page[CACHE_FIELD] = cache
    cache[str(mode)] = {
        "identity": qc_cache_identity(page, model=model, mode=mode),
        "clean_file_revision": list(revision),
        "result": copy.deepcopy(result),
    }


def load_page_qc_cache(page: dict, *, model: str, mode: str, clean_file_revision: tuple[int, int, int]) -> dict | None:
    revision = _normalize_file_revision(clean_file_revision)
    if revision is None:
        return None
    cache = page.get(CACHE_FIELD)
    if not isinstance(cache, dict):
        return None
    entry = cache.get(str(mode))
    if not isinstance(entry, dict):
        return None
    if not qc_cache_matches(entry.get("identity"), page, model=model, mode=mode):
        return None
    if _normalize_file_revision(entry.get("clean_file_revision")) != revision:
        return None
    result = entry.get("result")
    return copy.deepcopy(result) if isinstance(result, dict) else None
