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


def _cache_slot(mode: str, provider: str) -> str:
    provider = str(provider or "gemini").strip().lower()
    return str(mode) if provider == "gemini" else f"{provider}:{mode}"


def store_page_qc_cache(
    page: dict,
    *,
    model: str,
    mode: str,
    clean_file_revision: tuple[int, int, int],
    result: dict,
    provider: str = "gemini",
) -> None:
    revision = _normalize_file_revision(clean_file_revision)
    if revision is None:
        raise ValueError("clean_file_revision must be a 3-value file identity")
    cache = page.setdefault(CACHE_FIELD, {})
    if not isinstance(cache, dict):
        cache = {}
        page[CACHE_FIELD] = cache
    cache[_cache_slot(mode, provider)] = {
        "identity": qc_cache_identity(page, model=model, mode=mode),
        "clean_file_revision": list(revision),
        "result": copy.deepcopy(result),
    }


def load_page_qc_cache(
    page: dict,
    *,
    model: str,
    mode: str,
    clean_file_revision: tuple[int, int, int],
    provider: str = "gemini",
) -> dict | None:
    revision = _normalize_file_revision(clean_file_revision)
    if revision is None:
        return None
    cache = page.get(CACHE_FIELD)
    if not isinstance(cache, dict):
        return None
    entry = cache.get(_cache_slot(mode, provider))
    if not isinstance(entry, dict):
        return None
    if not qc_cache_matches(entry.get("identity"), page, model=model, mode=mode):
        return None
    if _normalize_file_revision(entry.get("clean_file_revision")) != revision:
        return None
    result = entry.get("result")
    return copy.deepcopy(result) if isinstance(result, dict) else None


def _entry_matches(
    entry: object,
    page: dict,
    *,
    model: str,
    mode: str,
    revision: tuple[int, int, int],
    source_revision: tuple[int, int, int] | None = None,
) -> bool:
    if not (
        isinstance(entry, dict)
        and qc_cache_matches(entry.get("identity"), page, model=model, mode=mode)
        and _normalize_file_revision(entry.get("clean_file_revision")) == revision
    ):
        return False
    if source_revision is not None:
        return _normalize_file_revision(entry.get("source_file_revision")) == source_revision
    return True


def store_region_qc_cache(
    page: dict,
    *,
    region_id: str,
    model: str,
    mode: str,
    clean_file_revision: tuple[int, int, int],
    result: dict,
    source_file_revision: tuple[int, int, int] | None = None,
    provider: str = "gemini",
) -> None:
    revision = _normalize_file_revision(clean_file_revision)
    if revision is None:
        raise ValueError("clean_file_revision must be a 3-value file identity")
    source_revision = None
    if source_file_revision is not None:
        source_revision = _normalize_file_revision(source_file_revision)
        if source_revision is None:
            raise ValueError("source_file_revision must be a 3-value file identity")

    cache = page.setdefault(CACHE_FIELD, {})
    if not isinstance(cache, dict):
        cache = {}
        page[CACHE_FIELD] = cache
    slot = _cache_slot(mode, provider)
    entry = cache.get(slot)
    if not _entry_matches(
        entry,
        page,
        model=model,
        mode=mode,
        revision=revision,
        source_revision=source_revision,
    ):
        entry = {
            "identity": qc_cache_identity(page, model=model, mode=mode),
            "clean_file_revision": list(revision),
            "regions": {},
        }
        if source_revision is not None:
            entry["source_file_revision"] = list(source_revision)
        cache[slot] = entry
    regions = entry.setdefault("regions", {})
    if not isinstance(regions, dict):
        regions = {}
        entry["regions"] = regions
    regions[str(region_id)] = copy.deepcopy(result)


def load_region_qc_cache(
    page: dict,
    *,
    region_id: str,
    model: str,
    mode: str,
    clean_file_revision: tuple[int, int, int],
    source_file_revision: tuple[int, int, int] | None = None,
    provider: str = "gemini",
) -> dict | None:
    revision = _normalize_file_revision(clean_file_revision)
    if revision is None:
        return None
    source_revision = None
    if source_file_revision is not None:
        source_revision = _normalize_file_revision(source_file_revision)
        if source_revision is None:
            return None

    cache = page.get(CACHE_FIELD)
    if not isinstance(cache, dict):
        return None
    entry = cache.get(_cache_slot(mode, provider))
    if not _entry_matches(
        entry,
        page,
        model=model,
        mode=mode,
        revision=revision,
        source_revision=source_revision,
    ):
        return None
    regions = entry.get("regions")
    if not isinstance(regions, dict):
        return None
    result = regions.get(str(region_id))
    return copy.deepcopy(result) if isinstance(result, dict) else None
