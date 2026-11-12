from __future__ import annotations

import statistics

BANDED_ROLES = frozenset({"dialogue", "narration", "thought"})
BAND_LO_FACTOR = 0.65
BAND_HI_FACTOR = 1.5
BAND_MIN_OBJECTS = 5


def _role_of(obj: dict) -> str:
    return str(obj.get("typography_role") or obj.get("role") or "dialogue")


def chapter_size_bands(pages) -> dict[str, tuple[float, float]]:
    caps: dict[str, list[float]] = {}
    for page in pages or []:
        if not isinstance(page, dict):
            continue
        for obj in page.get("text_objects") or []:
            if not isinstance(obj, dict) or obj.get("source_missing"):
                continue
            if not str(obj.get("translation") or "").strip():
                continue
            cap = obj.get("source_cap_px")
            if not isinstance(cap, (int, float)) or cap <= 0:
                continue
            role = _role_of(obj)
            if role not in BANDED_ROLES:
                continue
            caps.setdefault(role, []).append(float(cap))
    bands = {}
    for role, values in caps.items():
        if len(values) < BAND_MIN_OBJECTS:
            continue
        median = statistics.median(values)
        bands[role] = (BAND_LO_FACTOR * median, BAND_HI_FACTOR * median)
    return bands
