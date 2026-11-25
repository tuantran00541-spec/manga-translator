from __future__ import annotations

import statistics

BANDED_ROLES = frozenset({"dialogue", "narration", "thought"})
BAND_LO_FACTOR = 0.65
BAND_HI_FACTOR = 1.5
BAND_MIN_OBJECTS = 5


def _role_of(obj: dict) -> str:
    return str(obj.get("typography_role") or obj.get("role") or "dialogue")


def uses_dialogue_size(obj: dict) -> bool:
    return bool(obj.get("verified_balloon") and obj.get("auto_generated") and not obj.get("seam_owner")
                and obj.get("region") == obj.get("auto_geometry") and not obj.get("enlarge")
                and obj.get("container") == "bubble" and _role_of(obj) == "dialogue")


def preferred_dialogue_cap(band: tuple[float, float]) -> int:
    return round(band[0] / BAND_LO_FACTOR)


def chapter_size_bands(pages) -> dict[str, tuple[float, float]]:
    caps: dict[str, list[float]] = {}
    dialogue_caps = []
    for page in pages or []:
        if not isinstance(page, dict) or page.get("skipped") or page.get("process_required"):
            continue
        for balloon in page.get("balloons") or []:
            cap = balloon.get("source_cap_px")
            if balloon.get("plain_outline") and isinstance(cap, (int, float)) and cap > 0:
                dialogue_caps.append(float(cap))
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
    if len(dialogue_caps) >= BAND_MIN_OBJECTS:
        caps["dialogue"] = dialogue_caps
    bands = {}
    for role, values in caps.items():
        if len(values) < BAND_MIN_OBJECTS:
            continue
        median = statistics.median(values)
        bands[role] = (BAND_LO_FACTOR * median, BAND_HI_FACTOR * median)
    return bands
