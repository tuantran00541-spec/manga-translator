"""Chapter-level size bands: one chapter's body lettering should read as one size.

The fit sizes each bubble on its own, from the source lettering measured in
that bubble (``source_cap_px``). The measurement is noisy at the tails — dots
read as 7 px caps, art strokes as 170 px — so a chapter's dialogue spans
nearly 4x (run 3: 28.4–104.9 px at 800 wide) while the source lettering
itself is nearly uniform: the middle half of dialogue caps sits within ±8%
of the median, and rendered size tracks the measured cap (r=0.92) while
ignoring box area (r=-0.03). The band therefore clamps the *measured cap*,
not the finished size, toward the role median: the fit machinery below it
still decides the size, so text that needs to shrink to fit still shrinks,
and nothing is ever forced below readability.

Only the body roles are banded. SFX, titles, shouts and free text set their
size from the art — a sound effect is big because the art is big — so a
chapter median would be the wrong reference for them, and they stay
unbanded.
"""
from __future__ import annotations

import statistics

BANDED_ROLES = frozenset({"dialogue", "narration", "thought"})
BAND_LO_FACTOR = 0.65  # a body cap below 65% of its role median is measurement noise
BAND_HI_FACTOR = 1.5  # ... and above 150% likewise (run 3: only 6 of 191 body caps sit outside)
BAND_MIN_OBJECTS = 5  # fewer caps than this and the median is not a chapter reference


def _role_of(obj: dict) -> str:
    return str(obj.get("typography_role") or obj.get("role") or "dialogue")


def chapter_size_bands(pages) -> dict[str, tuple[float, float]]:
    """Per-role (lo, hi) source-cap band in slice pixels, from the chapter's own objects.

    ``pages`` is the manifest's page list. Only objects that will actually
    render count: a stored translation, a measured cap, not source-missing.
    A role with too few caps gets no band, so small chapters keep the old
    per-bubble sizing exactly.
    """
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


def banded_cap(role: str, cap: float, bands: dict[str, tuple[float, float]] | None) -> float:
    """The cap the fit should use: clamped into the role band, or untouched."""
    if not bands or cap is None:
        return cap
    band = bands.get(str(role or "dialogue"))
    if band is None:
        return cap
    lo, hi = band
    return min(max(float(cap), lo), hi)
