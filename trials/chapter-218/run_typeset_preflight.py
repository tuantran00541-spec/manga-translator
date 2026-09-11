#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).with_name("build_typography_plan.py")
spec = importlib.util.spec_from_file_location("chapter218_typography_plan", SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {SCRIPT}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

font_dir = mod.DEFAULT_FONT.parent
all_fonts = [p.name for p in sorted(font_dir.glob("*.[tT][tT][fF]"))]


def merged(preferred: list[str]) -> list[str]:
    out = []
    seen = set()
    for name in [*preferred, *all_fonts]:
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(name)
    return out


# Keep semantic preference order, but permit any bundled font that actually
# contains every glyph needed by that text class (notably Vietnamese + ★).
mod.PRIMARY_FONT_CANDIDATES = merged(mod.PRIMARY_FONT_CANDIDATES)
mod.SYSTEM_FONT_CANDIDATES = merged(mod.SYSTEM_FONT_CANDIDATES)
mod.SKILL_FONT_CANDIDATES = merged(mod.SKILL_FONT_CANDIDATES)

mod.main()
