"""Curated lettering fonts and their usage notes, read from app/static/fonts/guide/*.txt."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from app.config import DEFAULT_FONT

GUIDE_DIR = DEFAULT_FONT.parent / "guide"
DEFAULT_LETTERING_FONT = "dialogue.mac-dinh-3"


@dataclass(frozen=True)
class FontGuide:
    id: str
    look: str
    use: str
    avoid: str
    roles: tuple[str, ...]
    example: str


def _parse(text: str) -> FontGuide:
    fields = {}
    for line in text.splitlines():
        key, _, value = line.partition(":")
        fields[key.strip()] = value.strip()
    roles = tuple(role.strip() for role in fields.get("default for", "").split(",") if role.strip())
    return FontGuide(fields["id"], fields.get("look", ""), fields.get("use for", ""), fields.get("avoid", ""),
                     roles, fields.get("example", ""))


@lru_cache(maxsize=1)
def load_font_guides() -> tuple[FontGuide, ...]:
    return tuple(_parse(path.read_text(encoding="utf-8")) for path in sorted(GUIDE_DIR.glob("*.txt")))


def font_guide_prompt() -> str:
    """One line per curated font for the translation prompt."""
    return "\n".join(
        f'- {guide.id}: {guide.look} Use for {guide.use} Avoid {guide.avoid} Example: "{guide.example}"'
        + (f" Default for {', '.join(guide.roles)}." if guide.roles else "")
        for guide in load_font_guides()
    )


def lettering_font(choice: str | None, role: str | None) -> str:
    """The model's pick when it is a curated font, else the role's default."""
    guides = load_font_guides()
    if choice in {guide.id for guide in guides}:
        return str(choice)
    return next((guide.id for guide in guides if role in guide.roles), DEFAULT_LETTERING_FONT)
