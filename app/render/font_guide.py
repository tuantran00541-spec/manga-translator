"""Curated lettering fonts and their usage notes, read from app/static/fonts/guide/*.txt."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from app.config import DEFAULT_FONT

GUIDE_DIR = DEFAULT_FONT.parent / "guide"
DEFAULT_LETTERING_FONT = "dialogue.mac-dinh-3"
# The font follows what the text is and what it sits in, never the model's taste, so one kind of text always
# looks the same: speech in bubbles, narration in boxes, bold captions on the art, shouts in spiky balloons.
FONT_BY_ROLE = {"system_ui": "skill.exo-2", "skill_name": "skill.kanit", "sfx": "sfx.black-ops-one"}
FONT_BY_CONTAINER = {
    "bubble": {},
    "spiky": {"*": "emphasis.bangers"},
    "box": {"title": "emphasis.anton", "*": "narration.mac-dinh-2"},
    "screen": {"*": "skill.exo-2"},
    "free": {"narration": "emphasis.anton", "title": "emphasis.anton", "free_text": "emphasis.anton",
             "thought": "narration.mac-dinh-2", "shout": "emphasis.bangers", "dark_threat": "emphasis.bangers"},
}


@dataclass(frozen=True)
class FontGuide:
    id: str
    look: str
    use: str
    avoid: str
    default: bool
    example: str


def _parse(text: str) -> FontGuide:
    fields = {}
    for line in text.splitlines():
        key, _, value = line.partition(":")
        fields[key.strip()] = value.strip()
    return FontGuide(fields["id"], fields.get("look", ""), fields.get("use for", ""), fields.get("avoid", ""),
                     bool(fields.get("default for")), fields.get("example", ""))


@lru_cache(maxsize=1)
def load_font_guides() -> tuple[FontGuide, ...]:
    return tuple(_parse(path.read_text(encoding="utf-8")) for path in sorted(GUIDE_DIR.glob("*.txt")))


def font_for(role: str | None, container: str | None) -> str:
    """The lettering font for a text of this role in this container."""
    if role in FONT_BY_ROLE:
        return FONT_BY_ROLE[role]
    fonts = FONT_BY_CONTAINER.get(container or "", {})
    return fonts.get(role or "", fonts.get("*", DEFAULT_LETTERING_FONT))


def lettering_font(choice: str | None) -> str:
    """The model's pick when it is a curated font, else the base font."""
    return str(choice) if choice in {guide.id for guide in load_font_guides()} else DEFAULT_LETTERING_FONT
