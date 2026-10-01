"""Curated lettering fonts and their usage notes, read from app/static/fonts/guide/*.txt."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from app.config import DEFAULT_FONT

GUIDE_DIR = DEFAULT_FONT.parent / "guide"
DEFAULT_LETTERING_FONT = "dialogue.mac-dinh-3"
# The font follows what the text is and what it sits in, never the model's taste, so one kind of text always
# looks the same: speech and narration boxes in the comic font, bold captions on the art, shouts in spiky balloons.
FONT_BY_ROLE = {"system_ui": "skill.exo-2", "skill_name": "skill.kanit", "sfx": "sfx.black-ops-one"}
FONT_BY_CONTAINER = {
    "bubble": {},
    "spiky": {"*": "emphasis.bangers"},
    "box": {"title": "emphasis.anton"},
    "screen": {"*": "skill.exo-2"},
    "free": {"narration": "emphasis.anton", "title": "emphasis.anton", "free_text": "emphasis.anton",
             "thought": "narration.mac-dinh-2", "shout": "emphasis.bangers", "dark_threat": "emphasis.bangers"},
}
# The Mặc Định comic fonts carry Vietnamese and English letters only (no ñ, ç, ü, ß, ¿); other Latin languages
# letter that text in Comic Neue, which has them.
BASE_FONT_LANGS = frozenset({"vi", "en", "id"})
FULL_LATIN = {"dialogue.mac-dinh-3": "thought.comic-neue", "narration.mac-dinh-2": "thought.comic-neue"}


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


def font_for(role: str | None, container: str | None, target_lang: str = "vi") -> str:
    """The lettering font for a text of this role in this container, in a font that has the target's letters."""
    fonts = FONT_BY_CONTAINER.get(container or "", {})
    font = FONT_BY_ROLE.get(role or "") or fonts.get(role or "", fonts.get("*", DEFAULT_LETTERING_FONT))
    return font if str(target_lang or "vi").lower() in BASE_FONT_LANGS else FULL_LATIN.get(font, font)


def lettering_font(choice: str | None) -> str:
    """The model's pick when it is a curated font, else the base font."""
    return str(choice) if choice in {guide.id for guide in load_font_guides()} else DEFAULT_LETTERING_FONT
