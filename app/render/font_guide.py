"""The lettering font for each kind of text."""
from __future__ import annotations

DEFAULT_LETTERING_FONT = "dialogue.mac-dinh-3"
# The font follows what the text is and what it sits in, never the model's taste, with the dialogue font as
# the fallback where a container has no entry for a role: every bubble role letters in the dialogue font,
# since the bubble map is empty, so one role can letter differently in different containers.
FONT_BY_ROLE = {"system_ui": "skill.exo-2", "skill_name": "skill.kanit"}
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


def font_for(role: str | None, container: str | None, target_lang: str = "vi") -> str:
    """The lettering font for a text of this role in this container, in a font that has the target's letters."""
    fonts = FONT_BY_CONTAINER.get(container or "", {})
    font = FONT_BY_ROLE.get(role or "") or fonts.get(role or "", fonts.get("*", DEFAULT_LETTERING_FONT))
    return font if str(target_lang or "vi").lower() in BASE_FONT_LANGS else FULL_LATIN.get(font, font)
