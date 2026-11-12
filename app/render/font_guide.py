from __future__ import annotations

DEFAULT_LETTERING_FONT = "dialogue.mac-dinh-3"
FONT_BY_ROLE = {"system_ui": "skill.exo-2", "skill_name": "skill.kanit"}
FONT_BY_CONTAINER = {
    "bubble": {},
    "spiky": {"*": "emphasis.bangers"},
    "box": {"title": "emphasis.anton"},
    "screen": {"*": "skill.exo-2"},
    "free": {"narration": "emphasis.anton", "title": "emphasis.anton", "free_text": "emphasis.anton",
             "thought": "narration.mac-dinh-2", "shout": "emphasis.bangers", "dark_threat": "emphasis.bangers"},
}
BASE_FONT_LANGS = frozenset({"vi", "en", "id"})
FULL_LATIN = {"dialogue.mac-dinh-3": "thought.comic-neue", "narration.mac-dinh-2": "thought.comic-neue"}


def font_for(role: str | None, container: str | None, target_lang: str = "vi") -> str:
    fonts = FONT_BY_CONTAINER.get(container or "", {})
    font = FONT_BY_ROLE.get(role or "") or fonts.get(role or "", fonts.get("*", DEFAULT_LETTERING_FONT))
    return font if str(target_lang or "vi").lower() in BASE_FONT_LANGS else FULL_LATIN.get(font, font)
