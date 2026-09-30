from __future__ import annotations

import threading
from collections import deque

from app.render.font_guide import DEFAULT_LETTERING_FONT, MAX_CHAPTER_FONTS, font_guide_prompt

MAX_NOTES_CHARS = 1500
MAX_CHARACTERS = 40
MAX_ADDRESS_PAIRS = 80
RECENT_LINES = 12
MAX_FIELD_CHARS = 80
MAX_LINE_CHARS = 160
TYPOGRAPHY_ROLES = frozenset({
    "dialogue", "narration", "thought", "whisper", "shout", "dark_threat",
    "system_ui", "skill_name", "title", "free_text", "sfx",
})
CONTAINERS = frozenset({"bubble", "spiky", "box", "screen", "free"})
LOCKED_CONTAINERS = CONTAINERS - {"free"}  # text on the art varies; every drawn container keeps one font
EMPHASIS_ROLES = frozenset({"shout", "dark_threat", "sfx"})  # a shout in a plain bubble keeps its own font

_TASK = """
TASK
You are a comic localization editor translating into {target}. IMAGE 1 is the raw slice; IMAGE 2 is the same slice cleaned, where the translation will be lettered, with each text object's box outlined in red and labelled with its id. For each box, read the text IMAGE 1 shows there, translate it into {target} as naturally as a native reader would say it, pick the font whose FONT SAMPLES row looks closest to the raw lettering, and keep it short enough to letter at the raw size.

RULES
- source_text is an OCR hint, often wrong: trust IMAGE 1. max_chars is how many characters fit at the raw size, a hard ceiling. If a line cannot fit, shorten it; if it still cannot, set "review": true. If the raw text is too small to read at its size, set "enlarge": true.
- Read the scene: who speaks to whom, their relationship, emotion and the lines around it. Each character keeps one voice; translate the meaning, never the source sentence structure, and add nothing the source does not say.
- CHAPTER MEMORY is settled context: use its GLOSSARY names, terms and forms of address exactly. People's names keep their source spelling; other names, titles and signs are translated.
- Keep the source punctuation ("...", "?!", "!!"); no memes or jokes the source does not make.
- Scanlator credits, watermarks and URLs get an empty translated_text, and so does a sound effect drawn into the art (role "sfx"). A sound effect lettered as plain text gets a short onomatopoeia.
- Font: dialogue.mac-dinh-3 is the base font and always letters dialogue, thought and whisper. A chapter uses at most 3 fonts: prefer fonts_in_use, and each container kind keeps the font in container_fonts. When unsure, use the base font.
- Break lines with "\\n" at phrase boundaries; never leave a lone word on a line or split a name or word.
- role: dialogue, narration, thought, whisper, shout, dark_threat, system_ui, skill_name, title, free_text or sfx. container: bubble, spiky, box, screen, or free (lettered on the art). Add "color" (#rrggbb) only for a distinct colour such as red or glowing titles.
""".strip()

_VIETNAMESE = """
VIETNAMESE
- Choose pronouns from the relationship, never "tôi/bạn" by reflex: tôi/anh/chị/em, ta/ngươi, tao/mày, mình/cậu, thần/bệ hạ, thuộc hạ/ngài. Keep a pair until the story changes the relationship, and report the change in "address".
- Rewrite translationese such as "Điều mà tôi muốn nói là…" or "Đó là lý do tại sao…" into natural speech.
- Cut what speech drops: a subject the scene makes clear, "thì", "là", "mà", "một cách", "những", "các", and "đã/đang/sẽ" when the time is clear. Prefer the short word: "vì" over "bởi vì", "nếu" over "trong trường hợp", "giờ" over "bây giờ thì".
- Genre terms: cultivation uses sư phụ, sư huynh, đạo hữu, bổn tọa and Hán Việt realm names; game and system stories keep Level, Skill, Stat, Dungeon and use "Hệ thống", "hồi quy", "thức tỉnh"; military ranks become Vietnamese ranks.
""".strip()

_OUTPUT = """
Answer with JSON only:
{{"translations":[{{"id":"<id>","translated_text":"<text, lines split with \\n>","role":"<role>","container":"<container>","review":false,"enlarge":false,"color":"#rrggbb or omit"}}],
 "font_choices":{{"<id>":"<font_id from FONTS>"}},
 "speakers":{{"<id>":"<character name, or narration>"}},
 "characters":[{{"name":"<name>","note":"<role, age, relationship>"}}],
 "address":[{{"from":"<A>","to":"<B>","self":"<how A refers to themselves>","other":"<how A addresses B>"}}]}}
Return every id exactly once, with a font_choices entry for each. In "characters" and "address" list only what is new or changed in this slice.
""".strip()


def system_prompt(target_name: str, target_lang: str) -> str:
    parts = [_TASK.format(target=target_name), "FONTS\n" + font_guide_prompt()]
    if str(target_lang or "").lower() in {"vi", "vie", "vietnamese"}:
        parts.append(_VIETNAMESE)
    parts.append(_OUTPUT.format())
    return "\n\n".join(parts)


def _clean(value, limit: int = MAX_FIELD_CHARS) -> str:
    text = " ".join(str(value or "").split())
    return text[:limit]


class ChapterMemory:
    def __init__(self, notes: str = "", glossary: dict | None = None):
        self.notes = _clean(notes, MAX_NOTES_CHARS)
        self.glossary = glossary or {}
        self.characters: dict[str, str] = {}
        self.address: dict[tuple[str, str], dict[str, str]] = {}
        self.recent: deque[dict] = deque(maxlen=RECENT_LINES)
        self.fonts: list[str] = [DEFAULT_LETTERING_FONT]
        self.container_fonts: dict[str, str] = {}
        self._lock = threading.Lock()

    def _admit(self, font_id: str) -> str:
        if font_id in self.fonts:
            return font_id
        if len(self.fonts) < MAX_CHAPTER_FONTS:
            self.fonts.append(font_id)
            return font_id
        return DEFAULT_LETTERING_FONT

    def admit_font(self, font_id: str) -> str:
        """The font to letter with: a font already in use, a new one within the chapter budget, else the base font."""
        with self._lock:
            return self._admit(font_id)

    def container_font(self, container: str | None, role: str | None, font_id: str) -> str:
        """The font a container kind was first lettered in, which it keeps for the chapter; else ``admit_font``."""
        with self._lock:
            if container not in LOCKED_CONTAINERS or (role in EMPHASIS_ROLES and container != "spiky"):
                return self._admit(font_id)
            if container not in self.container_fonts:
                self.container_fonts[container] = self._admit(font_id)
            return self.container_fonts[container]

    def snapshot(self) -> dict:
        with self._lock:
            return {
                **({"glossary": self.glossary} if self.glossary else {}),
                "story_notes": self.notes,
                "characters": [{"name": name, "note": note} for name, note in self.characters.items()],
                "address": [{"from": a, "to": b, **terms} for (a, b), terms in self.address.items()],
                "recent_lines": list(self.recent),
                "fonts_in_use": list(self.fonts),
                "container_fonts": dict(self.container_fonts),
            }

    def update(self, slice_number: int, data: dict, translations: dict[str, str], order: list[str]) -> None:
        speakers = data.get("speakers") if isinstance(data.get("speakers"), dict) else {}
        with self._lock:
            for entry in data.get("characters") or []:
                if not isinstance(entry, dict):
                    continue
                name = _clean(entry.get("name"))
                if not name or (name not in self.characters and len(self.characters) >= MAX_CHARACTERS):
                    continue
                note = _clean(entry.get("note"), MAX_LINE_CHARS)
                self.characters[name] = note or self.characters.get(name, "")
            for entry in data.get("address") or []:
                if not isinstance(entry, dict):
                    continue
                pair = (_clean(entry.get("from")), _clean(entry.get("to")))
                if not all(pair) or (pair not in self.address and len(self.address) >= MAX_ADDRESS_PAIRS):
                    continue
                terms = {key: _clean(entry.get(key)) for key in ("self", "other") if _clean(entry.get(key))}
                if terms:
                    self.address[pair] = {**self.address.get(pair, {}), **terms}
            for item_id in order:
                text = translations.get(item_id)
                if text:
                    self.recent.append({
                        "slice": slice_number,
                        "speaker": _clean(speakers.get(item_id)) or "?",
                        "text": _clean(text, MAX_LINE_CHARS),
                    })
