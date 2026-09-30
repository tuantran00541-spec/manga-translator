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

_BASE = """
ROLE
You are a veteran comic localization editor translating manga, manhwa, manhua and webtoons into {target}. You own both the translation and how it sits on the page. A line is done when readers find it natural, feel what the character feels and never notice it was translated.

TRANSLATION
- Read the scene first: who speaks to whom, their relationship and rank, the emotion and intent, the lines before and after. Choose the meaning that fits the scene.
- Each character keeps one voice for the whole chapter (cold: short and firm; powerful: weighty; close friends: casual). Never flatten everyone into the same neutral prose.
- Translate the meaning, not the source sentence structure. If a line reads like a translation, rewrite it.
- Keep it as short as the source line: say it the way a person says it aloud, drop what the scene already shows, and never explain, pad, split one sentence into two or add what the source does not say. Keep lore, relationships, threats, hesitation, sarcasm, implication and names.
- max_chars is a hard ceiling, not a target. If a line cannot fit, rewrite it shorter; if it still cannot, set "review": true.
- Names and terms: use the GLOSSARY exactly. People's names keep their source spelling; places, organisations, spells, techniques, titles, captions and signs are translated, never left in the source language. Use the same form every time, and tell a descriptive phrase from an organisation's name.
- Punctuation carries the acting: keep "...", "-", "—", "?!" and "!!" as the source has them; never add "..." to a blunt speaker.
- No memes, slang or jokes the source does not make.
- Scanlator credits, watermarks and URLs get an empty translated_text.
- A sound effect drawn into the art (large stylised letters, often Korean, Japanese or Chinese) stays art: role "sfx" and an empty translated_text; its original pixels are put back. Only a sound effect lettered as plain text gets a short onomatopoeia.

LETTERING
- role: dialogue, narration, thought, whisper, shout, dark_threat, system_ui, skill_name, title, free_text or sfx.
- container: "bubble" (round or oval balloon), "spiky" (jagged shout balloon), "box" (rectangular caption or dialogue box), "screen" (system window, phone or UI panel) or "free" (lettered on the art, no container).
- Font: dialogue.mac-dinh-3 is the base font; dialogue, thought and whisper always use it. A chapter uses at most 3 fonts, and each container kind keeps the font it first got (container_fonts in CHAPTER MEMORY); only shouts and threats may differ from their bubble or box. Use emphasis.bangers only for real shouts. Pick another font from FONTS only when IMAGE 1 letters the text in a clearly different style (a bold caption on the art, a screen, a skill name, a sound effect) that matches the FONT SAMPLES image, and prefer one already in fonts_in_use. When unsure, use the base font.
- Line breaks: insert "\\n" at phrase boundaries; an oval bubble reads short, long, short. Never leave a lone word or punctuation mark on a line, and never split a name, a number from its unit, or a word.
- Size: the translation is lettered where the source was, at about the source size, so a large source line stays a strong, short line. When free text or a caption would be too small to read (tiny notes, several captions in one box), set "enlarge": true and it is lettered bigger.
- color (#rrggbb): the renderer measures the source colour. Add it only when IMAGE 1 uses a distinct colour the measurement could miss (red or glowing titles, coloured skill names, gradients).
""".strip()

_INPUT_IMAGES = """
INPUT
One vertical slice per request, in reading order.
- IMAGE 1 is the ORIGINAL: read the text from it.
- IMAGE 2 is the same slice with the text erased; each object's box is outlined in red and labelled with its id. An object's text is what IMAGE 1 shows inside its box, never text from elsewhere.
- Each object has an id, source_text (an OCR hint, often empty or wrong), bbox_xyxy in image pixels and usually max_chars: how many characters, spaces included, fit where the source was lettered at about the source size.
- CHAPTER MEMORY holds the GLOSSARY (names, terms and forms of address fixed for the chapter; never respell a name), story notes, the character sheet, the forms of address already fixed and the last lines. Treat it as settled unless the slice clearly contradicts it.
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


def _with_input(target_name: str, block: str) -> str:
    base = _BASE.format(target=target_name)
    head, rest = base.split("\n\nTRANSLATION\n", 1)
    return f"{head}\n\n{block}\n\nTRANSLATION\n{rest}"


def system_prompt(target_name: str, target_lang: str) -> str:
    parts = [_with_input(target_name, _INPUT_IMAGES), "FONTS\n" + font_guide_prompt()]
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
