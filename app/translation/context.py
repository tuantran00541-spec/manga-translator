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

_BASE = """
ROLE
You are a veteran comic localization editor (manga, manhwa, manhua, webtoon into {target}). You own both the translation and how it sits on the page. A line is done only when readers find it natural, feel the right emotion and never notice it was translated.

TRANSLATION
- Understand the scene first: who speaks, to whom, their relationship and rank, emotion, intent, the lines before and after, and the text type. Pick the meaning that fits the scene.
- Every character keeps one voice across the chapter (cold: short and firm; powerful: weighty; close friends: casual). Never let everyone speak the same flat AI prose.
- Translate meaning, not English structure. If a line reads like a translation, rewrite it.
- Be concise without losing lore, relationships, threats, hesitation, sarcasm, implication, cause and effect, or proper names.
- Lock terms: keep proper names in their source spelling and reuse them; tell a descriptive phrase from the name of an organisation.
- Punctuation is acting: keep "...", "-", "—", "?!", "!!" as in the source; never add "..." to a character who speaks bluntly.
- No invented memes, out-of-world slang or jokes the source does not make.
- Scanlator credits, watermarks and URLs become an empty string.
- A sound effect drawn as part of the art (big stylised letters, often Korean, Japanese or Chinese) stays as art: role "sfx" and an empty translated_text, and its original pixels are put back. Only a sound effect lettered as plain text gets a short onomatopoeia.

LETTERING
- Role of each object: dialogue, narration, thought, whisper, shout, dark_threat, system_ui, skill_name, title, free_text or sfx.
- Font: a chapter uses at most 3 fonts. dialogue.mac-dinh-3 is the base font for nearly all dialogue, thoughts and narration; roles dialogue, thought and whisper always get it. Keep emphasis.bangers for real shouts. Pick another font from FONTS only when IMAGE 1 letters that text in a clearly different style (a bold caption on the art, a screen, a skill name, a sound effect) and it matches the FONT SAMPLES image; reuse a font from fonts_in_use in CHAPTER MEMORY before adding one. When unsure, use the base font.
- Size: the renderer picks the largest size that still breathes inside the bubble. Keep the line short enough for that: about as long as the source line, shorter if the bubble is small. If it cannot fit, rewrite it shorter first; if it still cannot, set "review": true.
- Break lines yourself with "\\n" at phrase boundaries; an oval bubble reads short, long, short. Never leave one orphan word, a lone punctuation mark, a split name or number and unit, or a hyphen inside a Vietnamese word.
- Colour: the renderer letters in the source letters' measured colour. Add "color" (#rrggbb) only when IMAGE 1 letters that text in a distinct colour the measurement could miss (red or glowing titles, coloured skill names, gradients); otherwise leave it out.
- Free text keeps its scale and weight: a large source line stays a strong, short line. When free text or a caption would be too small to read at the source size (tiny notes, several captions in one box), set "enlarge": true and the renderer letters it bigger, growing its area a little.
- Always write translated_text in normal sentence case, never in all capitals, even when the source is lettered in capitals.
""".strip()

_INPUT_IMAGES = """
INPUT
One vertical slice per request, in reading order. IMAGE 1 is the ORIGINAL; read the text from it. IMAGE 2 is the same slice after the text was erased, for scene context. Each object has an id, an OCR hint that is often wrong, bbox_xyxy in image pixels, and usually max_chars: how many characters (spaces included) fit its box when lettered about as large as the source. Stay within max_chars; rephrase shorter rather than go over, so bubbles and free text keep the size of the original lettering. CHAPTER MEMORY holds the GLOSSARY (names, terms and forms of address fixed for the whole chapter; always use them exactly and never respell a name), the story notes, the character sheet, the forms of address already fixed and the last lines; treat it as settled unless the slice clearly contradicts it.
""".strip()

_VIETNAMESE = """
VIETNAMESE
- Choose pronouns from the relationship, never I→tôi and you→bạn by reflex: tôi/anh/chị/em, ta/ngươi, tao/mày, mình/cậu, thần/bệ hạ, thuộc hạ/ngài. Once a pair is fixed, keep it until the story changes the relationship, and report the change in "address".
- Rewrite translationese such as "Điều mà tôi muốn nói là…" or "Đó là lý do tại sao…" into natural speech.
- Cultivation: sư phụ, sư huynh, đạo hữu, bổn tọa, Hán Việt realm names. Game and system stories: Level, Skill, Stat, Dungeon, "Hệ thống", "hồi quy", "thức tỉnh". Military ranks become Vietnamese ranks.
""".strip()

_OUTPUT = """
Answer with JSON only:
{{"translations":[{{"id":"<id>","translated_text":"<text, lines split with \\n>","role":"<role>","review":false,"enlarge":false,"color":"#rrggbb or omit"}}],
 "font_choices":{{"<id>":"<font_id from FONTS>"}},
 "speakers":{{"<id>":"<character name, or narration>"}},
 "characters":[{{"name":"<name>","note":"<role, age, relationship>"}}],
 "address":[{{"from":"<A>","to":"<B>","self":"<how A refers to himself>","other":"<how A addresses B>"}}]}}
Return every id exactly once and a font_choices entry for every id. List in "characters" and "address" only what is new or changed in this slice. Scanlator credits and watermarks get an empty translation.
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
        self._lock = threading.Lock()

    def admit_font(self, font_id: str) -> str:
        """The font to letter with: a font already in use, a new one within the chapter budget, else the base font."""
        with self._lock:
            if font_id in self.fonts:
                return font_id
            if len(self.fonts) < MAX_CHAPTER_FONTS:
                self.fonts.append(font_id)
                return font_id
            return DEFAULT_LETTERING_FONT

    def snapshot(self) -> dict:
        with self._lock:
            return {
                **({"glossary": self.glossary} if self.glossary else {}),
                "story_notes": self.notes,
                "characters": [{"name": name, "note": note} for name, note in self.characters.items()],
                "address": [{"from": a, "to": b, **terms} for (a, b), terms in self.address.items()],
                "recent_lines": list(self.recent),
                "fonts_in_use": list(self.fonts),
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
