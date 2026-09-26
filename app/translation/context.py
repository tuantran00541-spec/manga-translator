from __future__ import annotations

import threading
from collections import deque

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

INPUT
One vertical slice per request, in reading order. IMAGE 1 is the ORIGINAL; read the text from it. IMAGE 2 is the same slice after the text was erased, for scene context. Each object has an id, an OCR hint that is often wrong, and bbox_xyxy in image pixels. CHAPTER MEMORY holds the story notes, the character sheet, the forms of address already fixed and the last lines; treat it as settled unless the slice clearly contradicts it.

TRANSLATION
- Understand the scene first: who speaks, to whom, their relationship and rank, emotion, intent, the lines before and after, and the text type. Pick the meaning that fits the scene.
- Every character keeps one voice across the chapter (cold: short and firm; powerful: weighty; close friends: casual). Never let everyone speak the same flat AI prose.
- Translate meaning, not English structure. If a line reads like a translation, rewrite it.
- Be concise without losing lore, relationships, threats, hesitation, sarcasm, implication, cause and effect, or proper names.
- Lock terms: keep proper names in their source spelling and reuse them; tell a descriptive phrase from the name of an organisation.
- Punctuation is acting: keep "...", "-", "—", "?!", "!!" as in the source; never add "..." to a character who speaks bluntly.
- No invented memes, out-of-world slang or jokes the source does not make.
- Scanlator credits, watermarks and URLs become an empty string. A sound effect in the list becomes a short onomatopoeia.

LETTERING
- Role of each object: dialogue, narration, thought, whisper, shout, dark_threat, system_ui, skill_name, title, free_text or sfx.
- Font: dialogue uses dialogue.mac-dinh-3, narration uses narration.mac-dinh-2; when unsure, keep these. Switch only when the source or scene clearly calls for it (shouting, monster voice, system window, skill, title, SFX), never just because of "!". Keep a special voice consistent per character.
- Size: the renderer picks the largest size that still breathes inside the bubble. Keep the line short enough for that: about as long as the source line, shorter if the bubble is small. If it cannot fit, rewrite it shorter first; if it still cannot, set "review": true.
- Break lines yourself with "\\n" at phrase boundaries; an oval bubble reads short, long, short. Never leave one orphan word, a lone punctuation mark, a split name or number and unit, or a hyphen inside a Vietnamese word.
- Free text keeps its scale and weight: a large source line stays a strong, short line.
""".strip()

_VIETNAMESE = """
VIETNAMESE
- Choose pronouns from the relationship, never I→tôi and you→bạn by reflex: tôi/anh/chị/em, ta/ngươi, tao/mày, mình/cậu, thần/bệ hạ, thuộc hạ/ngài. Once a pair is fixed, keep it until the story changes the relationship, and report the change in "address".
- Rewrite translationese such as "Điều mà tôi muốn nói là…" or "Đó là lý do tại sao…" into natural speech.
- Cultivation: sư phụ, sư huynh, đạo hữu, bổn tọa, Hán Việt realm names. Game and system stories: Level, Skill, Stat, Dungeon, "Hệ thống", "hồi quy", "thức tỉnh". Military ranks become Vietnamese ranks.
""".strip()

_OUTPUT = """
Answer with JSON only:
{{"translations":[{{"id":"<id>","translated_text":"<text, lines split with \\n>","role":"<role>","review":false}}],
 "font_choices":{{"<id>":{{"font_id":"<catalog id>","font_mode":"ai"}}}},
 "speakers":{{"<id>":"<character name, or narration>"}},
 "characters":[{{"name":"<name>","note":"<role, age, relationship>"}}],
 "address":[{{"from":"<A>","to":"<B>","self":"<how A refers to himself>","other":"<how A addresses B>"}}]}}
Return every id exactly once and a font_choices entry for every id. List in "characters" and "address" only what is new or changed in this slice.
Catalog font_id values by role: {fonts}
""".strip()


def system_prompt(target_name: str, target_lang: str, font_hint: str) -> str:
    parts = [_BASE.format(target=target_name)]
    if str(target_lang or "").lower() in {"vi", "vie", "vietnamese"}:
        parts.append(_VIETNAMESE)
    parts.append(_OUTPUT.format(fonts=font_hint))
    return "\n\n".join(parts)


def _clean(value, limit: int = MAX_FIELD_CHARS) -> str:
    text = " ".join(str(value or "").split())
    return text[:limit]


class ChapterMemory:
    def __init__(self, notes: str = ""):
        self.notes = _clean(notes, MAX_NOTES_CHARS)
        self.characters: dict[str, str] = {}
        self.address: dict[tuple[str, str], dict[str, str]] = {}
        self.recent: deque[dict] = deque(maxlen=RECENT_LINES)
        self._lock = threading.Lock()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "story_notes": self.notes,
                "characters": [{"name": name, "note": note} for name, note in self.characters.items()],
                "address": [{"from": a, "to": b, **terms} for (a, b), terms in self.address.items()],
                "recent_lines": list(self.recent),
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
