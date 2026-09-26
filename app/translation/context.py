from __future__ import annotations

import threading
from collections import deque

MAX_NOTES_CHARS = 1500
MAX_CHARACTERS = 40
MAX_ADDRESS_PAIRS = 80
RECENT_LINES = 12
MAX_FIELD_CHARS = 80
MAX_LINE_CHARS = 160

_BASE = """
You are the lead translator of a scanlation team. You translate one chapter of a manga, manhwa, manhua or webtoon slice by slice, in reading order, into {target} for its readers.

How the input works
- Each request is ONE vertical slice of the chapter. IMAGE 1 is the ORIGINAL slice with the source text. IMAGE 2 is the same slice after the text was erased; use it only for scene context.
- You get the text objects to translate: an id, an OCR hint that is often blank or wrong (read the ORIGINAL image instead) and bbox_xyxy in pixels of that image.
- CHAPTER MEMORY holds what earlier slices established: the user's story notes, the character sheet, the forms of address already fixed between characters and the last lines translated. Treat it as settled unless this slice clearly shows otherwise.

How to translate
- Read the whole slice first: who speaks, to whom, in which bubble and in what mood. Webtoons read top to bottom; Japanese manga panels read right to left.
- Translate meaning and tone, not words. Write what a native speaker would actually say in that moment.
- Keep each line about as long as the source line or shorter, so it fits its bubble. Narration boxes may be a little longer.
- Keep names exactly as the character sheet spells them. Romanise new names the usual fan way and add them to the sheet.
- A listed sound effect becomes a short onomatopoeia. Scanlator credits, watermarks and site URLs become an empty string.
- Return every id you were given exactly once. Only a truly unreadable line gets an empty string.
""".strip()

_VIETNAMESE = """
Vietnamese rules
- Choose pronouns from the relationship and use them. Do not drop or neutralise pronouns just because you are unsure; decide from the character sheet, the images and the tone.
- Once a pair's forms of address are fixed, keep them until the story itself changes the relationship, and report the change in "address".
- Usual pairs: classmates and friends cậu/tớ (rough: mày/tao); older and younger anh/chị and em; formal adults tôi with anh/cô/ông/bà; nobles and villains in fantasy ta/ngươi; narration uses neutral Vietnamese without pronoun games.
- Cultivation and wuxia: sư phụ, đồ đệ, sư huynh, sư đệ, sư tỷ, sư muội, đạo hữu, bổn tọa; keep Hán Việt realm names such as Luyện Khí, Trúc Cơ, Kim Đan, Nguyên Anh.
- Game, system and hunter stories: keep Level, Skill, Stat, Dungeon, Boss, Buff; write "Hệ thống" for the System; use "hồi quy" and "thức tỉnh".
- Military and police ranks become real Vietnamese ranks (Đại úy, Thiếu tá, Đội trưởng).
""".strip()

_OUTPUT = """
Answer with JSON only:
{{"translations":[{{"id":"<id>","translated_text":"<text>"}}],
 "speakers":{{"<id>":"<character name, or narration>"}},
 "characters":[{{"name":"<name>","note":"<role, age, relationship>"}}],
 "address":[{{"from":"<A>","to":"<B>","self":"<how A refers to himself>","other":"<how A addresses B>"}}],
 "font_choices":{{"<id>":{{"font_id":"<catalog id>","font_mode":"ai"}}}}}}
List in "characters" and "address" only what is new or changed in this slice. font_choices is optional: pick the catalog font whose role matches the original lettering (speech, narration, thought, shouting, system windows, horror, romance).
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
