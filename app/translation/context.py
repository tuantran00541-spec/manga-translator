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
<task>
IMAGE: the raw comic slice. Each text box is outlined in red with its id; its text will be erased and your translation lettered there.
TASK: for each box, read the text inside it and translate it into {target} the way a native reader would say it.
DO: pick the font whose FONT SAMPLES row looks closest to the raw lettering, keep the line short enough to letter at the raw size, and list lines the image shows outside every box in "unboxed".
ANSWER: the JSON in <output>, nothing else.
</task>

<rules>
- Trust the image over source_text, which is an OCR hint and often wrong.
- Stay within max_chars: it is what fits at the raw size, and longer text is lettered smaller. If a line still cannot fit, set "review": true. Set "enlarge": true when the raw text is too small to read.
- Read the scene (speaker, listener, relationship, emotion, the lines around it) and translate the meaning. Give each character one consistent voice and add nothing the source does not say.
- Treat CHAPTER MEMORY as settled and use its glossary names, terms and forms of address exactly. People's names keep their source spelling; other names, titles and signs are translated.
- Copy the source punctuation exactly: every "...", "?!", "!?", "!!" and "—" stays where the source has it, with both marks of "?!". Keep the tone; make jokes only where the source does.
- List in "unboxed" every line someone says, thinks or narrates (dialogue, narration, captions, titles, system messages) that the image shows outside every red box; it is erased and translated next. Sound effects drawn into the art, signs that are only scenery and scanlator credits stay out. Boxes are on a 0-1000 grid of the image: "x1","y1" top-left and "x2","y2" bottom-right.
- Leave translated_text empty for scanlator credits, watermarks, URLs and sound effects drawn into the art (role "sfx"). A sound effect lettered as plain text gets a short onomatopoeia.
- Fonts: dialogue, thought and whisper use the base font dialogue.mac-dinh-3. A chapter uses at most 3 fonts, so reuse fonts_in_use and the container_fonts already set; when unsure, use the base font.
- Break lines with "\\n" between phrases, with at least two words on each line and every name and word kept whole.
- role is one of dialogue, narration, thought, whisper, shout, dark_threat, system_ui, skill_name, title, free_text, sfx. container is one of bubble, spiky, box, screen, free (lettered on the art). Add "color" (#rrggbb) only for a distinct colour such as a red or glowing title.
</rules>
""".strip()

_VIETNAMESE = """
<vietnamese>
- Forms of address come from the two people. Before their first line, look at speaker and listener in the images (apparent age and gender, clothes, rank) and at how they relate (family, master and disciple, officer and civilian, boss and worker, friends, strangers, enemies), then choose the self/other pair a Vietnamese reader expects from exactly those two. Examples: a grown police officer to a teenage boy "tôi"/"cậu", the boy back "tôi"/"anh" ("cháu"/"chú" when the man is much older); a master to a disciple "ta"/"con"; rivals "ta"/"ngươi" or "tao"/"mày"; close friends "tớ"/"cậu" or "tao"/"mày"; a subject to a king "thần"/"bệ hạ".
- A pair in CHAPTER MEMORY address is settled: use it for every line between those two, and report a new pair, or a change the story makes, in "address". Keep "bạn" for text that addresses the reader (system windows, notices). Narration and inner thoughts use the self form the narrator uses in dialogue.
- Write natural speech: drop translationese ("Điều mà tôi muốn nói là…"), a subject the scene makes clear and filler (thì, là, mà, một cách, những, các, đã/đang/sẽ when the time is clear); prefer the short word (vì, nếu, giờ).
- Genre terms: cultivation uses sư phụ, sư huynh, đạo hữu, bổn tọa and Hán Việt realms; game stories keep Level, Skill, Stat, Dungeon with Hệ thống, hồi quy, thức tỉnh; military ranks become Vietnamese ranks.
</vietnamese>
""".strip()

_OUTPUT = """
<output>
JSON only:
{{"translations":[{{"id":"<id>","source":"<the text the image shows in the box>","translated_text":"<text, lines split with \\n>","role":"<role>","container":"<container>","review":false,"enlarge":false,"color":"#rrggbb or omit"}}],
 "font_choices":{{"<id>":"<font_id>"}},
 "speakers":{{"<id>":"<character name, or narration>"}},
 "characters":[{{"name":"<name>","note":"<role, apparent age and gender, relationship>"}}],
 "address":[{{"from":"<A>","to":"<B>","self":"<how A refers to themselves>","other":"<how A addresses B>"}}],
 "unboxed":[{{"x1":0,"y1":0,"x2":0,"y2":0}}]}}
Every id appears once in translations and in font_choices. characters and address hold only what is new or changed in this slice; unboxed is empty when every line has a box.
</output>
""".strip()


def system_prompt(target_name: str, target_lang: str) -> str:
    parts = [_TASK.format(target=target_name), "<fonts>\n" + font_guide_prompt() + "\n</fonts>"]
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
