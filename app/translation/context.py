from __future__ import annotations

import json
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
CONTAINERS = frozenset({"bubble", "spiky", "box", "screen", "free"})

_TASK = """
<task>
IMAGE: the raw comic slice. Each text box is outlined in red with its id; its text will be erased and your translation lettered there.
TASK: for each box, read the text inside it and translate it into {target} the way a native reader would say it.
DO: name each box's role and container (the lettering font follows them), keep the line short enough to letter at the raw size, and list lines the image shows outside every box in "unboxed".
ANSWER: the JSON in <output>, nothing else.
</task>

<rules>
- Trust the image over source_text, which is an OCR hint and often wrong.
- Stay within max_chars: it is what fits at the raw size, and longer text is lettered smaller. If a line still cannot fit, set "review": true. Set "enlarge": true when the raw text is too small to read.
- Read the scene (speaker, listener, relationship, emotion, the lines around it) and translate the meaning. Give each character one consistent voice and add nothing the source does not say.
- A sentence often runs on across boxes ("THEY CONQUERED THE WORLD IN THE" then "AGE OF HEROES"), and across slices: recent_lines give each earlier box's source and translation. Translate the whole sentence once, then split it over its boxes in reading order so each box ends at a natural break; never leave a box ending on a dangling word such as "trong", "của" or "và".
- Treat <chapter_glossary> and CHAPTER MEMORY as settled and use their names, terms and forms of address exactly. People's names keep their source spelling unless the setting register says otherwise (murim / cultivation letter names in Hán Việt); other names, titles and signs are translated.
- Copy the source punctuation exactly: every "...", "?!", "!?", "!!" and "—" stays where the source has it, with both marks of "?!". Keep the tone; make jokes only where the source does.
- List in "unboxed" every line someone says, thinks or narrates (dialogue, narration, captions, titles, system messages) that the image shows outside every red box; it is erased and translated next. Sound effects drawn into the art, signs that are only scenery and scanlator credits stay out. A sign is only scenery when it is illegible or purely decorative background lettering; a legible sign naming a place, and labels on a map or diagram, are translated — list them in "unboxed" when they have no red box. Boxes are on a 0-1000 grid of the image: "x1","y1" top-left and "x2","y2" bottom-right.
- Leave translated_text empty for scanlator credits, watermarks, URLs and sound effects drawn into the art (role "sfx"). A sound effect lettered as plain text gets a short onomatopoeia.
- Break lines with "\\n" between phrases, with at least two words on each line and every name and word kept whole.
- role is one of dialogue, narration, thought, whisper, shout, dark_threat, system_ui, skill_name, title, free_text, sfx. container is one of bubble, spiky, box, screen, free (lettered on the art). Add "color" (#rrggbb) only for a distinct colour such as a red or glowing title.
</rules>
""".strip()

_VIETNAMESE = """
<vietnamese>
- Forms of address come from the two people. Before their first line, look at speaker and listener in the images (apparent age and gender, clothes, rank) and at how they relate (family, master and disciple, officer and civilian, boss and worker, friends, strangers, enemies), then choose the self/other pair a Vietnamese reader expects from exactly those two, in the register of the setting. In a martial-arts / wuxia / murim / cultivation / historical world (swords, sects, cultivation, robes) use the kiếm-hiệp register: rivals and enemies "ta"/"ngươi", peers "tại hạ"/"các hạ" or "huynh"/"đệ", master to disciple "ta"/"con" or "sư phụ"/"đệ tử", a king to his subjects "trẫm"/"khanh", a subject to a king "thần"/"bệ hạ" — never "bạn"/"tôi"/"cậu"/"tớ" or "tao"/"mày" there, not even when speaking to an AI, a system, or a machine: everyone in these worlds uses the kiếm-hiệp pairs. Modern casual pairs ("bạn"/"tôi"/"cậu", "tớ"/"cậu", "tao"/"mày") belong only to modern-day settings. Examples: a grown police officer to a teenage boy "tôi"/"cậu", the boy back "tôi"/"anh" ("cháu"/"chú" when the man is much older); close friends in a modern story "tớ"/"cậu".
- A pair in the glossary or CHAPTER MEMORY address is settled: use it for every line between those two, and report a new pair, or a change the story makes, in "address". Keep "bạn" for text that addresses the reader (system windows, notices) — except in martial-arts / cultivation worlds, where the reader is addressed as "ngươi", never "bạn". Narration and inner thoughts use the self form the narrator uses in dialogue.
- Write natural speech: drop translationese ("Điều mà tôi muốn nói là…") and filler (thì, là, mà, một cách, những, các, đã/đang/sẽ when the time is clear); prefer the short word (vì, nếu, giờ). A bare subject may go when the scene makes it clear, but a line spoken to someone keeps the word that shows who they are to the speaker ("Ông trả công bao nhiêu?", not "Trả công bao nhiêu?").
- Write every Vietnamese word with its full diacritics, and start each sentence with a capital letter.
- One name per character, one reading per term: never add a second glossary/character entry for the same person or technique under another language or spelling — reuse the settled entry. Keep sword="kiếm", saber/blade="đao" consistent, and a single translation per source term across the chapter.
- Titles, skill names and free text must stay legible: when the source art lettering is small, letter them at a readable display size (set enlarge), not at the tiny source size.
- Setting register for this chapter: @@REGISTER@@
</vietnamese>
""".strip()

SETTING_REGISTERS = {
    "murim": 'Võ lâm / kiếm hiệp / murim: rivals and enemies "ta"/"ngươi", peers "tại hạ"/"các hạ" or "huynh"/"đệ", sects "sư phụ"/"đệ tử", "sư huynh"/"sư đệ", "đạo hữu", "bổn tọa"; never "bạn"/"tôi"/"cậu"/"tớ" or "tao"/"mày", not even to an AI/system. Letter person names AND technique/skill names in Hán Việt (Sino-Vietnamese), not Latin/English (e.g. "Thiên Ma", "Văn Quy", "Thái Kiếm Thần"); keep one Hán Việt reading per name across the chapter.',
    "cultivation": 'Tu tiên / huyền huyễn: the kiếm-hiệp pairs (ta/ngươi, sư phụ/đệ tử, đạo hữu, bổn tọa) plus Hán Việt realms (Luyện Khí, Trúc Cơ, Kim Đan…); immortals "bổn tọa"/"ngươi"; never "bạn"/"tôi"/"cậu"/"tớ" or "tao"/"mày", not even to an AI/system. Letter person names AND technique/skill names in Hán Việt (Sino-Vietnamese), not Latin/English; keep one reading per name across the chapter.',
    "court": 'Cổ trang / cung đình Á Đông: king "trẫm"/"khanh", subject "thần"/"bệ hạ", officials "hạ quan"/"đại nhân", servants "nô tài"/"nô tì".',
    "western": 'Trung cổ / quý tộc phương Tây (knights, lords): nobles "ta"/"ngươi" or "tôi"/"ngài", knight to lord "thần"/"đức ngài"/"bệ hạ", commoner to noble "tôi"/"ngài" or "cháu"/"lão gia".',
    "modern": 'Hiện đại (đời thường, công sở, học đường, y khoa, cảnh sát): "tôi"/"cậu", "tớ"/"cậu", "anh"/"em", "chú"/"cháu", boss–staff "tôi"/"sếp", doctor–patient "tôi"/"bác sĩ"; close friends "tao"/"mày" only here.',
    "game": 'Game / litRPG / hệ thống (tower, dungeon, hồi quy): keep Level, Skill, Stat, Dungeon, Hệ thống, thức tỉnh; players use the register of the world they are in; system windows address the reader as "bạn".',
    "scifi": 'Sci-fi / tận thế / cơ giáp: soldiers "ta"/"ngươi" or "tôi"/"cậu" by tone; AI/system "tôi"/"bạn".',
    "military": 'Quân đội: Vietnamese ranks, "thủ trưởng"/"tôi", "tôi"/"đồng chí", "thưa ngài".',
    "horror": 'Kinh dị / tâm linh: exorcist "thầy"/"con", spirits "ta"/"ngươi", ordinary people "tôi"/"cậu".',
    "family": 'Gia đình / lãng mạn: "anh"/"em", "bố"/"mẹ"/"con", "tớ"/"cậu".',
}
DEFAULT_SETTING = "modern"
_GENERAL_REGISTER = ('match the setting: martial-arts/historical worlds use "ta"/"ngươi" (king "trẫm", subject "thần"/"bệ hạ") '
                     'and letter person and technique names in Hán Việt (Sino-Vietnamese), not Latin/English; '
                     'never "bạn"/"tôi"/"cậu"/"tớ" in martial-arts/historical worlds, not even to an AI/system; '
                     'modern worlds use "tôi"/"cậu" or "tớ"/"cậu" and keep source spelling for people; never mix registers.')


def setting_register(setting: str | None) -> str:
    return SETTING_REGISTERS.get(str(setting or "").strip().lower(), _GENERAL_REGISTER)


def parse_setting(value) -> str | None:
    key = str(value or "").strip().lower()
    return key if key in SETTING_REGISTERS else None


def setting_question() -> str:
    options = "\n".join(f"- {key}: {desc}" for key, desc in SETTING_DESCRIPTIONS.items())
    return ("Look at the page and classify the story's setting. Answer with JSON only: "
            '{"setting":"<one key>"}.\nKeys:\n' + options)


SETTING_DESCRIPTIONS = {
    "murim": "martial-arts / wuxia / murim world (swordsmen, sects and clans fighting with blades and inner energy; no cultivation realms, no immortals, no ascension)",
    "cultivation": "cultivation / xianxia (characters cultivate through realms toward immortality: realms, immortals, dao, ascension)",
    "court": "eastern historical court (king, mandarins, palace)",
    "western": "western medieval / fantasy nobility (knights, lords, castles)",
    "modern": "modern day (city, school, office, hospital, police)",
    "game": "game / litRPG / system (levels, skills, dungeons, tower)",
    "scifi": "sci-fi / apocalypse / mecha (space, machines, future)",
    "military": "army / war (soldiers, ranks, battlefield)",
    "horror": "horror / spiritual (ghosts, exorcists)",
    "family": "family / romance / slice of life",
}

_OUTPUT = """
<output>
JSON only:
{{"translations":[{{"id":"<id>","source":"<the text the image shows in the box>","translated_text":"<text, lines split with \\n>","role":"<role>","container":"<container>","review":false,"enlarge":false,"color":"#rrggbb or omit"}}],
 "speakers":{{"<id>":"<character name, or narration>"}},
 "characters":[{{"name":"<name>","note":"<role, apparent age and gender, relationship>"}}],
 "address":[{{"from":"<A>","to":"<B>","self":"<how A refers to themselves>","other":"<how A addresses B>"}}],
 "unboxed":[{{"x1":0,"y1":0,"x2":0,"y2":0}}]}}
Every id appears once in translations. characters and address hold only what is new or changed in this slice; unboxed is empty when every line has a box.
</output>
""".strip()


def system_prompt(target_name: str, target_lang: str, settled: dict | None = None,
                  setting: str | None = None) -> str:
    parts = [_TASK.format(target=target_name)]
    if str(target_lang or "").lower() in {"vi", "vie", "vietnamese"}:
        parts.append(_VIETNAMESE.replace("@@REGISTER@@", setting_register(setting)))
    parts.append(_OUTPUT.format())
    if settled:
        parts.append("<chapter_glossary>\n" + json.dumps(settled, ensure_ascii=False, separators=(",", ":"))
                     + "\n</chapter_glossary>")
    return "\n\n".join(parts)


_TEXT_TASK = """
<task>
The source texts are given as JSON objects (id + source); there is no image. Translate each into {target} the way a native reader would say it.
DO: translate the meaning, keep each line short, and use the settled glossary and forms of address exactly.
ANSWER: the JSON in <output>, nothing else.
</task>

<rules>
- Translate the meaning. A sentence often runs across boxes: translate the whole sentence once, then split it over its boxes in reading order so each box ends at a natural break; never leave a box ending on a dangling word such as "trong", "của" or "và".
- Treat <chapter_glossary> and CHAPTER MEMORY as settled and use their names, terms and forms of address exactly. People's names keep their source spelling unless the setting register says otherwise (murim / cultivation letter names in Hán Việt); other names, titles and signs are translated.
- Copy the source punctuation exactly: every "...", "?!", "!?\", \"!!\" and \"—\" stays where the source has it. Keep the tone; make jokes only where the source does.
- Break lines with "\\n" between phrases, with at least two words on each line and every name and word kept whole.
</rules>
""".strip()

_TEXT_OUTPUT = """
<output>
JSON only:
{{"translations":[{{"id":"<id>","translated_text":"<text, lines split with \\n>"}}]}}
Every id appears once in translations.
</output>
""".strip()


def text_system_prompt(target_name: str, target_lang: str, settled: dict | None = None,
                       setting: str | None = None) -> str:
    parts = [_TEXT_TASK.format(target=target_name)]
    if str(target_lang or "").lower() in {"vi", "vie", "vietnamese"}:
        parts.append(_VIETNAMESE.replace("@@REGISTER@@", setting_register(setting)))
    parts.append(_TEXT_OUTPUT.format())
    if settled:
        parts.append("<chapter_glossary>\n" + json.dumps(settled, ensure_ascii=False, separators=(",", ":"))
                     + "\n</chapter_glossary>")
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
        self._lock = threading.Lock()

    def settled(self) -> dict:
        return {**({"glossary": self.glossary} if self.glossary else {}), **({"story_notes": self.notes} if self.notes else {})}

    def snapshot(self, settled: bool = True) -> dict:
        with self._lock:
            return {
                **(self.settled() if settled else {}),
                "characters": [{"name": name, "note": note} for name, note in self.characters.items()],
                "address": [{"from": a, "to": b, **terms} for (a, b), terms in self.address.items()],
                "recent_lines": list(self.recent),
            }

    def update(self, slice_number: int, data: dict, translations: dict[str, str], order: list[str]) -> None:
        speakers = data.get("speakers") if isinstance(data.get("speakers"), dict) else {}
        sources = {str(entry.get("id")): entry.get("source") for entry in data.get("translations") or []
                   if isinstance(entry, dict) and isinstance(entry.get("source"), str)}
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
                        **({"source": _clean(sources[item_id], MAX_LINE_CHARS)} if sources.get(item_id) else {}),
                        "text": _clean(text, MAX_LINE_CHARS),
                    })
