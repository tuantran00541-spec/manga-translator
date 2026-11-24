from __future__ import annotations

import difflib
from collections import Counter

import numpy as np

from app.ai_mode.vision_json import request_vision_json

MAX_NAMES = 40
MAX_TERMS = 40
MAX_ADDRESS = 30
MAX_FIELD = 60
SIMILAR = 0.8
MISREAD_VOTES = 2

GLOSSARY_PROMPT = (
    "{source}"
    "GOAL: build a single canonical chapter glossary for consistent {target} translation across all slices.\n"
    "RULES:\n"
    "- names: every person, clan, sect, place, item, and technique name (verify exact source spelling across occurrences). "
    "One entry per entity — never add duplicate entries under alternate spellings.\n"
    "  • Tie-breaker for person names: in modern/western/game/sci-fi stories, keep source spelling (e.g. \"Jinwoo\" -> \"Jinwoo\"); "
    "in murim/cultivation/eastern-historical worlds, use one consistent Hán Việt form for person AND technique names when clear "
    "(e.g. \"Cheonma\" -> \"Thiên Ma\", \"Iron Sword Pavilion\" -> \"Thiết Kiếm Các\"), or keep the exact romanized spelling if no clear Hán Việt exists (never guess inconsistent variants).\n"
    "  • Places, organisations, spells, titles, and signs are always translated.\n"
    "- terms: recurring titles, ranks, realms, and genre terms with one fixed {target} equivalent (e.g. sword=\"kiếm\", blade/saber=\"đao\").\n"
    "- address: for each recurring speaker->listener pair (a group like \"disciples\" counts as one speaker), pick one exact self/other pair "
    "from {basis} (apparent age, gender, rank, hostility/closeness) matching the story's world register.\n"
    "{language}"
    "ANSWER with JSON only: "
    '{{"names":[{{"source":"","target":"","note":"role/who"}}],"terms":[{{"source":"","target":""}}],'
    '"address":[{{"from":"","to":"","self":"","other":""}}]}}'
)
_VIETNAMESE = (
    "Vietnamese register guide:\n"
    "- Murim / cultivation / historical: kiếm-hiệp pairs (ta/ngươi, tại hạ/các hạ, sư phụ/con or đệ tử, sư huynh/sư đệ, đạo hữu, bổn tọa); "
    "Hán Việt realms (Luyện Khí, Trúc Cơ, Kim Đan). Never assign modern 'tôi/bạn/cậu/tớ' in these worlds.\n"
    "- Modern / slice-of-life: tôi/cậu, tớ/cậu, anh/em, chú/cháu, tao/mày (close friends/hostile peers only).\n"
)


def _field(value) -> str:
    return " ".join(str(value or "").split())[:MAX_FIELD]


_IMAGES = "IMAGES: consecutive slices of one manga or manhwa chapter in reading order, labelled SLICE <number>.\n"
_TRANSCRIPT = ("TRANSCRIPT: the lettering of consecutive slices of one manga or manhwa chapter in reading order, "
               "each line as [who] text, where who is how the speaker looks.\n{lines}\n")


def _prompt(target_name: str, target_lang: str, source: str, basis: str) -> str:
    language = _VIETNAMESE if str(target_lang or "").lower() in {"vi", "vie", "vietnamese"} else ""
    return GLOSSARY_PROMPT.format(source=source, target=target_name, language=language, basis=basis)


def read_glossary(provider, model: str, api_key: str, target_name: str, target_lang: str,
                  slices: list[tuple[int, np.ndarray]]) -> tuple[dict, float | None]:
    prompt = _prompt(target_name, target_lang, _IMAGES, "what the images show")
    images = [(f"SLICE {index + 1}", image) for index, image in slices]
    result = request_vision_json(provider, model, api_key, prompt, images, max_tokens=1000000, stage="glossary")
    return result.data, result.estimated_cost_usd


def transcript(lines: dict[int, tuple[tuple[str, str], ...]]) -> str:
    return "\n".join(f"SLICE {index + 1}\n" + "\n".join(f"[{who or 'unknown'}] {text}" for who, text in rows)
                     for index, rows in sorted(lines.items()) if rows)


def read_glossary_text(provider, model: str, api_key: str, target_name: str, target_lang: str,
                       lines: dict[int, tuple[tuple[str, str], ...]]) -> tuple[dict, float | None]:
    source = _TRANSCRIPT.replace("{lines}", transcript(lines))
    prompt = _prompt(target_name, target_lang, source, "how the transcript describes the look")
    result = request_vision_json(provider, model, api_key, prompt, [], max_tokens=1000000, stage="glossary")
    return result.data, result.estimated_cost_usd


def _winner(counter: Counter) -> str:
    return counter.most_common(1)[0][0]


def merge_glossaries(parts: list[dict]) -> dict:
    names: dict[str, dict] = {}
    terms: dict[str, Counter] = {}
    address: dict[tuple[str, str], dict[str, Counter]] = {}
    for part in parts:
        part = part if isinstance(part, dict) else {}
        for entry in part.get("names") or []:
            if isinstance(entry, dict) and (source := _field(entry.get("source"))):
                slot = names.setdefault(source.lower(), {"spelling": Counter(), "target": Counter(), "note": ""})
                slot["spelling"][source] += 1
                slot["target"][_field(entry.get("target")) or source] += 1
                slot["note"] = slot["note"] or _field(entry.get("note"))
        for entry in part.get("terms") or []:
            if isinstance(entry, dict) and (source := _field(entry.get("source"))) and _field(entry.get("target")):
                terms.setdefault(source.lower(), Counter())[_field(entry.get("target"))] += 1
        for entry in part.get("address") or []:
            if not isinstance(entry, dict):
                continue
            pair = (_field(entry.get("from")), _field(entry.get("to")))
            if not all(pair):
                continue
            slot = address.setdefault((pair[0].lower(), pair[1].lower()), {"from": Counter(), "to": Counter(),
                                                                          "self": Counter(), "other": Counter()})
            slot["from"][pair[0]] += 1
            slot["to"][pair[1]] += 1
            for key in ("self", "other"):
                if value := _field(entry.get(key)):
                    slot[key][value] += 1

    votes = {key: sum(slot["spelling"].values()) for key, slot in names.items()}
    misread = {
        key for key in names for other in names
        if key != other and votes[other] >= MISREAD_VOTES * votes[key]
        and difflib.SequenceMatcher(None, key, other).ratio() >= SIMILAR
    }
    ranked = sorted((key for key in names if key not in misread), key=lambda key: -votes[key])
    return {
        "names": [{"source": _winner(names[key]["spelling"]), "target": _winner(names[key]["target"]),
                   **({"note": names[key]["note"]} if names[key]["note"] else {})} for key in ranked[:MAX_NAMES]],
        "terms": [{"source": source, "target": _winner(counter)}
                  for source, counter in sorted(terms.items(), key=lambda item: -sum(item[1].values()))
                  if source not in names or source in misread][:MAX_TERMS],
        "address": [
            {"from": _winner(slot["from"]), "to": _winner(slot["to"]),
             **{key: _winner(slot[key]) for key in ("self", "other") if slot[key]}}
            for slot in sorted(address.values(), key=lambda slot: -sum(slot["from"].values()))[:MAX_ADDRESS]
        ],
    }
