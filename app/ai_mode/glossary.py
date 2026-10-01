"""Read the whole chapter once for names, terms and forms of address, so every slice uses the same ones."""
from __future__ import annotations

import difflib
from collections import Counter

import numpy as np

from app.ai_mode.vision_json import request_vision_json

MAX_NAMES = 40
MAX_TERMS = 40
MAX_ADDRESS = 30
MAX_FIELD = 60
SIMILAR = 0.8  # two spellings this close are one name misread
MISREAD_VOTES = 2  # a spelling read at least this many times as often as a close one wins

GLOSSARY_PROMPT = (
    "IMAGES: consecutive slices of one manga or manhwa chapter in reading order, labelled SLICE <number>.\n"
    "QUESTION: which names, terms and forms of address must every slice translate into {target} the same way?\n"
    "- names: every person, clan, sect, place, item and technique name, spelled exactly as the letters show "
    "(check each occurrence). \"target\" is its {target} form: people keep the source spelling; places, "
    "organisations, spells, techniques, titles and signs are translated (Hán Việt where the genre uses it).\n"
    "- terms: recurring titles, ranks, realms and genre terms, each with the one {target} term to use.\n"
    "- address: for each speaker and listener (a group such as \"disciples\" is one speaker), how the speaker "
    "refers to themselves (self) and to the listener (other), chosen from what the images show of both (apparent "
    "age, gender, rank) and how they relate; one form per pair.\n"
    "{language}"
    "ANSWER with JSON only: "
    '{{"names":[{{"source":"","target":"","note":"who or what"}}],"terms":[{{"source":"","target":""}}],'
    '"address":[{{"from":"","to":"","self":"","other":""}}]}}'
)
_VIETNAMESE = ("Vietnamese: cultivation stories use Hán Việt realms and titles (Luyện Khí, Trúc Cơ, sư phụ, sư huynh, "
               "đạo hữu, bổn tọa, tại hạ); disciples to their master: self \"con\" or \"bọn con\", other \"sư phụ\".\n")


def _field(value) -> str:
    return " ".join(str(value or "").split())[:MAX_FIELD]


def read_glossary(provider, model: str, api_key: str, target_name: str, target_lang: str,
                  slices: list[tuple[int, np.ndarray]]) -> tuple[dict, float | None]:
    """One request over a batch of original slices."""
    language = _VIETNAMESE if str(target_lang or "").lower() in {"vi", "vie", "vietnamese"} else ""
    prompt = GLOSSARY_PROMPT.format(target=target_name, language=language)
    images = [(f"SLICE {index + 1}", image) for index, image in slices]
    result = request_vision_json(provider, model, api_key, prompt, images, max_tokens=3000, stage="glossary")
    return result.data, result.estimated_cost_usd


def _winner(counter: Counter) -> str:
    return counter.most_common(1)[0][0]


def merge_glossaries(parts: list[dict]) -> dict:
    """Majority vote across batches; a rare spelling close to a frequent one is dropped as a misread."""
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
                  for source, counter in sorted(terms.items(), key=lambda item: -sum(item[1].values()))[:MAX_TERMS]],
        "address": [
            {"from": _winner(slot["from"]), "to": _winner(slot["to"]),
             **{key: _winner(slot[key]) for key in ("self", "other") if slot[key]}}
            for slot in sorted(address.values(), key=lambda slot: -sum(slot["from"].values()))[:MAX_ADDRESS]
        ],
    }
