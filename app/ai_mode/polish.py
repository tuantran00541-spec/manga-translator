from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import requests

from app.ai_mode import decider
from app.ai_mode.vision_json import request_vision_json
from app.parameters import TRANSLATION_CONNECT_TIMEOUT_SECONDS, TRANSLATION_READ_TIMEOUT_SECONDS

JUDGED_ROLES = frozenset({"dialogue", "narration", "thought", "whisper", "shout", "dark_threat", "system_ui"})
MIN_WORDS = 3
REWRITE_BATCH = 8
MAX_STATE_CHARS = 2000
MAX_SETTLED = 6
MIN_GAIN = 0.025
MAX_LOSS = 0.15

QUESTIONS = {
    "natural": {
        "type": "score",
        "instructions": "How natural is TRANSLATION as Vietnamese that a comic reader would say or read?",
        "criteria": ["broken or not real Vietnamese", "stiff, awkward or word-for-word", "understandable but clumsy",
                     "natural"],
    },
    "clear": {
        "type": "score",
        "instructions": "How easily does a Vietnamese reader understand what TRANSLATION means on its own?",
        "criteria": ["makes no sense", "unclear or ambiguous", "clear"],
    },
    "faithful": {
        "type": "score",
        "instructions": "How well does TRANSLATION keep what SOURCE says?",
        "criteria": ["says something else", "drops or adds part of the meaning", "keeps the meaning"],
    },
}
SETTLED = {
    "type": "score",
    "instructions": "Does TRANSLATION use the forms of address in ADDRESS and the terms in GLOSSARY wherever they apply?",
    "criteria": ["uses other forms or terms", "uses some of them", "uses them all"],
}
LIMITS = {"natural": 2.0, "clear": 1.6, "faithful": 1.5, "settled": 1.0}
PROBLEMS = {
    "natural": "it sounds unnatural or word-for-word in Vietnamese",
    "clear": "its meaning is unclear",
    "faithful": "it does not keep the source meaning",
    "settled": "it does not use the chapter's forms of address or glossary terms",
}

def in_language(value, language: str):
    if isinstance(value, str):
        return value.replace("Vietnamese", language)
    if isinstance(value, dict):
        return {key: in_language(item, language) for key, item in value.items()}
    if isinstance(value, list):
        return [in_language(item, language) for item in value]
    return value


REWRITE_PROMPT = (
    "INPUT: lines of a Vietnamese comic translation. Each item has the speaker when known, the SOURCE line, the "
    "current TRANSLATION, the lines before and after it, and the problem a reviewer found.\n"
    "TASK: rewrite each TRANSLATION so it says exactly what SOURCE says, as natural Vietnamese a reader of this "
    "comic expects.\n"
    "KEEP: the names, terms and forms of address in CHAPTER MEMORY; the source punctuation (every \"...\", "
    "\"?!\", \"!!\" and \"—\"); sentence case; the line breaks (\\n between phrases, as many lines as the current "
    "translation); max_chars when given. If the reviewer is wrong, return the translation unchanged.\n"
    'ANSWER with JSON only: {"rewrites":[{"id":"<id>","text":"<the rewritten line>"}]}'
)


@dataclass
class Line:
    page_index: int
    id: str
    source: str
    text: str
    before: str = ""
    after: str = ""
    max_chars: int | None = None
    speaker: str = ""
    scores: dict[str, float] = field(default_factory=dict)


def collect_lines(pages: list[dict], indices: list[int]) -> list[Line]:
    ordered = []
    for page_index in indices:
        page = pages[page_index]
        for obj in page.get("text_objects") or []:
            text = str(obj.get("translation") or "").strip() if isinstance(obj, dict) else ""
            if not text or obj.get("joined_into") or obj.get("overlap_dropped") or obj.get("source_missing"):
                continue
            ordered.append((page_index, obj, text))
    lines = []
    for n, (page_index, obj, text) in enumerate(ordered):
        source = str(obj.get("source_read") or "").strip()
        if (obj.get("typography_role") or "dialogue") not in JUDGED_ROLES or not source or len(text.split()) < MIN_WORDS:
            continue
        lines.append(Line(
            page_index, str(obj["id"]), source, text,
            before=ordered[n - 1][2] if n else "", after=ordered[n + 1][2] if n + 1 < len(ordered) else "",
            max_chars=obj.get("max_chars") if isinstance(obj.get("max_chars"), int) else None,
            speaker=str(obj.get("speaker") or "").strip(),
        ))
    return lines


def _flat(text: str) -> str:
    return " ".join(str(text or "").split())


_FIRST_PERSON = re.compile(r"\b(i|i'm|i've|i'll|i'd|me|my|mine|myself|we|we're|we've|we'll|us|our|ours|ourselves)\b")
_SECOND_PERSON = re.compile(r"\b(you|you're|you've|you'll|you'd|your|yours|yourself|yourselves|ya)\b")


def settled(line: Line, memory: dict | None) -> tuple[list[str], list[str]]:
    memory = memory or {}
    glossary = memory.get("glossary") or {}
    pairs = {}
    for entry in [*(glossary.get("address") or []), *(memory.get("address") or [])]:
        if isinstance(entry, dict) and line.speaker and _flat(entry.get("from")).lower() == line.speaker.lower():
            pairs[_flat(entry.get("to"))] = entry
    source = _flat(line.source).lower()
    english = source.isascii()
    selves = {_flat(entry.get("self")) for entry in pairs.values() if _flat(entry.get("self"))}
    address = [f"self \"{selves.pop()}\""] if len(selves) == 1 and (not english or _FIRST_PERSON.search(source)) else []
    if (len(pairs) == 1 and (other := _flat(next(iter(pairs.values())).get("other")))
            and (not english or _SECOND_PERSON.search(source) or next(iter(pairs)).lower() in source)):
        address.append(f"to {next(iter(pairs))}: \"{other}\"")
    terms = [f"{_flat(entry.get('source'))} = {_flat(entry.get('target'))}"
             for entry in [*(glossary.get("names") or []), *(glossary.get("terms") or [])]
             if isinstance(entry, dict) and _flat(entry.get("source")) and _flat(entry.get("source")).lower() in source
             and _flat(entry.get("target"))][:MAX_SETTLED]
    return address, terms


def line_state(line: Line, text: str | None = None, memory: dict | None = None, language: str = "Vietnamese") -> str:
    state = (f"SOURCE: {_flat(line.source)}\nTRANSLATION ({language}): {_flat(text or line.text)}\n"
             f"LINE BEFORE: {_flat(line.before) or '(none)'}\nLINE AFTER: {_flat(line.after) or '(none)'}")
    address, terms = settled(line, memory)
    if line.speaker:
        state += f"\nSPEAKER: {line.speaker}"
    if address:
        state += "\nADDRESS (how the speaker refers to themselves and to each listener): " + "; ".join(address)
    if terms:
        state += "\nGLOSSARY: " + "; ".join(terms)
    return state[:MAX_STATE_CHARS]


def questions(line: Line, memory: dict | None, language: str = "Vietnamese") -> dict:
    return in_language({**QUESTIONS, "settled": SETTLED} if any(settled(line, memory)) else QUESTIONS, language)


def evaluate_url(provider) -> str:
    return f"{str(provider.api_base).rstrip('/')}/evaluate"


def judge(provider, api_key: str, state: str, asked: dict = QUESTIONS) -> dict[str, float]:
    response = requests.post(
        evaluate_url(provider), headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={**({"model": decider.DECIDER_MODEL} if decider.available(provider, api_key) else {}),
              "state": state, "questions": asked},
        timeout=(TRANSLATION_CONNECT_TIMEOUT_SECONDS, TRANSLATION_READ_TIMEOUT_SECONDS), allow_redirects=False,
    )
    if not response.ok:
        raise RuntimeError(f"Judge HTTP {response.status_code}: {response.text[:200]}")
    answers = response.json().get("answers") or {}
    scores = {}
    for key in asked:
        answer = answers.get(key) if isinstance(answers, dict) else None
        if isinstance(answer, dict) and isinstance(answer.get("score"), (int, float)):
            scores[key] = float(answer["score"])
    return scores


def problems(scores: dict[str, float]) -> list[str]:
    return [PROBLEMS[key] for key, limit in LIMITS.items() if key in scores and scores[key] < limit]


def better(new: dict[str, float], old: dict[str, float]) -> bool:
    kept = all(new.get(key, 0.0) >= old.get(key, 0.0) - MAX_LOSS for key in ("natural", "faithful"))
    return kept and quality(new) >= quality(old) + MIN_GAIN


def quality(scores: dict[str, float]) -> float:
    asked = {**QUESTIONS, **({"settled": SETTLED} if "settled" in scores else {})}
    return sum(scores.get(key, 0.0) / (len(q["criteria"]) - 1) for key, q in asked.items()) / len(asked)


def rewrite(provider, model: str, api_key: str, lines: list[Line], memory: dict,
            language: str = "Vietnamese") -> tuple[dict[str, str], float | None]:
    items = [{"id": line.id, **({"speaker": line.speaker} if line.speaker else {}), "source": line.source, "translation": line.text, "before": line.before,
              "after": line.after, "problem": in_language("; ".join(problems(line.scores)), language),
              **({"max_chars": line.max_chars} if line.max_chars else {})} for line in lines]
    prompt = (in_language(REWRITE_PROMPT, language) + "\n\nCHAPTER MEMORY: "
              + json.dumps({k: memory.get(k) for k in ("glossary", "characters", "address") if memory.get(k)},
                           ensure_ascii=False, separators=(",", ":"))
              + "\n\nITEMS: " + json.dumps(items, ensure_ascii=False, separators=(",", ":")))
    result = request_vision_json(provider, model, api_key, prompt, [], max_tokens=1000000,
                                 stage="polish")
    wanted = {line.id for line in lines}
    out = {}
    for entry in result.data.get("rewrites") or []:
        if isinstance(entry, dict) and str(entry.get("id")) in wanted and isinstance(entry.get("text"), str):
            text = entry["text"].strip().replace("…", "...")
            if text:
                out[str(entry["id"])] = text
    return out, result.estimated_cost_usd
