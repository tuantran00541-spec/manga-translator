"""A text-only decision model grades each translated line; lines it doubts go back to the translator."""
from __future__ import annotations

import requests

from app.parameters import TRANSLATION_CONNECT_TIMEOUT_SECONDS, TRANSLATION_READ_TIMEOUT_SECONDS

FLAG = 0.7  # probability above which a line is sent back
SHORT_WORDS = 4  # titles and captions this short read as broken language to the judge too easily
FLAG_SHORT_LANGUAGE = 0.9  # so their language flag needs this much
QUESTION_TYPE = "boolean"  # minirouter's name for TypeSafe's yes/no ("noul") question
QUESTIONS = {
    "wrong_meaning": "The translation changes what the source says, adds something the source does not say, "
                     "or leaves part of the source out.",
    "bad_language": "The translation has a misspelled or broken Vietnamese word, an English word that should have "
                    "been translated, or clumsy word-for-word Vietnamese. Names of people, places, works and credits "
                    "kept in the source spelling are correct, and so are short title-style phrases.",
    "wrong_name": "The translation spells a name or term differently from the source or from the GLOSSARY.",
    "wrong_address": "The translation uses a pronoun or form of address that contradicts the GLOSSARY forms of "
                     "address for the people talking.",
    "duplicate": "The translation repeats words that belong to the previous or next line instead of its own source.",
}
NOTES = {
    "wrong_meaning": "the meaning differs from the source or part of it is missing",
    "bad_language": "misspelled, untranslated or word-for-word Vietnamese",
    "wrong_name": "a name or term does not match the glossary",
    "wrong_address": "the form of address contradicts the glossary",
    "duplicate": "it repeats the previous or next line",
}


def evaluate_url(chat_url: str) -> str:
    return chat_url.rsplit("/chat/completions", 1)[0] + "/evaluate"


def line_state(line: dict, previous: str, following: str, glossary: dict) -> str:
    """The text the judge reads for one line."""
    names = ", ".join(f'{item["source"]} = {item["target"]}' for item in glossary.get("names") or [])
    terms = ", ".join(f'{item["source"]} = {item["target"]}' for item in glossary.get("terms") or [])
    address = "; ".join(f'{item["from"]} to {item["to"]}: self "{item.get("self", "")}", other "{item.get("other", "")}"'
                        for item in glossary.get("address") or [])
    return "\n".join(part for part in (
        f"SOURCE: {line['source']}",
        f"TRANSLATION (Vietnamese): {line['translation']}",
        f"PREVIOUS LINE: {previous}" if previous else "",
        f"NEXT LINE: {following}" if following else "",
        f"GLOSSARY names: {names}" if names else "",
        f"GLOSSARY terms: {terms}" if terms else "",
        f"GLOSSARY forms of address: {address}" if address else "",
    ) if part)


def _probability(body, name: str) -> float | None:
    """The probability answered for ``name``, wherever the response nests it."""
    if isinstance(body, dict):
        if name in body:
            value = body[name]
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
            if isinstance(value, dict):
                for key in ("noul", "boolean", "probability", "value", "answer"):
                    if isinstance(value.get(key), (int, float)) and not isinstance(value.get(key), bool):
                        return float(value[key])
                odds = value.get("probabilities")
                if isinstance(odds, dict) and isinstance(odds.get("true"), (int, float)):
                    return float(odds["true"])
        for value in body.values():
            if (found := _probability(value, name)) is not None:
                return found
    elif isinstance(body, list):
        for value in body:
            if (found := _probability(value, name)) is not None:
                return found
    return None


def judge_line(url: str, api_key: str, state: str) -> tuple[dict[str, float], dict]:
    """Probabilities per question and the raw answer."""
    response = requests.post(
        url, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"state": state, "questions": {name: {"type": QUESTION_TYPE, "instructions": text}
                                            for name, text in QUESTIONS.items()}},
        timeout=(TRANSLATION_CONNECT_TIMEOUT_SECONDS, TRANSLATION_READ_TIMEOUT_SECONDS), allow_redirects=False,
    )
    if not response.ok:
        raise RuntimeError(f"Judge HTTP {response.status_code}: {response.text[:200]}")
    body = response.json()
    scores = {name: found for name in QUESTIONS if (found := _probability(body, name)) is not None}
    if not scores:
        raise RuntimeError(f"Judge answer has no probabilities: {str(body)[:200]}")
    return scores, body


def flagged(scores: dict[str, float], source: str = "") -> str:
    """Reviewer note for the translator, or "" when the line passes."""
    short = len(str(source or "").split()) <= SHORT_WORDS
    return "; ".join(NOTES[name] for name, value in scores.items()
                     if value >= (FLAG_SHORT_LANGUAGE if short and name == "bad_language" else FLAG))
