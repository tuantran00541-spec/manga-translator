from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2
import requests

from app.ai_mode.vision_json import request_vision_json, stage_headers
from app.box_format import scaled_box
from app.ai_providers import AIProvider, auth_headers
from app.logging_config import logger
from app.parameters import TRANSLATION_CONNECT_TIMEOUT_SECONDS, TRANSLATION_READ_TIMEOUT_SECONDS
from app.security import validate_url
from app.render.font_guide import DEFAULT_LETTERING_FONT, font_for
from app.translation.context import CONTAINERS, TYPOGRAPHY_ROLES, ChapterMemory, system_prompt, text_system_prompt, setting_question, parse_setting
from app.translation.cost import language_name, response_cost_usd
from app.visual_qc.deepseek_region_client import _extract_output_text, _safe_error_detail
from app.visual_qc.gemini import _encode_for_gemini, _read_image
from app.visual_qc.gemini_interactions import (
    GEMINI_INTERACTIONS_URL,
    extract_output_text as gemini_output_text,
    safe_error_detail as gemini_safe_error,
)

_TRANSLATIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "translations": {"type": "array", "items": {
            "type": "object",
            "properties": {"id": {"type": "string"}, "source": {"type": "string"}, "translated_text": {"type": "string"},
                           "role": {"type": "string"}, "container": {"type": "string"},
                           "color": {"type": "string"}, "enlarge": {"type": "boolean"}, "review": {"type": "boolean"}},
            "required": ["id", "translated_text"],
        }},
        "speakers": {"type": "object"},
        "characters": {"type": "array", "items": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "note": {"type": "string"}},
            "required": ["name"],
        }},
        "unboxed": {"type": "array", "items": {"type": "object", "properties": {
            k: {"type": "number"} for k in ("x1", "y1", "x2", "y2")}}},
        "address": {"type": "array", "items": {
            "type": "object",
            "properties": {k: {"type": "string"} for k in ("from", "to", "self", "other")},
            "required": ["from", "to"],
        }},
    },
    "required": ["translations"],
}


@dataclass(frozen=True)
class VisionTranslationResult:
    translations: dict[str, str]
    model: str
    usage: dict
    estimated_cost_usd: float | None
    font_choices: dict[str, dict] = field(default_factory=dict)
    roles: dict[str, str] = field(default_factory=dict)
    review_ids: frozenset[str] = frozenset()
    missing_ids: frozenset[str] = frozenset()
    enlarge_ids: frozenset[str] = frozenset()
    colors: dict[str, str] = field(default_factory=dict)
    containers: dict[str, str] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)
    speakers: dict[str, str] = field(default_factory=dict)
    unboxed: tuple[tuple[int, int, int, int], ...] = ()


def _merge_translation_results(
    first: VisionTranslationResult, second: VisionTranslationResult,
) -> VisionTranslationResult:
    usage: dict = {}
    for part in (first.usage, second.usage):
        for key, value in (part or {}).items():
            if isinstance(value, (int, float)):
                usage[key] = usage.get(key, 0) + value
    cost = None
    if first.estimated_cost_usd is not None or second.estimated_cost_usd is not None:
        cost = (first.estimated_cost_usd or 0.0) + (second.estimated_cost_usd or 0.0)
    return VisionTranslationResult(
        translations={**first.translations, **second.translations},
        model=first.model,
        usage=usage,
        estimated_cost_usd=cost,
        font_choices={**first.font_choices, **second.font_choices},
        roles={**first.roles, **second.roles},
        review_ids=first.review_ids | second.review_ids,
        missing_ids=(first.missing_ids | second.missing_ids) - {
            k for k, v in {**first.translations, **second.translations}.items() if str(v or "").strip()
        },
        enlarge_ids=first.enlarge_ids | second.enlarge_ids,
        colors={**first.colors, **second.colors},
        containers={**first.containers, **second.containers},
        sources={**first.sources, **second.sources},
        speakers={**first.speakers, **second.speakers},
        unboxed=tuple(first.unboxed) + tuple(second.unboxed),
    )


PLACEHOLDER_TRANSLATIONS = {"n/a", "na", "n/a.", "none", "none.", "null", "undefined", "[n/a]", "(n/a)", "not applicable"}


def collapse_repeated_lines(text: str) -> str:
    if not isinstance(text, str):
        return text
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1 and len(set(lines)) == 1:
        return lines[0]
    return text


def parse_vision_translation(content: str, expected_ids: set[str], *, allow_missing: bool = False) -> dict[str, str]:
    source = str(content or "").strip()
    if source.startswith(chr(96) * 3):
        source = source.split("\n", 1)[-1].rsplit(chr(96) * 3, 1)[0].strip()
    try:
        data = json.loads(source)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Vision model returned invalid JSON") from exc
    entries = data.get("translations") if isinstance(data, dict) else None
    if isinstance(entries, dict):
        entries = [{"id": key, "translated_text": value} for key, value in entries.items()]
    if isinstance(entries, list):
        entries = [
            {**entry, "id": str(entry["id"])}
            if isinstance(entry, dict) and isinstance(entry.get("id"), int) and not isinstance(entry.get("id"), bool)
            else entry
            for entry in entries
        ]
    if not isinstance(entries, list):
        shape = sorted(data)[:8] if isinstance(data, dict) else type(data).__name__
        raise RuntimeError(f"Vision model must return a translations array (got {shape})")
    results: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            raise RuntimeError("Vision model returned an invalid object ID")
        obj_id = entry["id"]
        value = entry.get("translated_text")
        if allow_missing and (obj_id not in expected_ids or obj_id in results):
            continue
        if obj_id not in expected_ids or obj_id in results or not isinstance(value, str):
            raise RuntimeError("Vision model returned an unknown, repeated or malformed translation")
        if len(value) > 4000:
            raise RuntimeError("Vision model returned oversized text")
        cleaned = value.strip().replace("…", "...")
        results[obj_id] = "" if cleaned.lower() in PLACEHOLDER_TRANSLATIONS else cleaned
    if set(results) != expected_ids:
        if not allow_missing or not results:
            raise RuntimeError("Vision model omitted one or more text-object IDs")
        results.update({item_id: "" for item_id in expected_ids - set(results)})
    return results


def _salvage_translation_entries(choice: dict, expected_ids: set[str]) -> list[dict]:
    message = choice.get("message") if isinstance(choice, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        return []
    start = content.find('"translations"')
    open_bracket = content.find("[", start) if start >= 0 else -1
    if open_bracket < 0:
        return []
    salvaged: list[dict] = []
    seen_ids: set[str] = set()
    depth = 0
    in_string = False
    escaped = False
    element_start: int | None = None
    for pos in range(open_bracket + 1, len(content)):
        char = content[pos]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                element_start = pos
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and element_start is not None:
                try:
                    entry = json.loads(content[element_start:pos + 1])
                except ValueError:
                    entry = None
                if (isinstance(entry, dict) and isinstance(entry.get("id"), str)
                        and entry["id"] in expected_ids and isinstance(entry.get("translated_text"), str)
                        and entry["id"] not in seen_ids):
                    salvaged.append(entry)
                    seen_ids.add(entry["id"])
                element_start = None
        elif char == "]" and depth == 0:
            break
    return salvaged


def _salvage_translations(choice: dict, expected_ids: set[str]) -> dict[str, str]:
    return {
        entry["id"]: entry["translated_text"]
        for entry in _salvage_translation_entries(choice, expected_ids)
    }


def _parse_vision_payload(content: str, expected_ids: set[str]) -> tuple[dict[str, str], dict[str, dict], dict]:
    source = str(content or "").strip()
    if source.startswith(chr(96) * 3):
        source = source.split("\n", 1)[-1].rsplit(chr(96) * 3, 1)[0].strip()
    try:
        data = json.loads(source)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Vision model returned invalid JSON") from exc
    translations = parse_vision_translation(source, expected_ids, allow_missing=True)
    if isinstance(data, dict) and isinstance(data.get("translations"), dict):
        data["translations"] = [{"id": str(k), "translated_text": v} for k, v in data["translations"].items()]
    choices = {item_id: {"font_id": DEFAULT_LETTERING_FONT, "font_mode": "ai"} for item_id in translations}
    return translations, choices, data if isinstance(data, dict) else {}


def _unalias(result: VisionTranslationResult, data: dict, real: dict[str, str]) -> tuple[VisionTranslationResult, dict]:
    back = lambda key: real.get(str(key), str(key))  # noqa: E731
    data = dict(data)
    if isinstance(data.get("translations"), list):
        data["translations"] = [
            {**entry, "id": back(entry.get("id"))} if isinstance(entry, dict) else entry
            for entry in data["translations"]
        ]
    if isinstance(data.get("speakers"), dict):
        data["speakers"] = {back(k): v for k, v in data["speakers"].items()}
    return replace(
        result,
        translations={back(k): v for k, v in result.translations.items()},
        font_choices={back(k): v for k, v in (result.font_choices or {}).items()},
    ), data


MAX_UNBOXED = 8
_HEX_COLOR = re.compile(r"#[0-9a-fA-F]{6}")

_FOREIGN_SCRIPT = re.compile("[\u1100-\u11ff\u3040-\u30ff\u3130-\u318f\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff]")


_ENGLISH_WORDS = frozenset(
    "the and with you your of is are was were its it's this that for my we what i".split())
_WEAK_ENGLISH_WORDS = frozenset("to me in on".split())

_ENGLISH_CONTENT_WORDS = frozenset(
    "senior brother sister problem school teacher student please thanks thank sorry hello goodbye "
    "friend friends family father mother morning tonight today tomorrow yesterday people person "
    "woman women children world think thought want need help stop wait come look know good great "
    "little first never always really very much many thing things something nothing everything "
    "everyone somebody nobody where when why how which whose about after before because could "
    "should would there here then".split())

_CONFUSABLES = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "“": '"', "”": '"', "„": '"',
    " ": " ", "…": "...", "—": "-", "–": "-",
})
_ZERO_WIDTH = ("​", "‌", "‍", "﻿")


def _normalize_for_script_check(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    normalized = normalized.translate(_CONFUSABLES)
    for char in _ZERO_WIDTH:
        normalized = normalized.replace(char, "")
    return "".join(
        " " if char.isspace() else ("" if unicodedata.category(char) == "Cf" else char)
        for char in normalized)


_ASCII_WORD = re.compile(r"[A-Za-z]+")


def _ascii_words(text: str) -> list[str]:
    return _ASCII_WORD.findall(unicodedata.normalize("NFKC", text or ""))


_WORD = re.compile(r"[^\W\d_]+")


def _letter_words(text: str) -> list[str]:
    return _WORD.findall(text or "")


def _has_non_ascii_letter(text: str) -> bool:
    return any(char.isalpha() and ord(char) > 127 for char in (text or ""))


def _english_segment_spans(line: str) -> list[tuple[int, int]]:
    normalized = _normalize_for_script_check(line)
    norm_words = [(match.group(0), match.start(), match.end())
                  for match in _WORD.finditer(normalized)]
    orig_words = [(match.group(0), match.start(), match.end())
                  for match in _WORD.finditer(line)]
    if len(norm_words) != len(orig_words):
        return []
    spans: list[tuple[int, int]] = []
    run: list[int] = []

    def flush() -> None:
        nonlocal run
        if run and any(norm_words[i][0].lower() in _ENGLISH_CONTENT_WORDS for i in run):
            spans.append((orig_words[run[0]][1], orig_words[run[-1]][2]))
        run = []

    for index, (_word, _start, _end) in enumerate(orig_words):
        if norm_words[index][0].isascii():
            run.append(index)
        else:
            flush()
    flush()
    return spans


def _strip_english_segments(line: str) -> str:
    spans = _english_segment_spans(line)
    if not spans:
        return line
    out, pos = [], 0
    for start, end in spans:
        while end < len(line) and not line[end].isalnum() and not line[end].isspace():
            end += 1
        out.append(line[pos:start])
        pos = end
    out.append(line[pos:])
    return re.sub(r"\s+", " ", "".join(out)).strip()


def drawn_in_art(source: str) -> bool:
    return bool(_FOREIGN_SCRIPT.search(source or ""))


def _line_is_english(line: str) -> bool:
    line = _normalize_for_script_check(line)
    words = [word.lower() for word in _letter_words(line)]
    if len(words) < 2:
        return False
    if not _has_non_ascii_letter(line):
        if any(word in _ENGLISH_WORDS or word in _ENGLISH_CONTENT_WORDS for word in words):
            return True
        return sum(1 for word in words if word in _WEAK_ENGLISH_WORDS) >= 2
    return bool(_english_segment_spans(line))


def _untranslated_english(text: str) -> bool:
    lines = str(text or "").splitlines()
    if any(_line_is_english(line) for line in lines):
        return True
    if lines and not any(_has_non_ascii_letter(line) for line in lines):
        return _line_is_english(" ".join(lines))
    return False


_BARE_VIETNAMESE = frozenset(
    "khong mot nguoi cua dang duoc nhung chung toi nay that roi cung minh phai trong nhu thanh dieu ngay".split())


def _missing_diacritics(text: str) -> bool:
    for line in text.splitlines():
        line = _normalize_for_script_check(line)
        if _has_non_ascii_letter(line):
            continue
        words = [word.lower() for word in _ascii_words(line)]
        if sum(word in _BARE_VIETNAMESE for word in words) >= 2:
            return True
    return False


def _latin_noise(text: str) -> bool:
    return any(_line_is_latin_noise(line) for line in str(text or "").splitlines())


def _line_is_latin_noise(line: str) -> bool:
    words = _ascii_words(line)
    if any(len(word) >= 2 for word in words) or _has_non_ascii_letter(line):
        return False
    normalized = _normalize_for_script_check(line)
    return "/" in normalized or len(words) >= 2


def _bare_scream(text: str) -> bool:
    return any(_line_is_bare_scream(line) for line in str(text or "").splitlines())


def _strip_diacritics(text: str) -> str:
    return "".join(char for char in unicodedata.normalize("NFKD", text or "") if not unicodedata.combining(char))


def _line_is_bare_scream(line: str) -> bool:
    raw_line = line
    line = _strip_diacritics(line)
    words = _ascii_words(line)
    if len(words) != 1:
        return False
    word = words[0]
    if len(word) < 3 or len(set(word)) != 1:
        return False
    if _has_non_ascii_letter(line):
        return False
    if _has_non_ascii_letter(raw_line) and not word.isupper():
        return False
    rest = _normalize_for_script_check(line).replace(word, "", 1)
    return all(not char.isalnum() for char in rest)


_HANGUL = re.compile(r"[\uac00-\ud7a3\u1100-\u11ff\u3130-\u318f]")
_VOWELS = frozenset("aeiou")


def _hangul_misread(source: str, text: str) -> bool:
    if not _HANGUL.search(source or ""):
        return False
    return any(_line_is_hangul_misread(source, line) for line in str(text or "").splitlines())


def _line_is_hangul_misread(source: str, line: str) -> bool:
    if not _HANGUL.search(source or ""):
        return False
    if _has_non_ascii_letter(line):
        return False
    words = _ascii_words(line)
    return bool(words) and not any(
        char.lower() in _VOWELS for word in words for char in word)


def _filter_foreign_lines(value: str, source: str) -> str:
    lines = str(value or "").splitlines()
    if not any(_has_non_ascii_letter(line) for line in lines):
        if _untranslated_english(value):
            return ""
        kept = [line for line in lines
                if not (_missing_diacritics(line) or _latin_noise(line)
                        or _bare_scream(line) or _hangul_misread(source, line))]
        return "\n".join(kept)
    kept = []
    ascii_lines = [line for line in lines if not _has_non_ascii_letter(line)]
    ascii_group_english = (
        bool(ascii_lines)
        and not any(_untranslated_english(line) for line in ascii_lines)
        and _untranslated_english("\n".join(ascii_lines)))
    for line in lines:
        if _FOREIGN_SCRIPT.search(line):
            continue
        if not _has_non_ascii_letter(line):
            if (ascii_group_english or _untranslated_english(line) or _missing_diacritics(line)
                    or _latin_noise(line) or _bare_scream(line)
                    or _hangul_misread(source, line)):
                continue
            kept.append(line)
        else:
            if _bare_scream(line):
                continue
            stripped = _strip_english_segments(line)
            if stripped.strip():
                kept.append(stripped)
    return "\n".join(kept)


def _memory_name_is_english(entry: dict, key: str) -> bool:
    fields = ("name",) if key == "characters" else ("from", "to")
    return any(isinstance(entry, dict) and _untranslated_english(entry.get(field) or "")
               for field in fields)


def _drop_foreign_script(result: VisionTranslationResult, data: dict) -> tuple[VisionTranslationResult, dict]:
    sources = {str(entry.get("id")): str(entry.get("source") or "")
               for entry in data.get("translations") or []
               if isinstance(entry, dict)}
    foreign: set[str] = set()
    kept: dict[str, str] = {}
    for key, value in result.translations.items():
        filtered = _filter_foreign_lines(value, sources.get(key, ""))
        if filtered == "":
            if value.strip():
                foreign.add(key)
        elif filtered != value:
            kept[key] = filtered
    data = dict(data)
    if isinstance(data.get("translations"), list):
        updated = []
        for entry in data["translations"]:
            if not isinstance(entry, dict):
                updated.append(entry)
                continue
            entry_id = str(entry.get("id"))
            if entry_id in foreign:
                continue
            if entry_id in kept:
                entry = {**entry, "translated_text": kept[entry_id]}
            updated.append(entry)
        data["translations"] = updated
    for key in ("characters", "address"):
        if isinstance(data.get(key), list):
            data[key] = [entry for entry in data[key]
                         if not _FOREIGN_SCRIPT.search(json.dumps(entry, ensure_ascii=False))
                         and not _memory_name_is_english(entry, key)]
    if foreign or kept:
        result = replace(result, translations={key: ("" if key in foreign else kept.get(key, value))
                                               for key, value in result.translations.items()})
    return result, data


MARK_COLOR = (0, 0, 230)
ELSEWHERE_COLOR = (128, 128, 128)
ELSEWHERE_LABEL = "X"


def mark_objects(image, objects: list[dict], elsewhere: list | tuple = ()):
    marked = image.copy()
    width = marked.shape[1]
    thickness = max(2, width // 400)
    scale = max(0.6, width / 1000)
    boxes = [(obj["bbox_xyxy"], str(obj["id"]), MARK_COLOR) for obj in objects]
    boxes += [(box, ELSEWHERE_LABEL, ELSEWHERE_COLOR) for box in elsewhere]
    originals = [tuple(int(v) for v in box) for box, _, _ in boxes]
    for box, label, color in boxes:
        pad = 2 * thickness
        x1, y1, x2, y2 = (int(v) for v in box)
        x1, y1, x2, y2 = x1 - pad, y1 - pad, x2 + pad, y2 + pad
        cv2.rectangle(marked, (x1, y1), (x2, y2), color, thickness)
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        lw, lh = tw + 4, th + base
        lx = min(max(x1, 0), max(0, width - lw))
        cy = (y1 + y2) // 2 - lh // 2
        candidates = [(lx, y1 - lh, lx + lw, y1), (lx, y2, lx + lw, y2 + lh),
                      (x2, cy, x2 + lw, cy + lh), (x1 - lw, cy, x1, cy + lh)]

        def _intrusion(rect) -> int:
            rx1, ry1, rx2, ry2 = rect
            if rx1 < 0 or rx2 > width or ry1 < 0 or ry2 > marked.shape[0]:
                return 1 << 30
            return sum(max(0, min(rx2, ox2) - max(rx1, ox1)) * max(0, min(ry2, oy2) - max(ry1, oy1))
                       for ox1, oy1, ox2, oy2 in originals)

        lx, top, _, bottom = min(candidates, key=_intrusion)
        cv2.rectangle(marked, (lx, top), (lx + lw, bottom), color, -1)
        cv2.putText(marked, label, (lx + 2, bottom - base), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness)
    return marked


def retranslate_text(
    provider: AIProvider, model: str, api_key: str, pairs: list[tuple[str, str]],
    *, source_lang: str, target_lang: str, memory_snapshot: dict | None = None,
    settled: dict | None = None, setting: str | None = None, max_tokens: int = 65536,
) -> tuple[dict[str, str], float | None, dict]:
    pairs = [(str(i), str(s).strip()) for i, s in pairs if str(s).strip()]
    if not pairs:
        return {}, None, {}
    source_name = (
        "the original language"
        if str(source_lang or "").lower() in {"", "auto"}
        else language_name(source_lang)
    )
    prompt = (
        text_system_prompt(language_name(target_lang), target_lang, settled, setting=setting)
        + "\n\n"
        + (f"CHAPTER MEMORY (context from earlier slices, not to be copied into the answer): "
           f"{json.dumps(memory_snapshot, ensure_ascii=False, separators=(',', ':'))}\n\n"
           if memory_snapshot else "")
        + f"Translate these text objects from {source_name} to {language_name(target_lang)}.\n"
        + json.dumps({"objects": [{"id": i, "source": s} for i, s in pairs]},
                     ensure_ascii=False, separators=(",", ":"))
    )
    result = request_vision_json(provider, model, api_key, prompt, [], max_tokens=max_tokens, stage="retranslate")
    expected = {i for i, _ in pairs}
    out = {}
    translations = result.data.get("translations") or []
    if isinstance(translations, dict):
        translations = [{"id": key, "translated_text": value} for key, value in translations.items()]
    for entry in translations:
        if not isinstance(entry, dict):
            continue
        key, value = str(entry.get("id") or ""), entry.get("translated_text")
        if key in expected and isinstance(value, str) and value.strip():
            out[key] = value
    if pairs and not out:
        logger.warning("Re-translate returned no translations for {} pairs; keeping the vision wording", len(pairs))
    return out, result.estimated_cost_usd, result.usage


class VisionPageTranslator:
    def __init__(self, provider: AIProvider, model: str, *, translate_provider: AIProvider | None = None,
                 translate_model: str | None = None, translate_api_key: str | None = None):
        if not provider.supports_visual_qc:
            raise ValueError(f"{provider.label} cannot receive image inputs")
        self.provider, self.model = provider, model
        self.translate_provider, self.translate_model, self.translate_api_key = (
            translate_provider, translate_model, translate_api_key,
        )

    def detect_setting(self, original_path: Path, api_key: str) -> str | None:
        image = _read_image(Path(original_path))
        setting = None
        for budget in (4096, 8192):
            try:
                result = request_vision_json(self.provider, self.model, api_key, setting_question(),
                                             [("page", image)], max_tokens=budget, reasoning_effort="low",
                                             stage="setting")
            except RuntimeError:
                continue
            setting = parse_setting((result.data or {}).get("setting"))
            if setting:
                break
        logger.info("Chapter setting detected: {}", setting or "unknown (general register)")
        return setting

    def translate_page(
        self, original_path: Path, cleaned_path: Path, items: list[dict],
        *, api_key: str, source_lang: str, target_lang: str,
        memory: ChapterMemory | None = None, slice_number: int | None = None, slice_total: int | None = None,
        elsewhere: list | tuple = (), setting: str | None = None, _max_tokens: int = 1000000,
    ) -> VisionTranslationResult:
        if not api_key.strip():
            raise ValueError(f"{self.provider.label} API key is not configured")
        if not items:
            return VisionTranslationResult({}, self.model, {}, 0.0 if self.provider.tracks_cost else None)
        try:
            return self._translate_batch(
                original_path, cleaned_path, items, api_key=api_key, source_lang=source_lang,
                target_lang=target_lang, memory=memory, slice_number=slice_number,
                slice_total=slice_total, elsewhere=elsewhere, setting=setting, max_tokens=_max_tokens,
            )
        except RuntimeError as exc:
            if "finish_reason=length" not in str(exc):
                raise
            if len(items) >= 2:
                mid = len(items) // 2
                first = self.translate_page(
                    original_path, cleaned_path, items[:mid], api_key=api_key, source_lang=source_lang,
                    target_lang=target_lang, memory=memory, slice_number=slice_number,
                    slice_total=slice_total, elsewhere=elsewhere, setting=setting, _max_tokens=_max_tokens,
                )
                second = self.translate_page(
                    original_path, cleaned_path, items[mid:], api_key=api_key, source_lang=source_lang,
                    target_lang=target_lang, memory=memory, slice_number=slice_number,
                    slice_total=slice_total, elsewhere=elsewhere, setting=setting, _max_tokens=_max_tokens,
                )
                return _merge_translation_results(first, second)
            if _max_tokens < 16384:
                return self.translate_page(
                    original_path, cleaned_path, items, api_key=api_key, source_lang=source_lang,
                    target_lang=target_lang, memory=memory, slice_number=slice_number,
                    slice_total=slice_total, elsewhere=elsewhere, setting=setting,
                    _max_tokens=min(16384, _max_tokens * 2),
                )
            raise

    def _translate_batch(
        self, original_path: Path, cleaned_path: Path, items: list[dict],
        *, api_key: str, source_lang: str, target_lang: str,
        memory: ChapterMemory | None = None, slice_number: int | None = None, slice_total: int | None = None,
        elsewhere: list | tuple = (), setting: str | None = None, max_tokens: int = 1000000,
    ) -> VisionTranslationResult:
        original, cleaned = _read_image(original_path), _read_image(cleaned_path)
        if original.shape[:2] != cleaned.shape[:2]:
            raise ValueError("Original and cleaned slices have different dimensions")
        h, w = original.shape[:2]
        real = {str(n): str(item["id"]) for n, item in enumerate(items, start=1)}
        objects = [{"id": alias, "source_text": item["text"], "bbox_xyxy": item["region"],
                    **({"max_chars": item["max_chars"]} if item.get("max_chars") else {})}
                   for alias, item in zip(real, items)]
        source_name = (
            "the original language shown in the image"
            if str(source_lang or "").lower() in {"", "auto"}
            else language_name(source_lang)
        )
        system = system_prompt(language_name(target_lang), target_lang, memory.settled() if memory is not None else None,
                               setting=setting)
        where = f"SLICE {slice_number} of {slice_total}. " if slice_number and slice_total else ""
        prompt = (
            (f"CHAPTER MEMORY (context from earlier slices, not to be copied into the answer): "
             f"{json.dumps(memory.snapshot(settled=False), ensure_ascii=False, separators=(',', ':'))}\n\n"
             if memory is not None else "")
            + f"{where}Translate these text objects from {source_name}.\n"
            + json.dumps({"image_width": w, "image_height": h, "objects": objects,
                          **({"lettered_elsewhere": [list(map(int, box)) for box in elsewhere]} if elsewhere else {})},
                         ensure_ascii=False, separators=(",", ":"))
            + ("\nlettered_elsewhere boxes (grey X in the image) hold text the neighbouring slice translates: "
               "never translate it, and never fold its words or meaning into any object's translation."
               if elsewhere else "")
            + '\n\nTranslate every object above and answer with one JSON object that starts with {"translations":[. '
            + "Write translated_text in sentence case, even when the source is in capitals."
        )
        marked_b64 = _encode_for_gemini(mark_objects(original, objects, elsewhere))
        ids = set(real)
        if self.provider.protocol == "gemini":
            result, data = self._gemini(system, prompt, marked_b64, api_key=api_key, ids=ids, max_tokens=max_tokens)
        else:
            result, data = self._openai(system, prompt, marked_b64, api_key=api_key, ids=ids, max_tokens=max_tokens)
        result, data = _unalias(result, data, real)
        ids = set(real.values())
        read_sources = {
            str(entry.get("id")): " ".join(str(entry.get("source") or "").split())[:1000]
            for entry in data.get("translations") or []
            if isinstance(entry, dict) and str(entry.get("source") or "").strip()
        }
        if self.translate_provider is not None and self.translate_model:
            pairs = [
                (str(item["id"]), (read_sources.get(str(item["id"]))
                                   or str(item.get("text") or "").strip()))
                for item in items
            ]
            retold, retold_cost, retold_usage = {}, None, {}
            try:
                retold, retold_cost, retold_usage = retranslate_text(
                    self.translate_provider, self.translate_model, self.translate_api_key or "",
                    pairs, source_lang=source_lang, target_lang=target_lang,
                    memory_snapshot=memory.snapshot(settled=False) if memory is not None else None,
                    settled=memory.settled() if memory is not None else None,
                    setting=setting,
                )
            except (RuntimeError, ValueError, OSError) as exc:
                logger.warning("Re-translate failed for slice {}: {}; keeping the vision wording",
                               slice_number, str(exc)[:150])
            usage = dict(result.usage or {})
            for key, value in (retold_usage or {}).items():
                if isinstance(value, (int, float)):
                    usage[key] = usage.get(key, 0) + value
            cost = result.estimated_cost_usd
            if cost is not None or retold_cost is not None:
                cost = (cost or 0.0) + (retold_cost or 0.0)
            result = replace(result, usage=usage, estimated_cost_usd=cost)
            if retold:
                result = replace(result, translations={**result.translations,
                                                       **{k: v for k, v in retold.items() if k in result.translations}})
                for entry in data.get("translations") or []:
                    if isinstance(entry, dict) and str(entry.get("id")) in retold:
                        entry["translated_text"] = retold[str(entry["id"])]
        if str(target_lang or "").lower() in {"vi", "vie", "vietnamese"}:
            for entry in data.get("translations") or []:
                if isinstance(entry, dict) and not str(entry.get("source") or "").strip():
                    ocr = next((str(item.get("text") or "")
                                for item in items if str(item.get("id")) == str(entry.get("id"))), "")
                    if ocr.strip():
                        entry["source"] = ocr
            result, data = _drop_foreign_script(result, data)
        roles, containers, review, answered, enlarge, colors, sources = {}, {}, set(), set(), set(), {}, {}
        for entry in data.get("translations") or []:
            if not isinstance(entry, dict) or str(entry.get("id")) not in ids:
                continue
            answered.add(str(entry["id"]))
            role = str(entry.get("role") or "").strip().lower()
            if role in TYPOGRAPHY_ROLES:
                roles[str(entry["id"])] = role
            container = str(entry.get("container") or "").strip().lower()
            if container in CONTAINERS:
                containers[str(entry["id"])] = container
            if entry.get("review") is True:
                review.add(str(entry["id"]))
            if entry.get("enlarge") is True:
                enlarge.add(str(entry["id"]))
            if str(entry.get("id")) in read_sources:
                sources[str(entry["id"])] = read_sources[str(entry["id"])]
            elif str(entry.get("source") or "").strip():
                sources[str(entry["id"])] = " ".join(str(entry["source"]).split())[:1000]
            if isinstance(entry.get("color"), str) and _HEX_COLOR.fullmatch(entry["color"].strip()):
                colors[str(entry["id"])] = entry["color"].strip().lower()
        fonts = {key: {"font_id": font_for(roles.get(key), containers.get(key), target_lang), "font_mode": "ai"}
                 for key in (result.font_choices or {})}
        if memory is not None:
            memory.update(slice_number or 0, data,
                          {key: collapse_repeated_lines(value) for key, value in result.translations.items()},
                          [str(item["id"]) for item in items])
        speakers = {str(k): " ".join(str(v).split())[:80] for k, v in (data.get("speakers") or {}).items()
                    if str(k) in ids and isinstance(v, str) and v.strip()} if isinstance(data.get("speakers"), dict) else {}
        unboxed = tuple(
            (int(box[0]), int(box[1]), int(box[2]) + 1, int(box[3]) + 1)
            for raw in (data.get("unboxed") or [])[:MAX_UNBOXED] if (box := scaled_box(raw, w, h)) is not None
        )
        return replace(
            result, font_choices=fonts, roles=roles, unboxed=unboxed, review_ids=frozenset(review), missing_ids=frozenset(ids - answered),
            enlarge_ids=frozenset(enlarge), colors=colors, containers=containers, sources=sources, speakers=speakers,
        )

    def _openai(self, system, prompt, marked, *, api_key, ids, max_tokens):
        url = str(self.provider.chat_url or "")
        if not url:
            raise ValueError("Provider does not have chat completions")
        if not self.provider.builtin:
            validate_url(url)
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": [
                {"type": "text", "text": "RAW SLICE WITH TEXT BOXES"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{marked}"}},
                {"type": "text", "text": prompt},
            ]}],
            "response_format": {"type": "json_object"},
            "max_tokens": max_tokens,
            "stream": False,
        }
        payload.update(self.provider.chat_completion_extras())
        try:
            response = requests.post(
                url,
                headers={**auth_headers(self.provider, api_key), "Content-Type": "application/json",
                         **stage_headers(self.provider, "translate")},
                json=payload,
                timeout=(TRANSLATION_CONNECT_TIMEOUT_SECONDS, TRANSLATION_READ_TIMEOUT_SECONDS),
                allow_redirects=False,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"{self.provider.label} vision request failed") from exc
        if 300 <= response.status_code < 400:
            raise RuntimeError(f"{self.provider.label} redirected the image request")
        if not response.ok:
            raise RuntimeError(
                f"{self.provider.label} HTTP {response.status_code}: "
                f"{_safe_error_detail(response, api_key.strip())}"
            )
        body = None
        try:
            body = response.json()
            answer = _extract_output_text(body)
        except (TypeError, KeyError, ValueError) as exc:
            choice = (body.get("choices") or [{}])[0] if isinstance(body, dict) else {}
            reason = choice.get("finish_reason") if isinstance(choice, dict) else None
            salvaged = _salvage_translation_entries(choice, ids) if reason == "length" else []
            if salvaged:
                translations, font_choices, data = _parse_vision_payload(
                    json.dumps({"translations": salvaged}, ensure_ascii=False), ids)
                usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
                cost = response_cost_usd(usage, self.provider.id, self.model, getattr(response, "headers", None))
                return VisionTranslationResult(translations, str(body.get("model") or self.model), usage, cost, font_choices), data
            raise RuntimeError(f"{self.provider.label} returned no translation text ({exc}; finish_reason={reason})") from exc
        translations, font_choices, data = _parse_vision_payload(answer, ids)
        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        cost = response_cost_usd(usage, self.provider.id, self.model, getattr(response, "headers", None))
        return VisionTranslationResult(translations, str(body.get("model") or self.model), usage, cost, font_choices), data

    def _gemini(self, system, prompt, marked, *, api_key, ids, max_tokens):
        payload = {
            "model": self.model, "store": False,
            "input": [
                {"type": "text", "text": system},
                {"type": "text", "text": "RAW SLICE WITH TEXT BOXES"},
                {"type": "image", "data": marked, "mime_type": "image/jpeg"},
                {"type": "text", "text": prompt},
            ],
            "response_format": {"type": "text", "mime_type": "application/json", "schema": _TRANSLATIONS_SCHEMA},
            "generation_config": {"thinking_level": "low", "max_output_tokens": max_tokens},
        }
        try:
            response = requests.post(
                GEMINI_INTERACTIONS_URL,
                headers={"Content-Type": "application/json", "x-goog-api-key": api_key.strip(), "Api-Revision": "2026-05-20"},
                json=payload,
                timeout=(TRANSLATION_CONNECT_TIMEOUT_SECONDS, TRANSLATION_READ_TIMEOUT_SECONDS),
                allow_redirects=False,
            )
        except requests.RequestException as exc:
            raise RuntimeError("Gemini vision translation failed") from exc
        if not response.ok:
            raise RuntimeError(f"Gemini HTTP {response.status_code}: {gemini_safe_error(response)}")
        try:
            body = response.json()
            answer = gemini_output_text(body)
        except (TypeError, KeyError, ValueError) as exc:
            raise RuntimeError("Gemini returned no translation text") from exc
        translations, font_choices, data = _parse_vision_payload(answer, ids)
        return VisionTranslationResult(
            translations, self.model,
            body.get("usage", {}) if isinstance(body.get("usage"), dict) else {}, None,
            font_choices,
        ), data
