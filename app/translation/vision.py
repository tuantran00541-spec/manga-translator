from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2
import requests

from app.ai_mode.vision_json import request_vision_json, stage_headers
from app.box_format import scaled_box
from app.ai_providers import AIProvider, auth_headers
from app.parameters import TRANSLATION_CONNECT_TIMEOUT_SECONDS, TRANSLATION_READ_TIMEOUT_SECONDS
from app.security import validate_url
from app.render.font_guide import DEFAULT_LETTERING_FONT, font_for
from app.translation.context import CONTAINERS, TYPOGRAPHY_ROLES, ChapterMemory, system_prompt, text_system_prompt
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
                           "role": {"type": "string"}, "container": {"type": "string"}},
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
    # Objects the model did not answer for at all (as opposed to answering "").
    missing_ids: frozenset[str] = frozenset()
    enlarge_ids: frozenset[str] = frozenset()
    colors: dict[str, str] = field(default_factory=dict)
    containers: dict[str, str] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)  # the source text the model read in each box
    speakers: dict[str, str] = field(default_factory=dict)  # who says each box's line, or "narration"
    unboxed: tuple[tuple[int, int, int, int], ...] = ()  # pixel boxes of lines the model saw outside every box


def _merge_translation_results(
    first: VisionTranslationResult, second: VisionTranslationResult,
) -> VisionTranslationResult:
    # Halves of one batch: every per-object map unions, usage and cost add up.
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
        missing_ids=first.missing_ids | second.missing_ids,
        enlarge_ids=first.enlarge_ids | second.enlarge_ids,
        colors={**first.colors, **second.colors},
        containers={**first.containers, **second.containers},
        sources={**first.sources, **second.sources},
        speakers={**first.speakers, **second.speakers},
        unboxed=tuple(first.unboxed) + tuple(second.unboxed),
    )


PLACEHOLDER_TRANSLATIONS = {"n/a", "na", "n/a.", "none", "none.", "null", "undefined", "[n/a]", "(n/a)", "not applicable"}


def collapse_repeated_lines(text: str) -> str:
    # One box can hold two copies of the same source text; the model then
    # translates each copy and joins them, so the same line letters twice.
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
        # Not every lettering font has the ellipsis glyph.
        cleaned = value.strip().replace("…", "...")
        # A placeholder is not lettering; blank lets the preserve-original path keep the source art.
        results[obj_id] = "" if cleaned.lower() in PLACEHOLDER_TRANSLATIONS else cleaned
    if set(results) != expected_ids:
        if not allow_missing or not results:
            raise RuntimeError("Vision model omitted one or more text-object IDs")
        results.update({item_id: "" for item_id in expected_ids - set(results)})
    return results


def _salvage_translations(choice: dict, expected_ids: set[str]) -> dict[str, str]:
    """Complete entries from a reply cut off mid-list; the cut tail entry never parses, so it drops."""
    message = choice.get("message") if isinstance(choice, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        return {}
    start = content.find('"translations"')
    open_bracket = content.find("[", start) if start >= 0 else -1
    if open_bracket < 0:
        return {}
    salvaged: dict[str, str] = {}
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
                        and entry["id"] not in salvaged):
                    salvaged[entry["id"]] = entry["translated_text"]
                element_start = None
        elif char == "]" and depth == 0:
            break
    return salvaged


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
    # The font follows each text's role and container once those are read; until then the base font.
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


MAX_UNBOXED = 8  # lines outside every box one slice may report
_HEX_COLOR = re.compile(r"#[0-9a-fA-F]{6}")

# Chinese, Japanese and Korean letters never belong in a Vietnamese translation.
_FOREIGN_SCRIPT = re.compile("[\u1100-\u11ff\u3040-\u30ff\u3130-\u318f\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff]")


_ENGLISH_WORDS = frozenset(
    "the and with you your to of is are was were its it's this that for in on my me we what i".split())


def drawn_in_art(source: str) -> bool:
    """True for lettering in Chinese, Japanese or Korean script, which in a translated chapter the artist drew."""
    return bool(_FOREIGN_SCRIPT.search(source or ""))


def _untranslated_english(text: str) -> bool:
    """True for a line left in English: plain ASCII with at least two words, one of them an English function word."""
    words = [word.strip(".,!?;:\"'()-—…").lower() for word in text.split()]
    words = [word for word in words if word]
    return text.isascii() and len(words) >= 2 and any(word in _ENGLISH_WORDS for word in words)


# Common Vietnamese words written without their diacritics.
_BARE_VIETNAMESE = frozenset(
    "khong mot nguoi cua dang duoc nhung chung toi nay that roi cung minh phai trong nhu thanh dieu ngay".split())


def _missing_diacritics(text: str) -> bool:
    """True when any line is Vietnamese written in plain ASCII, which readers cannot accept."""
    for line in text.splitlines():
        words = [word.strip(".,!?;:\"'()-—…").lower() for word in line.split()]
        if line.isascii() and sum(word in _BARE_VIETNAMESE for word in words) >= 2:
            return True
    return False


def _drop_foreign_script(result: VisionTranslationResult, data: dict) -> tuple[VisionTranslationResult, dict]:
    """Treat CJK, untranslated English or diacritic-less lines as unanswered so they are retried, and keep them out of memory."""
    foreign = {key for key, value in result.translations.items()
               if _FOREIGN_SCRIPT.search(value) or _untranslated_english(value) or _missing_diacritics(value)}
    data = dict(data)
    if isinstance(data.get("translations"), list):
        data["translations"] = [entry for entry in data["translations"]
                                if not (isinstance(entry, dict) and str(entry.get("id")) in foreign)]
    for key in ("characters", "address"):
        if isinstance(data.get(key), list):
            data[key] = [entry for entry in data[key]
                         if not _FOREIGN_SCRIPT.search(json.dumps(entry, ensure_ascii=False))]
    if foreign:
        result = replace(result, translations={key: "" if key in foreign else value
                                               for key, value in result.translations.items()})
    return result, data


MARK_COLOR = (0, 0, 230)  # BGR red
ELSEWHERE_COLOR = (128, 128, 128)  # text lettered by the neighbouring slice
ELSEWHERE_LABEL = "X"


def mark_objects(image, objects: list[dict], elsewhere: list | tuple = ()):
    """A copy of the slice with each object's box outlined and labelled with its id; grey X boxes are lettered elsewhere."""
    marked = image.copy()
    width = marked.shape[1]
    thickness = max(2, width // 400)
    scale = max(0.6, width / 1000)
    boxes = [(obj["bbox_xyxy"], str(obj["id"]), MARK_COLOR) for obj in objects]
    boxes += [(box, ELSEWHERE_LABEL, ELSEWHERE_COLOR) for box in elsewhere]
    originals = [tuple(int(v) for v in box) for box, _, _ in boxes]
    for box, label, color in boxes:
        # Outlined just outside the box, so the line never covers the letters the model reads.
        pad = 2 * thickness
        x1, y1, x2, y2 = (int(v) for v in box)
        x1, y1, x2, y2 = x1 - pad, y1 - pad, x2 + pad, y2 + pad
        cv2.rectangle(marked, (x1, y1), (x2, y2), color, thickness)
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        lw, lh = tw + 4, th + base
        lx = min(max(x1, 0), max(0, width - lw))
        # The label sits clear of the box it names, above it when there is room
        # and below it at the top edge, so it never covers text the model reads.
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
    settled: dict | None = None, max_tokens: int = 65536,
) -> tuple[dict[str, str], float | None, dict]:
    """Re-translate (id, source-text) pairs with a text-only model.

    The vision model reads the image and judges role/container (which picks the
    font); this re-translates the source lines it read, so the wording is the
    text model's. Returns ({id: translated_text}, cost, usage) for the pairs the
    model answered, so the second billed pass joins the vision read's budget.
    """
    pairs = [(str(i), str(s).strip()) for i, s in pairs if str(s).strip()]
    if not pairs:
        return {}, None, {}
    source_name = (
        "the original language"
        if str(source_lang or "").lower() in {"", "auto"}
        else language_name(source_lang)
    )
    # The same conventions the vision call translates under; without them the text
    # model writes literal, unidiomatic Vietnamese (a bare "translate this" prompt).
    prompt = (
        text_system_prompt(language_name(target_lang), target_lang, settled)
        + "\n\n"
        + (f"CHAPTER MEMORY (context from earlier slices, not to be copied into the answer): "
           f"{json.dumps(memory_snapshot, ensure_ascii=False, separators=(',', ':'))}\n\n"
           if memory_snapshot else "")
        + f"Translate these text objects from {source_name} to {language_name(target_lang)}.\n"
        + json.dumps({"objects": [{"id": i, "source": s} for i, s in pairs]},
                     ensure_ascii=False, separators=(",", ":"))
    )
    # Its own checkpoint: sharing "translate" would send this text-only second
    # pass to the vision model again instead of the retranslate model.
    result = request_vision_json(provider, model, api_key, prompt, [], max_tokens=max_tokens, stage="retranslate")
    expected = {i for i, _ in pairs}
    out = {}
    for entry in (result.data.get("translations") or []):
        if not isinstance(entry, dict):
            continue
        key, value = str(entry.get("id") or ""), entry.get("translated_text")
        if key in expected and isinstance(value, str) and value.strip():
            out[key] = value
    return out, result.estimated_cost_usd, result.usage


class VisionPageTranslator:
    def __init__(self, provider: AIProvider, model: str, *, translate_provider: AIProvider | None = None,
                 translate_model: str | None = None, translate_api_key: str | None = None):
        if not provider.supports_visual_qc:
            raise ValueError(f"{provider.label} cannot receive image inputs")
        self.provider, self.model = provider, model
        # A second model re-translates the source lines the vision model read, so the
        # wording is the text model's while role/container (which picks the font) stay the vision model's.
        self.translate_provider, self.translate_model, self.translate_api_key = (
            translate_provider, translate_model, translate_api_key,
        )

    def translate_page(
        self, original_path: Path, cleaned_path: Path, items: list[dict],
        *, api_key: str, source_lang: str, target_lang: str,
        memory: ChapterMemory | None = None, slice_number: int | None = None, slice_total: int | None = None,
        elsewhere: list | tuple = (), _max_tokens: int = 1000000,
    ) -> VisionTranslationResult:
        if not api_key.strip():
            raise ValueError(f"{self.provider.label} API key is not configured")
        if not items:
            return VisionTranslationResult({}, self.model, {}, 0.0 if self.provider.tracks_cost else None)
        try:
            return self._translate_batch(
                original_path, cleaned_path, items, api_key=api_key, source_lang=source_lang,
                target_lang=target_lang, memory=memory, slice_number=slice_number,
                slice_total=slice_total, elsewhere=elsewhere, max_tokens=_max_tokens,
            )
        except RuntimeError as exc:
            if "finish_reason=length" not in str(exc):
                raise
            if len(items) >= 2:
                # A reply cut off mid-list loses the whole batch when nothing salvages;
                # halves carry half the output each, so the cut moves past the content.
                mid = len(items) // 2
                first = self.translate_page(
                    original_path, cleaned_path, items[:mid], api_key=api_key, source_lang=source_lang,
                    target_lang=target_lang, memory=memory, slice_number=slice_number,
                    slice_total=slice_total, elsewhere=elsewhere, _max_tokens=_max_tokens,
                )
                second = self.translate_page(
                    original_path, cleaned_path, items[mid:], api_key=api_key, source_lang=source_lang,
                    target_lang=target_lang, memory=memory, slice_number=slice_number,
                    slice_total=slice_total, elsewhere=elsewhere, _max_tokens=_max_tokens,
                )
                return _merge_translation_results(first, second)
            if _max_tokens < 16384:
                # One box alone still cut off: the budget went on reasoning before
                # the answer, so the answer itself needs more headroom, not fewer boxes.
                return self.translate_page(
                    original_path, cleaned_path, items, api_key=api_key, source_lang=source_lang,
                    target_lang=target_lang, memory=memory, slice_number=slice_number,
                    slice_total=slice_total, elsewhere=elsewhere,
                    _max_tokens=min(16384, _max_tokens * 2),
                )
            raise

    def _translate_batch(
        self, original_path: Path, cleaned_path: Path, items: list[dict],
        *, api_key: str, source_lang: str, target_lang: str,
        memory: ChapterMemory | None = None, slice_number: int | None = None, slice_total: int | None = None,
        elsewhere: list | tuple = (), max_tokens: int = 1000000,
    ) -> VisionTranslationResult:
        original, cleaned = _read_image(original_path), _read_image(cleaned_path)
        if original.shape[:2] != cleaned.shape[:2]:
            raise ValueError("Original and cleaned slices have different dimensions")
        h, w = original.shape[:2]
        # Send short numeric ids; models copy them more reliably.
        real = {str(n): str(item["id"]) for n, item in enumerate(items, start=1)}
        objects = [{"id": alias, "source_text": item["text"], "bbox_xyxy": item["region"],
                    **({"max_chars": item["max_chars"]} if item.get("max_chars") else {})}
                   for alias, item in zip(real, items)]
        source_name = (
            "the original language shown in the image"
            if str(source_lang or "").lower() in {"", "auto"}
            else language_name(source_lang)
        )
        system = system_prompt(language_name(target_lang), target_lang, memory.settled() if memory is not None else None)
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
        # One image: the boxes are drawn on the raw slice, whose text the model reads.
        marked_b64 = _encode_for_gemini(mark_objects(original, objects, elsewhere))
        ids = set(real)
        # A cap, not a target: raising it costs nothing unless the reply actually grows.
        if self.provider.protocol == "gemini":
            result, data = self._gemini(system, prompt, marked_b64, api_key=api_key, ids=ids, max_tokens=max_tokens)
        else:
            result, data = self._openai(system, prompt, marked_b64, api_key=api_key, ids=ids, max_tokens=max_tokens)
        result, data = _unalias(result, data, real)
        ids = set(real.values())
        # The source lines the vision model read off the image. result.sources
        # is still empty here, and A.I mode candidates carry no OCR text, so
        # without this the re-translate below gets empty pairs and never runs.
        read_sources = {
            str(entry.get("id")): " ".join(str(entry.get("source") or "").split())[:1000]
            for entry in data.get("translations") or []
            if isinstance(entry, dict) and str(entry.get("source") or "").strip()
        }
        if self.translate_provider is not None and self.translate_model:
            # The vision model reads the image and judges role/container; a text model
            # re-translates the source lines it read, so the wording is the text model's.
            pairs = [
                (str(item["id"]), (read_sources.get(str(item["id"]))
                                   or str(item.get("text") or "").strip()))
                for item in items
            ]
            retold, retold_cost, retold_usage = retranslate_text(
                self.translate_provider, self.translate_model, self.translate_api_key or "",
                pairs, source_lang=source_lang, target_lang=target_lang,
                memory_snapshot=memory.snapshot(settled=False) if memory is not None else None,
                settled=memory.settled() if memory is not None else None,
            )
            # The re-translate is billed separately, so its cost and usage join the vision read's.
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
            if isinstance(entry.get("color"), str) and _HEX_COLOR.fullmatch(entry["color"].strip()):
                colors[str(entry["id"])] = entry["color"].strip().lower()
        fonts = {key: {"font_id": font_for(roles.get(key), containers.get(key), target_lang), "font_mode": "ai"}
                 for key in (result.font_choices or {})}
        if memory is not None:
            memory.update(slice_number or 0, data, result.translations, [str(item["id"]) for item in items])
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
                {"type": "text", "text": prompt},  # the ask goes last, after the images it refers to
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
        try:
            body = response.json()
            answer = _extract_output_text(body)
        except (TypeError, KeyError, ValueError) as exc:
            choice = (body.get("choices") or [{}])[0] if isinstance(body, dict) else {}
            reason = choice.get("finish_reason") if isinstance(choice, dict) else None
            salvaged = _salvage_translations(choice, ids) if reason == "length" else {}
            if salvaged:
                # Keep what the reply finished; the missing ids still route to retry downstream.
                translations, font_choices, data = _parse_vision_payload(
                    json.dumps({"translations": [
                        {"id": key, "translated_text": text} for key, text in salvaged.items()
                    ]}, ensure_ascii=False), ids)
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
