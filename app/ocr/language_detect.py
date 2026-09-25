"""Detect a chapter's source language from its own text regions.

The OCR engine depends on the source language (manga-ocr for Japanese, the
Korean Paddle recogniser for Korean, the unified Paddle recogniser for Chinese
and English), so the language must be known before OCR. Instead of asking the
user, sample a few detector-approved text regions, read each one with the
unified recogniser (Chinese / Japanese / English) and the Korean recogniser,
and vote by Unicode script. Vertical Japanese can come back from Paddle as
kanji-only noise, so a Chinese or inconclusive vote is re-checked with
manga-ocr: Japanese dialogue carries kana in nearly every bubble, Chinese none.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import urlparse

import cv2
import numpy as np

from app.config import RAW_DIR
from app.image_io import read_image
from app.region_policy import geometry_center_in_regions, page_preserve_regions
from app.security import validate_managed_path

SOURCE_LANGS = ("ja", "ch", "korean", "en")
_ALIASES = {"japan": "ja", "zh": "ch", "ko": "korean", "english": "en"}

MAX_SAMPLES = 8
MAX_SAMPLE_PAGES = 6
MIN_LETTERS = 8
MIN_SHARE = 0.6
CROP_PADDING = 6

# Chapter sources whose language is fixed. Used only when the text itself is
# inconclusive (e.g. a chapter that is almost all artwork).
SITE_LANGUAGE_HINTS = {
    "asuracomic.net": "en",
    "asurascans.com": "en",
    "asuratoon.com": "en",
}


def normalize_source_lang(value: object) -> str | None:
    lang = str(value or "").strip().lower()
    lang = _ALIASES.get(lang, lang)
    return lang if lang in SOURCE_LANGS else None


def script_counts(text: str) -> dict[str, int]:
    counts = {"hangul": 0, "kana": 0, "han": 0, "latin": 0}
    for ch in str(text or ""):
        code = ord(ch)
        if 0xAC00 <= code <= 0xD7A3 or 0x1100 <= code <= 0x11FF or 0x3130 <= code <= 0x318F:
            counts["hangul"] += 1
        elif 0x3040 <= code <= 0x30FF or 0x31F0 <= code <= 0x31FF or 0xFF66 <= code <= 0xFF9D:
            counts["kana"] += 1
        elif 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF or 0xF900 <= code <= 0xFAFF:
            counts["han"] += 1
        elif ch.isascii() and ch.isalpha():
            counts["latin"] += 1
    return counts


@dataclass(frozen=True)
class ProbeReading:
    """One text region read by both recognisers."""

    unified_text: str
    unified_confidence: float | None
    korean_text: str
    korean_confidence: float | None


@dataclass(frozen=True)
class LanguageDetection:
    lang: str | None
    confidence: float
    samples: int
    counts: dict[str, int] = field(default_factory=dict)
    reason: str = ""

    def as_dict(self) -> dict:
        return {
            "lang": self.lang,
            "confidence": round(self.confidence, 3),
            "samples": self.samples,
            "counts": dict(self.counts),
            "reason": self.reason,
        }


def _score(confidence: float | None) -> float:
    return 0.5 if confidence is None else float(confidence)


def classify_readings(readings: Iterable[ProbeReading]) -> LanguageDetection:
    """Each region casts one vote, so dense scripts (one Hangul/Han glyph per
    syllable or word) are not outweighed by letter-heavy Latin text."""
    votes = {"hangul": 0, "kana": 0, "han": 0, "latin": 0}
    letters = 0
    for reading in readings:
        unified = script_counts(reading.unified_text)
        korean = script_counts(reading.korean_text)
        unified_letters = unified["kana"] + unified["han"] + unified["latin"]
        if not korean["hangul"] and not unified_letters:
            continue
        # Each recogniser produces confident text only for its own scripts; on
        # foreign text it returns low-confidence noise. Keep the stronger read.
        korean_wins = korean["hangul"] > 0 and (
            not unified_letters
            or _score(reading.korean_confidence) >= _score(reading.unified_confidence)
        )
        if korean_wins:
            votes["hangul"] += 1
            letters += korean["hangul"] + korean["latin"]
        else:
            letters += unified_letters
            if unified["kana"]:
                votes["kana"] += 1
            elif unified["han"] >= unified["latin"]:
                votes["han"] += 1
            else:
                votes["latin"] += 1

    samples = sum(votes.values())
    if letters < MIN_LETTERS or not samples:
        return LanguageDetection(None, 0.0, samples, votes, "not-enough-text")

    cjk = votes["kana"] + votes["han"]
    ranked = {"korean": votes["hangul"], "cjk": cjk, "en": votes["latin"]}
    winner = max(ranked, key=ranked.get)
    share = ranked[winner] / samples
    if winner == "cjk":
        # Japanese dialogue mixes kana into kanji, so kanji-only bubbles still
        # count as Japanese once enough other bubbles carry kana. Chinese has none.
        winner = "ja" if votes["kana"] >= max(1, 0.2 * cjk) else "ch"
    if share < MIN_SHARE:
        return LanguageDetection(None, share, samples, votes, "mixed-scripts")
    return LanguageDetection(winner, share, samples, votes, "script-vote")


def refine_with_japanese_reader(first: LanguageDetection, japanese_texts: list[str]) -> LanguageDetection:
    """Promote to Japanese when manga-ocr finds kana in most sampled bubbles."""
    read = [text for text in japanese_texts if str(text or "").strip()]
    if not read:
        return first
    kana_bubbles = sum(1 for text in read if script_counts(text)["kana"] >= 2)
    share = kana_bubbles / len(read)
    if share >= 0.5:
        return LanguageDetection("ja", share, len(read), {**first.counts, "kana_bubbles": kana_bubbles}, "manga-ocr-kana")
    return first


def _needs_japanese_check(detection: LanguageDetection) -> bool:
    if detection.lang == "ch":
        return True
    return detection.lang is None and not detection.counts.get("hangul")


def site_language_hint(source_url: object) -> str | None:
    host = (urlparse(str(source_url or "")).hostname or "").lower()
    for domain, lang in SITE_LANGUAGE_HINTS.items():
        if host == domain or host.endswith("." + domain):
            return lang
    return None


def _box_is_dialogue_sample(box: object, preserve_regions: list[dict]) -> bool:
    from app.ocr.service import ocr_target_skip_reason

    if not isinstance(box, dict) or box.get("removed") or box.get("ocr_eligible") is False:
        return False
    if geometry_center_in_regions(box, preserve_regions) or ocr_target_skip_reason(box):
        return False
    try:
        x1, y1, x2, y2 = (int(box[key]) for key in ("x1", "y1", "x2", "y2"))
    except (KeyError, TypeError, ValueError):
        return False
    return x2 - x1 >= 12 and y2 - y1 >= 12


def sample_regions(manifest: dict, limit: int = MAX_SAMPLES) -> list[tuple[int, dict]]:
    """Pick the largest dialogue boxes, spread round-robin over the first pages."""
    per_page: list[list[tuple[int, dict]]] = []
    for page_index, page in enumerate(manifest.get("pages") or []):
        if len(per_page) >= MAX_SAMPLE_PAGES:
            break
        if not isinstance(page, dict) or page.get("skipped") or page.get("process_required"):
            continue
        if not page.get("original"):
            continue
        preserve = page_preserve_regions(page)
        boxes = [box for box in page.get("boxes") or [] if _box_is_dialogue_sample(box, preserve)]
        if not boxes:
            continue
        boxes.sort(key=lambda b: (int(b["x2"]) - int(b["x1"])) * (int(b["y2"]) - int(b["y1"])), reverse=True)
        per_page.append([(page_index, box) for box in boxes])

    picked: list[tuple[int, dict]] = []
    depth = 0
    while len(picked) < limit and any(depth < len(boxes) for boxes in per_page):
        for boxes in per_page:
            if depth < len(boxes) and len(picked) < limit:
                picked.append(boxes[depth])
        depth += 1
    return picked


def _crop(image: np.ndarray, box: dict) -> np.ndarray:
    h, w = image.shape[:2]
    x1 = max(0, int(box["x1"]) - CROP_PADDING)
    y1 = max(0, int(box["y1"]) - CROP_PADDING)
    x2 = min(w, int(box["x2"]) + CROP_PADDING)
    y2 = min(h, int(box["y2"]) + CROP_PADDING)
    return cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)


Reader = Callable[[np.ndarray, str], object]


def detect_manifest_language(chapter_id: str, manifest: dict, reader: Reader) -> LanguageDetection:
    """Read sampled regions with ``reader(rgb, lang)`` and vote on the script."""
    images: dict[int, np.ndarray] = {}
    readings: list[ProbeReading] = []
    crops: list[np.ndarray] = []
    pages = manifest.get("pages") or []
    for page_index, box in sample_regions(manifest):
        if page_index not in images:
            path: Path = validate_managed_path(pages[page_index]["original"], RAW_DIR / chapter_id)
            images[page_index] = read_image(path)
        crop = _crop(images[page_index], box)
        if not crop.size:
            continue
        crops.append(crop)
        unified = reader(crop, "ch")
        korean = reader(crop, "korean")
        readings.append(ProbeReading(
            str(getattr(unified, "text", "") or ""), getattr(unified, "confidence", None),
            str(getattr(korean, "text", "") or ""), getattr(korean, "confidence", None),
        ))
    detection = classify_readings(readings)
    if crops and _needs_japanese_check(detection):
        japanese = [str(getattr(reader(crop, "ja"), "text", "") or "") for crop in crops]
        detection = refine_with_japanese_reader(detection, japanese)
    if detection.lang is None:
        hinted = site_language_hint(manifest.get("source_url"))
        if hinted:
            return LanguageDetection(hinted, 1.0, detection.samples, detection.counts, "site-hint")
    return detection
