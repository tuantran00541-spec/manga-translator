from __future__ import annotations

from dataclasses import dataclass
import unicodedata


@dataclass(frozen=True)
class OCRQuality:
    status: str
    reason: str | None = None


_JA_ALIASES = {"ja", "japan"}
_ZH_ALIASES = {"ch", "zh", "chinese"}
_KO_ALIASES = {"ko", "korean"}
_EN_ALIASES = {"en", "english"}


def _normalized_lang(lang: str) -> str:
    value = (lang or "").strip().lower()
    if value in _JA_ALIASES:
        return "ja"
    if value in _ZH_ALIASES:
        return "zh"
    if value in _KO_ALIASES:
        return "ko"
    if value in _EN_ALIASES:
        return "en"
    return value


def _is_latin(ch: str) -> bool:
    name = unicodedata.name(ch, "")
    return "LATIN" in name


def _is_han(ch: str) -> bool:
    code = ord(ch)
    return (
        0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
    )


def _is_hiragana(ch: str) -> bool:
    code = ord(ch)
    return 0x3040 <= code <= 0x309F


def _is_katakana(ch: str) -> bool:
    code = ord(ch)
    return 0x30A0 <= code <= 0x30FF or 0x31F0 <= code <= 0x31FF


def _is_hangul(ch: str) -> bool:
    code = ord(ch)
    return (
        0x1100 <= code <= 0x11FF
        or 0x3130 <= code <= 0x318F
        or 0xA960 <= code <= 0xA97F
        or 0xAC00 <= code <= 0xD7AF
        or 0xD7B0 <= code <= 0xD7FF
    )


def classify_ocr_quality(
    text: str,
    lang: str,
    *,
    confidence: float | None = None,
) -> OCRQuality:
    """Classify OCR output conservatively for downstream automation.

    Only strong evidence becomes ``reject``. Ambiguous cases become ``review``
    so OCR text remains visible/editable and can still be translated when the
    user explicitly corrects it.
    """

    value = str(text or "").strip()
    if not value:
        return OCRQuality("reject", "empty")

    for ch in value:
        if unicodedata.category(ch) in {"Cc", "Cs"} and ch not in "\n\t":
            return OCRQuality("reject", "control-character")

    content = [ch for ch in value if ch.isalnum()]
    if not content:
        return OCRQuality("reject", "no-content")

    try:
        conf = None if confidence is None else float(confidence)
    except (TypeError, ValueError):
        conf = None
    if conf is not None:
        if conf < 0.35:
            return OCRQuality("reject", "very-low-confidence")
        if conf < 0.55:
            return OCRQuality("review", "low-confidence")

    letters = [ch for ch in value if unicodedata.category(ch).startswith("L")]
    latin = sum(_is_latin(ch) for ch in letters)
    han = sum(_is_han(ch) for ch in letters)
    kana = sum(_is_hiragana(ch) or _is_katakana(ch) for ch in letters)
    hangul = sum(_is_hangul(ch) for ch in letters)
    lang_key = _normalized_lang(lang)

    if lang_key == "en" and len(letters) >= 2:
        non_latin_cjk = han + kana + hangul
        if non_latin_cjk >= 2 and non_latin_cjk / len(letters) >= 0.5:
            return OCRQuality("reject", "script-mismatch")

    if lang_key == "ja" and len(letters) >= 3:
        if han + kana == 0 and latin / len(letters) >= 0.85:
            return OCRQuality("review", "unexpected-script")

    if lang_key == "zh" and len(letters) >= 3:
        if han == 0 and latin / len(letters) >= 0.85:
            return OCRQuality("review", "unexpected-script")

    if lang_key == "ko" and len(letters) >= 3:
        if hangul == 0 and latin / len(letters) >= 0.85:
            return OCRQuality("review", "unexpected-script")

    compact = [ch.casefold() for ch in content]
    if len(compact) >= 5 and len(set(compact)) == 1:
        return OCRQuality("review", "repeated-character")

    symbols = [
        ch
        for ch in value
        if not ch.isspace()
        and not ch.isalnum()
        and not unicodedata.category(ch).startswith("P")
    ]
    visible = [ch for ch in value if not ch.isspace()]
    if visible and len(symbols) / len(visible) > 0.45 and len(content) <= 2:
        return OCRQuality("reject", "symbol-noise")

    return OCRQuality("good", None)


def should_block_translation(obj: dict) -> bool:
    """Block only untouched machine OCR that was classified as reject.

    Auto-generated text objects preserve ``auto_ocr_text``. Once a user edits
    ``ocr_text`` away from that value, the manual correction is allowed through
    even if the original machine OCR carried a reject classification.
    """

    if str(obj.get("ocr_quality") or "").strip().lower() != "reject":
        return False
    if not obj.get("auto_generated"):
        return False
    current = str(obj.get("ocr_text") or "").strip()
    automatic = str(obj.get("auto_ocr_text") or "").strip()
    return bool(current) and current == automatic
