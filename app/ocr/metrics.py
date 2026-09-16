"""Ground-truth metrics for OCR experiments.

These helpers are deliberately dependency-light and independent of any OCR
runtime.  They make it possible to compare a batch candidate with the current
reader without treating confidence or a ``good`` label as ground truth.
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any


_SPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace while preserving punctuation."""
    value = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return _SPACE_RE.sub(" ", value).strip()


def _edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for ref_index, ref_value in enumerate(reference, start=1):
        current = [ref_index]
        for hyp_index, hyp_value in enumerate(hypothesis, start=1):
            substitution = previous[hyp_index - 1] + int(ref_value != hyp_value)
            insertion = current[hyp_index - 1] + 1
            deletion = previous[hyp_index] + 1
            current.append(min(substitution, insertion, deletion))
        previous = current
    return int(previous[-1])


def character_error_rate(reference: str, hypothesis: str) -> float:
    ref = list(normalize_text(reference).replace(" ", ""))
    hyp = list(normalize_text(hypothesis).replace(" ", ""))
    return _edit_distance(ref, hyp) / float(max(1, len(ref)))


def _word_tokens(text: str) -> list[str]:
    normalized = normalize_text(text)
    tokens = normalized.split()
    # CJK and single-token scripts commonly have no word separators. Character
    # tokens provide a useful WER-like signal instead of reporting 0/1 only.
    if len(tokens) <= 1 and len(normalized.replace(" ", "")) > 1:
        return list(normalized.replace(" ", ""))
    return tokens


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref = _word_tokens(reference)
    hyp = _word_tokens(hypothesis)
    return _edit_distance(ref, hyp) / float(max(1, len(ref)))


def line_recall(reference: str, hypothesis: str) -> float:
    ref_lines = [normalize_text(line) for line in str(reference or "").splitlines()]
    ref_lines = [line for line in ref_lines if line]
    hyp_lines = [normalize_text(line) for line in str(hypothesis or "").splitlines()]
    hyp_lines = [line for line in hyp_lines if line]
    if not ref_lines:
        return 1.0 if not hyp_lines else 0.0
    remaining = list(hyp_lines)
    matched = 0
    for line in ref_lines:
        try:
            index = remaining.index(line)
        except ValueError:
            continue
        matched += 1
        del remaining[index]
    return matched / float(len(ref_lines))


def summarize_ocr_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    """Return aggregate CER/WER/line recall for ``reference``/``hypothesis`` rows."""
    rows = list(rows)
    if not rows:
        return {
            "count": 0,
            "cer": 0.0,
            "wer": 0.0,
            "line_recall": 0.0,
        }
    references = [str(row.get("reference") or "") for row in rows]
    hypotheses = [str(row.get("hypothesis") or "") for row in rows]
    ref_chars = sum(len(normalize_text(value).replace(" ", "")) for value in references)
    ref_words = sum(len(_word_tokens(value)) for value in references)
    cer_errors = sum(
        _edit_distance(
            list(normalize_text(reference).replace(" ", "")),
            list(normalize_text(hypothesis).replace(" ", "")),
        )
        for reference, hypothesis in zip(references, hypotheses)
    )
    wer_errors = sum(
        _edit_distance(_word_tokens(reference), _word_tokens(hypothesis))
        for reference, hypothesis in zip(references, hypotheses)
    )
    return {
        "count": len(rows),
        "cer": round(cer_errors / float(max(1, ref_chars)), 6),
        "wer": round(wer_errors / float(max(1, ref_words)), 6),
        "line_recall": round(
            sum(
                line_recall(reference, hypothesis)
                for reference, hypothesis in zip(references, hypotheses)
            )
            / float(len(rows)),
            6,
        ),
    }


__all__ = [
    "character_error_rate",
    "line_recall",
    "normalize_text",
    "summarize_ocr_rows",
    "word_error_rate",
]
