""" Deterministic CPU font matching for original lettering crops. """

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.render.font_catalog import FontRecord, load_font_catalog

__all__ = ["FontMatch", "clear_match_caches", "match_fonts"]


_SAMPLE_SIZE = (128, 64)


@dataclass(frozen=True)
class FontMatch:
    font_id: str
    score: float
    confidence: str
    evidence: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "font_id": self.font_id,
            "score": round(float(self.score), 6),
            "confidence": self.confidence,
            "evidence": self.evidence,
        }


def _crop_image(image: Image.Image | None, region: tuple[int, int, int, int] | None) -> Image.Image | None:
    if image is None:
        return None
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    image = image.convert("L")
    if region is None:
        return image
    x1, y1, x2, y2 = (int(value) for value in region)
    x1, x2 = sorted((max(0, x1), max(0, x2)))
    y1, y2 = sorted((max(0, y1), max(0, y2)))
    x2, y2 = min(image.width, x2), min(image.height, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return image.crop((x1, y1, x2, y2))


def _normalize(image: Image.Image) -> np.ndarray:
    image = image.resize(_SAMPLE_SIZE, Image.Resampling.LANCZOS)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    ink = 1.0 - arr
    # Projection profiles retain character spacing and weight while being more
    # robust to anti-aliasing than a hard threshold.
    rows = ink.mean(axis=1)
    cols = ink.mean(axis=0)
    row_edges = np.abs(np.diff(rows, prepend=rows[:1]))
    col_edges = np.abs(np.diff(cols, prepend=cols[:1]))
    bbox = np.argwhere(ink > 0.18)
    if bbox.size:
        y1, x1 = bbox.min(axis=0)
        y2, x2 = bbox.max(axis=0)
        width_ratio = (x2 - x1 + 1) / _SAMPLE_SIZE[0]
        height_ratio = (y2 - y1 + 1) / _SAMPLE_SIZE[1]
    else:
        width_ratio = height_ratio = 0.0
    return np.concatenate(
        [
            rows,
            cols,
            row_edges,
            col_edges,
            np.asarray([float(ink.mean()), width_ratio, height_ratio], dtype=np.float32),
        ]
    )


def _fit_font(font_path: Path, text: str) -> Image.Image:
    canvas = Image.new("L", _SAMPLE_SIZE, 255)
    draw = ImageDraw.Draw(canvas)
    text = str(text or "Ag")[:160]
    chosen = ImageFont.truetype(str(font_path), 48)
    for size in range(48, 7, -1):
        candidate = ImageFont.truetype(str(font_path), size)
        bbox = draw.textbbox((0, 0), text, font=candidate, stroke_width=0)
        if bbox[2] - bbox[0] <= _SAMPLE_SIZE[0] - 8 and bbox[3] - bbox[1] <= _SAMPLE_SIZE[1] - 8:
            chosen = candidate
            break
    bbox = draw.textbbox((0, 0), text, font=chosen)
    x = (_SAMPLE_SIZE[0] - (bbox[2] - bbox[0])) // 2 - bbox[0]
    y = (_SAMPLE_SIZE[1] - (bbox[3] - bbox[1])) // 2 - bbox[1]
    draw.text((x, y), text, font=chosen, fill=0)
    return canvas


@lru_cache(maxsize=512)
def _font_descriptor(font_path_str: str, source_text: str) -> np.ndarray:
    return _normalize(_fit_font(Path(font_path_str), source_text))


def _similarity(target: np.ndarray, candidate: np.ndarray) -> tuple[float, dict[str, float]]:
    scale = np.maximum(target.std() * candidate.std(), 1e-6)
    correlation = float(np.clip(np.mean((target - target.mean()) * (candidate - candidate.mean())) / scale, -1.0, 1.0))
    correlation_score = (correlation + 1.0) / 2.0
    distance = float(np.mean(np.abs(target - candidate)))
    distance_score = float(np.clip(1.0 - distance / 0.8, 0.0, 1.0))
    score = 0.65 * correlation_score + 0.35 * distance_score
    return score, {"correlation": round(correlation_score, 4), "geometry_distance": round(distance, 4)}


def _confidence(score: float, gap: float) -> str:
    if score >= 0.78 and gap >= 0.05:
        return "high"
    if score >= 0.55 and gap >= 0.02:
        return "medium"
    return "low"


def clear_match_caches() -> None:
    _font_descriptor.cache_clear()


def _fallback(records: list[FontRecord], top_k: int) -> list[FontMatch]:
    return [
        FontMatch(record.id, 0.0, "low", {"reason": "source_crop_unavailable"})
        for record in records[:top_k]
    ]


def match_fonts(
    image: Image.Image | None,
    region: tuple[int, int, int, int] | None,
    source_text: str | None,
    *,
    category: str | None = None,
    top_k: int = 3,
) -> list[FontMatch]:
    """Return the closest bundled fonts, ranked deterministically."""

    top_k = max(1, min(int(top_k), 5))
    records = [record for record in load_font_catalog().records if not category or record.category == category]
    records = sorted(records, key=lambda record: record.default_rank)
    if not records:
        return []
    crop = _crop_image(image, region)
    if crop is None or crop.width < 2 or crop.height < 2:
        return _fallback(records, top_k)

    text = str(source_text or "Ag")
    target = _normalize(crop)
    scored: list[tuple[float, FontRecord, dict[str, float]]] = []
    for record in records:
        try:
            candidate = _font_descriptor(str(record.path), text)
        except (OSError, ValueError):
            continue
        score, evidence = _similarity(target, candidate)
        scored.append((score, record, evidence))
    scored.sort(key=lambda item: (-item[0], item[1].default_rank, item[1].id))
    if not scored:
        return _fallback(records, top_k)
    best_score = scored[0][0]
    return [
        FontMatch(
            record.id,
            float(score),
            _confidence(float(score), float(score - best_score if index == 0 else scored[index - 1][0] - score)),
            {**evidence, "category": record.category, "source_text": text},
        )
        for index, (score, record, evidence) in enumerate(scored[:top_k])
    ]
