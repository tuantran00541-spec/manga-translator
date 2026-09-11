#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw

from app.config import DEFAULT_FONT
from app.parameters import (
    RENDER_DEFAULT_PADDING,
    RENDER_PADDING_RATIO_MAX,
)
from app.render.text_renderer import _fits

# Final render sizes are always explicit numbers.  We intentionally do not use
# a small set of 28/32/40/48 buckets: the size is derived per object from the
# amount of text and the geometry of its reviewed text region.
MIN_CANDIDATE_FONT_SIZE = 16
MAX_CANDIDATE_FONT_SIZE = 96
SOFT_READABILITY_WARNING_SIZE = 22

SYSTEM_UI_IDS = {
    "text_f8805da6a21c4d26",
    "text_18ebdaaab3804f0f",
    "text_e23413aa1e0e49dc",
    "text_6397b163920e4216",
    "text_7ca5d0527da84af5",
    "text_473d9887641b4026",
    "text_9edf052182db40ad",
    "text_8afdc7f11f1142b0",
    "text_29f55d4796e84bc1",
    "text_643c5dc7b65f4350",
    "text_388f85a855394323",
    "text_9070e21b65a44a72",
    "text_e432a813a33b417e",
    "text_d5b806e496784aac",
    "text_54b93201c49b4ac7",
    "text_64b56e211cab46fb",
    "text_5562636cb2f34195",
    "text_a3771febd50a45ce",
    "text_b478707b63c54344",
    "text_6b6e678650b64f71",
    "text_ca9d8fbd11e94caa",
    "text_32e576d63cf64ae1",
    "text_56e1a72d660e47fe",
}

DENSE_STATUS_IDS = {
    "text_388f85a855394323",
    "text_9070e21b65a44a72",
    "text_e432a813a33b417e",
    "text_d5b806e496784aac",
    "text_54b93201c49b4ac7",
    "text_64b56e211cab46fb",
    "text_5562636cb2f34195",
    "text_a3771febd50a45ce",
    "text_b478707b63c54344",
    "text_6b6e678650b64f71",
    "text_ca9d8fbd11e94caa",
    "text_32e576d63cf64ae1",
    "text_56e1a72d660e47fe",
}

SKILL_IDS = {
    "text_65fea25a175344b9",
    "text_8cda75cd20ec4218",
    "text_407467da00d74a1a",
    "text_647d2ac741d84416",
}

EMPHASIS_IDS = {
    "text_5d19f1df1c074fe7",
    "text_ea1c06a76fa94882",
    "text_9941759390bf4116",
    "text_c176d66b1ce44c9b",
    "text_c571f8cf5518485b",
}

STATUS_LEFT_ALIGN_IDS = set(DENSE_STATUS_IDS)

PRIMARY_FONT_CANDIDATES = [
    "Manga-fonts.ttf",
    "Mac-dinh-2.ttf",
    "Mac-dinh-3.ttf",
    "default.ttf",
]
SYSTEM_FONT_CANDIDATES = [
    "Skill-fonts-1.ttf",
    "Granite.ttf",
    "default.ttf",
]
SKILL_FONT_CANDIDATES = [
    "Skill-fonts-2.ttf",
    "Skill-fonts-1.ttf",
    "Shadow-fonts.ttf",
    "default.ttf",
]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def visible_codepoints(texts: list[str]) -> set[int]:
    return {ord(ch) for text in texts for ch in text if not ch.isspace()}


def cmap_codepoints(path: Path) -> set[int]:
    font = TTFont(str(path), lazy=True)
    try:
        cmap = font.getBestCmap() or {}
        return set(cmap)
    finally:
        font.close()


def choose_global_font(
    font_dir: Path,
    candidates: list[str],
    texts: list[str],
    *,
    fallback: Path | None = None,
):
    required = visible_codepoints(texts)
    checked = []
    for name in candidates:
        path = font_dir / name
        if not path.is_file():
            checked.append({"font": name, "exists": False, "missing": len(required)})
            continue
        cmap = cmap_codepoints(path)
        missing = required - cmap
        checked.append({"font": name, "exists": True, "missing": len(missing)})
        if not missing:
            return path, checked
    if fallback is not None and fallback.is_file():
        cmap = cmap_codepoints(fallback)
        missing = required - cmap
        checked.append(
            {"font": fallback.name, "exists": True, "missing": len(missing), "fallback": True}
        )
        if not missing:
            return fallback, checked
    return None, checked


def semantic_kind(oid: str) -> str:
    if oid in DENSE_STATUS_IDS:
        return "status_ui"
    if oid in SYSTEM_UI_IDS:
        return "system_ui"
    if oid in SKILL_IDS:
        return "skill_attack"
    if oid in EMPHASIS_IDS:
        return "emphasis"
    return "dialogue"


def text_load(text: str) -> dict[str, float | int]:
    words = [w for w in text.replace("\n", " ").split() if w]
    word_count = len(words)
    char_count = sum(1 for ch in text if not ch.isspace())
    longest_word = max((len(w) for w in words), default=0)
    avg_word_length = (sum(len(w) for w in words) / word_count) if word_count else 0.0
    return {
        "word_count": word_count,
        "char_count": char_count,
        "longest_word": longest_word,
        "avg_word_length": round(avg_word_length, 2),
    }


def word_count_height_ratio(word_count: int) -> float:
    """Comfort ceiling: short text may be large; dense text starts smaller.

    This is not the final size.  The exact font metrics still have to fit the
    reviewed region, so both word load and bubble geometry participate.
    """
    if word_count <= 2:
        return 0.72
    if word_count <= 5:
        return 0.60
    if word_count <= 9:
        return 0.50
    if word_count <= 14:
        return 0.42
    if word_count <= 22:
        return 0.35
    if word_count <= 32:
        return 0.30
    return 0.26


def sizing_window(text: str, box_w: int, box_h: int) -> dict[str, float | int]:
    load = text_load(text)
    word_count = int(load["word_count"])
    char_count = int(load["char_count"])
    aspect = box_w / max(1, box_h)
    area = max(1, box_w * box_h)

    # Start from a word-count-dependent fraction of bubble height.  Wide speech
    # balloons can tolerate a slightly larger ceiling because wrapping is cheap;
    # very narrow balloons get a mild reduction to avoid staircase wrapping.
    ratio = word_count_height_ratio(word_count)
    if aspect >= 2.4:
        ratio *= 1.10
    elif aspect <= 0.75:
        ratio *= 0.90

    comfort_ceiling = int(round(box_h * ratio))

    # Area scaling keeps a tiny crop from proposing absurdly large type while
    # still allowing 50-80px lettering on genuinely large, sparse bubbles.
    area_ceiling = int(round(math.sqrt(area) * 0.42))
    ceiling = max(
        MIN_CANDIDATE_FONT_SIZE,
        min(MAX_CANDIDATE_FONT_SIZE, comfort_ceiling, max(24, area_ceiling)),
    )

    # Readability floor is adaptive rather than a fixed 28px rule.  It only
    # rejects extremely small type; it never forces every object toward 40/48.
    short_side = min(box_w, box_h)
    adaptive_floor = int(round(short_side * 0.11))
    adaptive_floor = max(MIN_CANDIDATE_FONT_SIZE, min(24, adaptive_floor))
    adaptive_floor = min(adaptive_floor, ceiling)

    density = char_count / math.sqrt(area)
    return {
        **load,
        "box_aspect_ratio": round(aspect, 3),
        "box_area": area,
        "text_density": round(density, 4),
        "comfort_ceiling": ceiling,
        "adaptive_floor": adaptive_floor,
    }


def choose_fixed_size(
    draw: ImageDraw.ImageDraw,
    text: str,
    box_w: int,
    box_h: int,
    font_path: Path,
    stroke_width: int,
) -> tuple[int | None, list[str] | None, dict[str, float | int]]:
    stats = sizing_window(text, box_w, box_h)
    ceiling = int(stats["comfort_ceiling"])
    floor = int(stats["adaptive_floor"])

    # Continuous integer search, not coarse buckets.  The chosen numeric size is
    # written into the manifest, so final rendering never depends on auto-size.
    for size in range(ceiling, floor - 1, -1):
        ok, lines = _fits(draw, text, box_w, box_h, str(font_path), size, stroke_width)
        if ok:
            return size, lines, stats
    return None, None, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--translation-root", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    source_root = Path(args.translation_root)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    manifest = load_json(source_root / "processed-manifest-translated.json")
    translation_summary = load_json(source_root / "translation-review-summary.json")

    failures: list[str] = []
    if translation_summary.get("status") != "PASS":
        failures.append("translation checkpoint is not PASS")
    if translation_summary.get("active_untranslated_story_objects") != 0:
        failures.append("translation checkpoint still has untranslated story objects")

    active = []
    tombstones = []
    for page_index, page in enumerate(manifest.get("pages") or []):
        for obj in page.get("text_objects") or []:
            if not isinstance(obj, dict):
                continue
            if obj.get("source_missing"):
                tombstones.append((page_index, page, obj))
            else:
                active.append((page_index, page, obj))

    if len(active) != 98:
        failures.append(f"expected 98 active story objects, got {len(active)}")

    font_dir = DEFAULT_FONT.parent
    dialogue_texts = [
        str(obj.get("translation") or "")
        for _, _, obj in active
        if semantic_kind(str(obj.get("id"))) in {"dialogue", "emphasis"}
    ]
    system_texts = [
        str(obj.get("translation") or "")
        for _, _, obj in active
        if semantic_kind(str(obj.get("id"))) in {"system_ui", "status_ui"}
    ]
    skill_texts = [
        str(obj.get("translation") or "")
        for _, _, obj in active
        if semantic_kind(str(obj.get("id"))) == "skill_attack"
    ]

    primary_font, primary_checks = choose_global_font(
        font_dir, PRIMARY_FONT_CANDIDATES, dialogue_texts, fallback=DEFAULT_FONT
    )
    if primary_font is None:
        failures.append("no primary dialogue font covers all Vietnamese glyphs")
        primary_font = DEFAULT_FONT
    system_font, system_checks = choose_global_font(
        font_dir, SYSTEM_FONT_CANDIDATES, system_texts, fallback=primary_font
    )
    if system_font is None:
        failures.append("no system/status font covers all assigned glyphs")
        system_font = primary_font
    skill_font, skill_checks = choose_global_font(
        font_dir, SKILL_FONT_CANDIDATES, skill_texts, fallback=primary_font
    )
    if skill_font is None:
        failures.append("no skill font covers all assigned glyphs")
        skill_font = primary_font

    font_cmaps = {
        str(primary_font): cmap_codepoints(primary_font),
        str(system_font): cmap_codepoints(system_font),
        str(skill_font): cmap_codepoints(skill_font),
    }

    canvas = Image.new("RGB", (8, 8), "white")
    draw = ImageDraw.Draw(canvas)
    plan_objects = {}
    overflow = []
    glyph_missing = []
    readability_warnings = []
    size_counter: Counter[int] = Counter()
    font_counter: Counter[str] = Counter()
    kind_counter: Counter[str] = Counter()

    for page_index, page, obj in active:
        oid = str(obj.get("id"))
        text = str(obj.get("translation") or "").strip()
        if not text:
            failures.append(f"{oid}: empty translation")
            continue
        region = obj.get("region") or {}
        try:
            x1, y1, x2, y2 = [int(region[k]) for k in ("x1", "y1", "x2", "y2")]
        except Exception:
            failures.append(f"{oid}: malformed region")
            continue
        raw_w = x2 - x1
        raw_h = y2 - y1
        if raw_w <= 0 or raw_h <= 0:
            failures.append(f"{oid}: invalid region dimensions")
            continue

        kind = semantic_kind(oid)
        kind_counter[kind] += 1
        if kind in {"system_ui", "status_ui"}:
            font_path = system_font
        elif kind == "skill_attack":
            font_path = skill_font
        else:
            font_path = primary_font

        missing = sorted(
            {ord(ch) for ch in text if not ch.isspace()} - font_cmaps[str(font_path)]
        )
        if missing:
            glyph_missing.append(
                {"id": oid, "font": font_path.name, "codepoints": missing}
            )
            failures.append(f"{oid}: assigned font lacks glyphs")
            continue

        pad = max(
            2,
            min(
                RENDER_DEFAULT_PADDING,
                int(min(raw_w, raw_h) * RENDER_PADDING_RATIO_MAX),
            ),
        )
        box_w = raw_w - pad * 2
        box_h = raw_h - pad * 2
        stroke_width = 2
        picked_size, picked_lines, sizing = choose_fixed_size(
            draw,
            text,
            box_w,
            box_h,
            font_path,
            stroke_width,
        )
        if picked_size is None:
            overflow.append(
                {
                    "id": oid,
                    "page_index": page_index,
                    "source_page": page.get("source_page"),
                    "slice_index": page.get("slice_index"),
                    "region": region,
                    "semantic": kind,
                    "text": text,
                    "sizing": sizing,
                }
            )
            continue

        if picked_size < SOFT_READABILITY_WARNING_SIZE:
            readability_warnings.append(
                {
                    "id": oid,
                    "page_index": page_index,
                    "font_size": picked_size,
                    "word_count": sizing["word_count"],
                    "region": region,
                }
            )

        horizontal_align = "left" if oid in STATUS_LEFT_ALIGN_IDS else "center"
        vertical_align = "top" if oid in STATUS_LEFT_ALIGN_IDS else "middle"
        is_bold = kind in {"system_ui", "status_ui", "skill_attack", "emphasis"}
        style = {
            "font": font_path.stem,
            "fontSize": str(picked_size),
            "bold": is_bold,
            "color": "auto",
            "strokeWidth": str(stroke_width),
            "strokeColor": "auto",
            "bgColor": "transparent",
            "cornerRadius": "0",
            "horizontalAlign": horizontal_align,
            "verticalAlign": vertical_align,
        }
        obj["style"] = style
        plan_objects[oid] = {
            "page_index": page_index,
            "source_page": page.get("source_page"),
            "slice_index": page.get("slice_index"),
            "semantic": kind,
            "region": region,
            "translation": text,
            "style": style,
            "wrapped_lines": picked_lines,
            "sizing": {
                **sizing,
                "selected_font_size": picked_size,
                "wrapped_line_count": len(picked_lines or []),
            },
            "fit": {
                "box_width": box_w,
                "box_height": box_h,
                "padding": pad,
                "passed": True,
            },
        }
        size_counter[picked_size] += 1
        font_counter[font_path.stem] += 1

    if overflow:
        failures.append(
            f"{len(overflow)} object(s) cannot fit within their adaptive readability window"
        )
    if glyph_missing:
        failures.append(f"{len(glyph_missing)} object(s) have missing glyphs")
    if len(plan_objects) != len(active):
        failures.append(f"styled {len(plan_objects)}/{len(active)} active objects")

    auto_font_sizes = [
        oid
        for oid, item in plan_objects.items()
        if str((item.get("style") or {}).get("fontSize", "")).lower() == "auto"
    ]
    min_size = min(
        (int(item["style"]["fontSize"]) for item in plan_objects.values()),
        default=None,
    )
    max_size = max(
        (int(item["style"]["fontSize"]) for item in plan_objects.values()),
        default=None,
    )
    if auto_font_sizes:
        failures.append("fontSize:auto remained in active typography plan")

    status = "PASS" if not failures else "FAIL"
    manifest["typography_review"] = {
        "status": status,
        "source_checkpoint": "04-translation-review",
        "active_story_objects": len(active),
        "styled_story_objects": len(plan_objects),
        "font_size_auto_count": len(auto_font_sizes),
        "minimum_font_size": min_size,
        "maximum_font_size": max_size,
        "overflow_count": len(overflow),
        "missing_glyph_count": len(glyph_missing),
        "readability_warning_count": len(readability_warnings),
        "sizing_policy": "word-count + bubble-geometry adaptive fixed sizing",
    }

    typography_plan = {
        "chapter_id": manifest.get("chapter_id"),
        "checkpoint": "05-typeset-preflight",
        "status": status,
        "policy": {
            "sizing_mode": "adaptive_fixed_per_object",
            "sizing_inputs": [
                "word_count",
                "character_count",
                "text_density",
                "bubble_width",
                "bubble_height",
                "bubble_aspect_ratio",
                "actual_font_metrics",
            ],
            "candidate_font_size_range": [
                MIN_CANDIDATE_FONT_SIZE,
                MAX_CANDIDATE_FONT_SIZE,
            ],
            "font_size_buckets": None,
            "font_size_auto_allowed": False,
            "primary_dialogue_font_stable": True,
            "semantic_font_variants": ["system_ui", "status_ui", "skill_attack"],
            "overflow_policy": "FAIL; shorten/rebreak text or safely expand region rather than silently shrinking below the adaptive readability floor",
        },
        "fonts": {
            "primary_dialogue": primary_font.stem,
            "system_ui": system_font.stem,
            "skill_attack": skill_font.stem,
            "selection_checks": {
                "primary": primary_checks,
                "system": system_checks,
                "skill": skill_checks,
            },
        },
        "counts": {
            "active_story_objects": len(active),
            "styled_story_objects": len(plan_objects),
            "tombstones_excluded": len(tombstones),
            "font_size_auto": len(auto_font_sizes),
            "overflow": len(overflow),
            "missing_glyph": len(glyph_missing),
            "readability_warnings": len(readability_warnings),
        },
        "font_size_range_selected": {
            "minimum": min_size,
            "maximum": max_size,
        },
        "size_histogram": {
            str(k): v for k, v in sorted(size_counter.items(), reverse=True)
        },
        "font_usage": dict(sorted(font_counter.items())),
        "semantic_usage": dict(sorted(kind_counter.items())),
        "overflow_objects": overflow,
        "readability_warning_objects": readability_warnings,
        "missing_glyph_objects": glyph_missing,
        "objects": plan_objects,
        "failures": failures,
        "next_action": (
            "RENDER_FROM_APPROVED_CLEAN" if status == "PASS" else "LOCAL_TYPESET_REPAIR"
        ),
    }

    dump_json(out / "typography-plan.json", typography_plan)
    dump_json(out / "processed-manifest-typeset.json", manifest)
    summary = {
        "checkpoint": "05-typeset-preflight",
        "chapter_id": manifest.get("chapter_id"),
        "status": status,
        "active_story_objects": len(active),
        "styled_story_objects": len(plan_objects),
        "tombstones_excluded": len(tombstones),
        "primary_dialogue_font": primary_font.stem,
        "system_font": system_font.stem,
        "skill_font": skill_font.stem,
        "font_size_auto_count": len(auto_font_sizes),
        "minimum_font_size": min_size,
        "maximum_font_size": max_size,
        "overflow_count": len(overflow),
        "readability_warning_count": len(readability_warnings),
        "missing_glyph_count": len(glyph_missing),
        "failures": failures,
        "next_action": typography_plan["next_action"],
    }
    dump_json(out / "typeset-preflight-summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
