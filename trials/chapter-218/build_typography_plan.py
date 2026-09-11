#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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

FONT_BUCKETS = [48, 44, 40, 38, 36, 34, 32, 30, 28]
HARD_MIN_FONT_SIZE = 28

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


def choose_global_font(font_dir: Path, candidates: list[str], texts: list[str], *, fallback: Path | None = None):
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
        checked.append({"font": fallback.name, "exists": True, "missing": len(missing), "fallback": True})
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


def preferred_size(kind: str) -> int:
    if kind == "status_ui":
        return 36
    if kind == "system_ui":
        return 40
    if kind == "skill_attack":
        return 48
    if kind == "emphasis":
        return 48
    return 40


def ladder_for(preferred: int) -> list[int]:
    return [s for s in FONT_BUCKETS if s <= preferred and s >= HARD_MIN_FONT_SIZE]


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

        missing = sorted({ord(ch) for ch in text if not ch.isspace()} - font_cmaps[str(font_path)])
        if missing:
            glyph_missing.append({"id": oid, "font": font_path.name, "codepoints": missing})
            failures.append(f"{oid}: assigned font lacks glyphs")
            continue

        pad = max(2, min(RENDER_DEFAULT_PADDING, int(min(raw_w, raw_h) * RENDER_PADDING_RATIO_MAX)))
        box_w = raw_w - pad * 2
        box_h = raw_h - pad * 2
        stroke_width = 2
        picked_size = None
        picked_lines = None
        for size in ladder_for(preferred_size(kind)):
            ok, lines = _fits(draw, text, box_w, box_h, str(font_path), size, stroke_width)
            if ok:
                picked_size = size
                picked_lines = lines
                break
        if picked_size is None:
            overflow.append({
                "id": oid,
                "page_index": page_index,
                "source_page": page.get("source_page"),
                "slice_index": page.get("slice_index"),
                "region": region,
                "semantic": kind,
                "text": text,
                "minimum_size": HARD_MIN_FONT_SIZE,
            })
            continue

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
            "fit": {
                "box_width": box_w,
                "box_height": box_h,
                "padding": pad,
                "hard_min_font_size": HARD_MIN_FONT_SIZE,
                "passed": True,
            },
        }
        size_counter[picked_size] += 1
        font_counter[font_path.stem] += 1

    if overflow:
        failures.append(f"{len(overflow)} object(s) do not fit at >= {HARD_MIN_FONT_SIZE}px")
    if glyph_missing:
        failures.append(f"{len(glyph_missing)} object(s) have missing glyphs")
    if len(plan_objects) != len(active):
        failures.append(f"styled {len(plan_objects)}/{len(active)} active objects")

    auto_font_sizes = [
        oid for oid, item in plan_objects.items()
        if str((item.get("style") or {}).get("fontSize", "")).lower() == "auto"
    ]
    min_size = min((int(item["style"]["fontSize"]) for item in plan_objects.values()), default=None)
    if auto_font_sizes:
        failures.append("fontSize:auto remained in active typography plan")
    if min_size is not None and min_size < HARD_MIN_FONT_SIZE:
        failures.append(f"minimum planned font size is {min_size}px")

    status = "PASS" if not failures else "FAIL"
    manifest["typography_review"] = {
        "status": status,
        "source_checkpoint": "04-translation-review",
        "active_story_objects": len(active),
        "styled_story_objects": len(plan_objects),
        "font_size_auto_count": len(auto_font_sizes),
        "minimum_font_size": min_size,
        "overflow_count": len(overflow),
        "missing_glyph_count": len(glyph_missing),
    }

    typography_plan = {
        "chapter_id": manifest.get("chapter_id"),
        "checkpoint": "05-typeset-preflight",
        "status": status,
        "policy": {
            "font_size_buckets": FONT_BUCKETS,
            "hard_min_font_size": HARD_MIN_FONT_SIZE,
            "font_size_auto_allowed": False,
            "primary_dialogue_font_stable": True,
            "semantic_font_variants": ["system_ui", "status_ui", "skill_attack"],
            "overflow_policy": "FAIL; shorten translation / improve line breaks / safely expand region before reducing below 28px",
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
        },
        "size_histogram": {str(k): v for k, v in sorted(size_counter.items(), reverse=True)},
        "font_usage": dict(sorted(font_counter.items())),
        "semantic_usage": dict(sorted(kind_counter.items())),
        "overflow_objects": overflow,
        "missing_glyph_objects": glyph_missing,
        "objects": plan_objects,
        "failures": failures,
        "next_action": "RENDER_FROM_APPROVED_CLEAN" if status == "PASS" else "LOCAL_TYPESET_REPAIR",
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
        "overflow_count": len(overflow),
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
