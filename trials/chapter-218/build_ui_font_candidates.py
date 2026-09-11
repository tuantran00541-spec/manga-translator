#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

WARNING_IDS = {
    "text_b478707b63c54344",
    "text_64b56e211cab46fb",
    "text_5562636cb2f34195",
    "text_ca9d8fbd11e94caa",
}
PAGE_INDEX = 64
CANDIDATE_LIMIT = 7


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def font_cmap(path: Path) -> set[int]:
    font = TTFont(path, lazy=True)
    try:
        return set((font.getBestCmap() or {}).keys())
    finally:
        font.close()


def text_chars(texts: list[str]) -> set[int]:
    return {ord(ch) for text in texts for ch in text if not ch.isspace()}


def line_width(font: ImageFont.FreeTypeFont, text: str) -> int:
    box = font.getbbox(text, stroke_width=2)
    return max(0, box[2] - box[0])


def line_height(font: ImageFont.FreeTypeFont) -> int:
    box = font.getbbox("Ágjỵ", stroke_width=2)
    return max(1, box[3] - box[1])


def wrap_paragraph(draw: ImageDraw.ImageDraw, font: ImageFont.FreeTypeFont, text: str, max_width: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    cur = words[0]
    for word in words[1:]:
        trial = f"{cur} {word}"
        if line_width(font, trial) <= max_width:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    lines.append(cur)
    return lines


def wrap_text(draw: ImageDraw.ImageDraw, font: ImageFont.FreeTypeFont, text: str, max_width: int) -> list[str]:
    lines: list[str] = []
    for para in text.split("\n"):
        lines.extend(wrap_paragraph(draw, font, para, max_width))
    return lines


def fit_text(font_path: Path, text: str, width: int, height: int, floor: int = 14, ceiling: int = 72):
    dummy = Image.new("RGB", (8, 8), "black")
    draw = ImageDraw.Draw(dummy)
    max_w = max(8, width - 16)
    max_h = max(8, height - 14)
    for size in range(ceiling, floor - 1, -1):
        font = ImageFont.truetype(str(font_path), size=size)
        lines = wrap_text(draw, font, text, max_w)
        lh = line_height(font)
        spacing = max(1, round(size * 0.10))
        total_h = len(lines) * lh + max(0, len(lines) - 1) * spacing
        widest = max((line_width(font, line) for line in lines), default=0)
        if widest <= max_w and total_h <= max_h:
            return size, lines, widest, total_h
    font = ImageFont.truetype(str(font_path), size=floor)
    lines = wrap_text(draw, font, text, max_w)
    lh = line_height(font)
    spacing = max(1, round(floor * 0.10))
    return floor, lines, max((line_width(font, l) for l in lines), default=0), len(lines) * lh + max(0, len(lines)-1) * spacing


def compactness_score(font_path: Path, texts: list[str]) -> float:
    font = ImageFont.truetype(str(font_path), size=32)
    widths = []
    for text in texts:
        for line in text.split("\n"):
            widths.append(line_width(font, line))
    return sum(widths) / max(1, len(widths))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--typeset-root", required=True)
    ap.add_argument("--clean-root", required=True)
    ap.add_argument("--font-dir", default="app/static/fonts")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    typeset_root = Path(args.typeset_root)
    clean_root = Path(args.clean_root)
    font_dir = Path(args.font_dir)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    manifest = read_json(typeset_root / "processed-manifest-typeset.json")
    plan = read_json(typeset_root / "typography-plan.json")
    pages = manifest.get("pages") or []
    page = pages[PAGE_INDEX]

    # Semantic classification is durable in typography-plan.json; the render
    # manifest intentionally stores only render-facing fields. Resolve status
    # object IDs from the plan, then map them back to the page text objects.
    plan_objects = plan.get("objects") or {}
    status_ids = {
        oid for oid, meta in plan_objects.items()
        if isinstance(meta, dict)
        and int(meta.get("page_index", -1)) == PAGE_INDEX
        and str(meta.get("semantic") or "") == "status_ui"
    }
    status = [
        obj for obj in (page.get("text_objects") or [])
        if isinstance(obj, dict)
        and not obj.get("source_missing")
        and str(obj.get("id") or "") in status_ids
    ]
    if len(status) < 8:
        raise SystemExit(f"unexpected status_ui count on p{PAGE_INDEX}: {len(status)} ids={sorted(status_ids)}")
    ids = {str(obj.get("id") or "") for obj in status}
    if not WARNING_IDS.issubset(ids):
        raise SystemExit(f"warning ids missing from status page: {sorted(WARNING_IDS - ids)}")

    clean_name = Path(str(page.get("clean") or "")).name
    clean_path = clean_root / "processed" / clean_name
    if not clean_path.is_file():
        raise SystemExit(f"missing clean slice: {clean_path}")

    texts = [str(obj.get("translation") or "").strip() for obj in status]
    required = text_chars(texts)
    candidates = []
    for fp in sorted(font_dir.glob("*.ttf")):
        cmap = font_cmap(fp)
        missing = sorted(required - cmap)
        if missing:
            continue
        score = compactness_score(fp, texts)
        candidates.append((score, fp))
    if not candidates:
        raise SystemExit("no bundled font covers status UI glyphs")

    preferred_order = [
        "Mac-dinh-3.ttf", "Mac-dinh-2.ttf", "Manga-fonts.ttf",
        "Curves-Regular.ttf", "Granite.ttf", "Skill-fonts-3.ttf", "Skill-fonts-4.ttf",
    ]
    by_name = {fp.name: (score, fp) for score, fp in candidates}
    selected = []
    for name in preferred_order:
        if name in by_name:
            selected.append(by_name[name])
    for item in sorted(candidates, key=lambda x: x[0]):
        if item not in selected:
            selected.append(item)
        if len(selected) >= CANDIDATE_LIMIT:
            break
    selected = selected[:CANDIDATE_LIMIT]

    with Image.open(clean_path) as src:
        clean = src.convert("RGB")
    # Skill grid only: enough vertical context to expose inter-column collisions.
    crop_box = (40, 2820, min(clean.width, 800), min(clean.height, 3610))
    base = clean.crop(crop_box)
    cards = []
    report = []

    for score, fp in selected:
        card = base.copy()
        draw = ImageDraw.Draw(card)
        rows = []
        for obj in status:
            region = obj.get("region") or {}
            x1, y1, x2, y2 = [int(region[k]) for k in ("x1", "y1", "x2", "y2")]
            # Only render objects intersecting the skill-grid crop.
            if y2 < crop_box[1] or y1 > crop_box[3]:
                continue
            text = str(obj.get("translation") or "").strip()
            size, lines, widest, total_h = fit_text(fp, text, x2-x1, y2-y1)
            font = ImageFont.truetype(str(fp), size=size)
            line_h = line_height(font)
            spacing = max(1, round(size * 0.10))
            rx = x1 - crop_box[0] + 8
            ry = y1 - crop_box[1] + 7
            for li, line in enumerate(lines):
                draw.text((rx, ry), line, font=font, fill="white", stroke_width=2, stroke_fill="black")
                ry += line_h + (spacing if li < len(lines)-1 else 0)
            rows.append({
                "id": obj.get("id"), "text": text, "font_size": size, "lines": lines,
                "region": region, "measured_width": widest, "measured_height": total_h,
                "warning_target": obj.get("id") in WARNING_IDS,
            })

        # title strip
        titled = Image.new("RGB", (card.width, card.height + 38), "white")
        td = ImageDraw.Draw(titled)
        td.text((8, 10), f"{fp.stem} | compactness={score:.1f}", fill="black")
        titled.paste(card, (0, 38))
        cards.append(titled)
        report.append({"font": fp.stem, "compactness": round(score, 2), "objects": rows})

    sheet_w = max(c.width for c in cards)
    sheet_h = sum(c.height for c in cards)
    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    y = 0
    for card in cards:
        sheet.paste(card, (0, y))
        y += card.height
    sheet.save(out / "status-ui-font-candidates.jpg", quality=94)
    (out / "candidate-report.json").write_text(json.dumps({
        "page_index": PAGE_INDEX,
        "warning_ids": sorted(WARNING_IDS),
        "candidate_fonts": report,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"candidates": [x[1].stem for x in selected], "status_objects": len(status)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
