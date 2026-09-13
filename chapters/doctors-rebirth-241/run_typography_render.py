from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw

from app.render.text_renderer import _fits, get_font_path, render_text_in_box

CID = "c241d001"
OUT = Path("checkpoint/05-render")
PROOF = OUT / "proof"
RENDERED = OUT / "rendered_slices"
FINAL = OUT / "final_pages"
WORK = Path(f"data/processed/{CID}")

SKILL = {
    "box_a53dc05e63fc4e11", "box_b6f5d5d07b01453a", "box_a17f4b50bce94125", "box_6a040b96ebaf4a22",
    "box_791027f1066d4b38", "box_82028db6e4d14c94", "box_69e33a25154f452e", "box_670cd228d5b649d2",
    "box_1bf8f8b8b4fd4bdf", "box_05d25fc270f246f7",
}
SHOUT = {
    "box_de950b2d1fee470f", "box_ca2b94c697684b79", "box_d496a35c143b462f", "box_8722e35a52e94af5",
    "box_57aacdb47d4f41c4", "box_c5f930806b2a48ce", "box_a5389b0635394c59",
}
PUNCT = {"box_ebbb5475e55d4a32", "box_a5b4216f8a584c31"}
ROLE_FONT = {"dialogue": "Mac-dinh-3", "skill": "Skill-fonts-1", "shout": "Granite", "punctuation": "Mac-dinh-3"}
ROLE_SIZES = {
    "dialogue": [64, 60, 56, 52, 48, 44, 40, 38, 36, 34, 32, 30, 28],
    "skill": [92, 84, 76, 68, 60, 56, 52, 48, 44, 40, 36, 32, 30, 28],
    "shout": [80, 72, 68, 64, 60, 56, 52, 48, 44, 40, 36, 32, 30, 28],
    "punctuation": [80, 72, 64, 56, 48, 40, 36, 32, 28],
}
_CMAP: dict[str, set[int]] = {}


def role_for(box_id: str) -> str:
    if box_id in SKILL:
        return "skill"
    if box_id in SHOUT:
        return "shout"
    if box_id in PUNCT:
        return "punctuation"
    return "dialogue"


def font_supports(name: str, text: str) -> bool:
    if name not in _CMAP:
        path = get_font_path(name)
        if not path.is_file():
            _CMAP[name] = set()
        else:
            tt = TTFont(str(path))
            cmap: set[int] = set()
            for table in tt["cmap"].tables:
                cmap.update(table.cmap.keys())
            tt.close()
            _CMAP[name] = cmap
    cmap = _CMAP[name]
    return bool(cmap) and all(ord(ch) in cmap for ch in text if ch not in "\n\r\t")


def resolve_font(preferred: str, text: str) -> tuple[str, str | None]:
    if font_supports(preferred, text):
        return preferred, None
    if font_supports("Mac-dinh-3", text):
        return "Mac-dinh-3", f"{preferred} missing Vietnamese glyph(s)"
    if font_supports("BeVietnamPro-SemiBold", text):
        return "BeVietnamPro-SemiBold", f"{preferred}/Mac-dinh-3 missing Vietnamese glyph(s)"
    raise SystemExit(f"No Vietnamese-capable font for {text!r}")


def move_inside_core(region: dict, core: dict, image_h: int) -> tuple[dict, bool]:
    r = {k: int(region[k]) for k in ("x1", "y1", "x2", "y2")}
    before = dict(r)
    height = r["y2"] - r["y1"]
    core_y1 = int(core.get("core_y1") or 0)
    core_y2 = int(core.get("core_y2") or image_h)
    if height > core_y2 - core_y1:
        return r, False
    shift = 0
    if r["y1"] < core_y1:
        shift = core_y1 - r["y1"]
    if r["y2"] + shift > core_y2:
        shift -= r["y2"] + shift - core_y2
    r["y1"] += shift
    r["y2"] += shift
    return r, r != before


def build_proofs(plan: list[dict], rendered_lookup: dict[tuple[int, int], Path], seams: list[dict]) -> list[str]:
    cells = []
    for row in plan:
        im = Image.open(rendered_lookup[(row["source_page"], row["slice_index"])]).convert("RGB")
        r = row["region"]
        margin = 45
        crop = im.crop((
            max(0, r["x1"] - margin), max(0, r["y1"] - margin),
            min(im.width, r["x2"] + margin), min(im.height, r["y2"] + margin),
        ))
        cells.append((row, crop))

    names = []
    cell_w, cell_h, per = 470, 380, 12
    for start in range(0, len(cells), per):
        batch = cells[start:start + per]
        sheet = Image.new("RGB", (3 * cell_w, math.ceil(len(batch) / 3) * cell_h), "white")
        draw = ImageDraw.Draw(sheet)
        for n, (row, crop) in enumerate(batch):
            crop.thumbnail((cell_w - 20, 270))
            x = (n % 3) * cell_w + 10
            y = (n // 3) * cell_h + 10
            draw.text((x, y), f"{row['box_id']} p{row['source_page']} s{row['slice_index']}", fill="black")
            draw.text((x, y + 20), f"{row['role']} | {row['font']} {row['font_size']}px | lines={row['line_count']}", fill="black")
            if row.get("font_fallback_reason"):
                draw.text((x, y + 40), "fallback: " + row["font_fallback_reason"], fill="black")
            sheet.paste(crop, (x, y + 78))
        name = f"object-proof-{start // per:02d}.jpg"
        sheet.save(PROOF / name, quality=92)
        names.append(f"proof/{name}")

    overview = Image.new("RGB", (4 * 340, math.ceil(17 / 4) * 1240), "white")
    draw = ImageDraw.Draw(overview)
    for n in range(17):
        im = Image.open(FINAL / f"{n:03d}.png").convert("RGB")
        im.thumbnail((320, 1200))
        x = (n % 4) * 340 + 10
        y = (n // 4) * 1240 + 10
        draw.text((x, y), f"source {n:03d}", fill="black")
        overview.paste(im, (x, y + 24))
    overview.save(PROOF / "full-page-overview.jpg", quality=90)

    if seams:
        sw, sh = 500, 320
        sheet = Image.new("RGB", (2 * sw, math.ceil(len(seams) / 2) * sh), "white")
        draw = ImageDraw.Draw(sheet)
        for n, seam in enumerate(seams):
            im = Image.open(FINAL / f"{seam['source_page']:03d}.png").convert("RGB")
            seam_y = seam["y"]
            crop = im.crop((0, max(0, seam_y - 120), im.width, min(im.height, seam_y + 120)))
            crop.thumbnail((sw - 20, 250))
            x = (n % 2) * sw + 10
            y = (n // 2) * sh + 10
            draw.text((x, y), f"p{seam['source_page']:03d} seam y={seam_y}", fill="black")
            sheet.paste(crop, (x, y + 30))
        sheet.save(PROOF / "stitch-seams.jpg", quality=92)
    return names


def main() -> None:
    for directory in (OUT, PROOF, RENDERED, FINAL):
        directory.mkdir(parents=True, exist_ok=True)

    translation_summary = json.loads(Path("checkpoint/04-translation/translation-review-summary.json").read_text(encoding="utf-8"))
    if translation_summary.get("status") != "PASS" or not translation_summary.get("editorial_pass") or translation_summary.get("active_story_objects") != 76:
        raise SystemExit(f"translation gate invalid: {translation_summary}")
    manifest = json.loads(Path("checkpoint/04-translation/translated-manifest.json").read_text(encoding="utf-8"))
    pages = manifest.get("pages") or []
    if len(pages) != 68:
        raise SystemExit(f"expected 68 slices, got {len(pages)}")

    plan: list[dict] = []
    failures: list[dict] = []
    shifted: list[dict] = []
    active_count = 0

    for page_index, page in enumerate(pages):
        source_page = int(page.get("source_page") or 0)
        slice_index = int(page.get("slice_index") or 0)
        clean = WORK / f"clean_{source_page:03d}_{slice_index:02d}.png"
        if not clean.is_file():
            raise SystemExit(f"missing approved CLEAN slice {clean}")
        with Image.open(clean) as base:
            image_h = base.height
            draw_image = base.convert("RGB")
        draw = ImageDraw.Draw(draw_image)
        core = page.get("stitch_core") or {}
        for obj in page.get("text_objects") or []:
            if not isinstance(obj, dict) or not obj.get("translation_reviewed"):
                continue
            refs = [v for v in (obj.get("source_boxes") or []) if isinstance(v, str)]
            if len(refs) != 1:
                raise SystemExit(f"bad source refs for {obj.get('id')}")
            box_id = refs[0]
            text = str(obj.get("translation") or "").strip()
            if not text:
                raise SystemExit(f"empty translation {box_id}")
            active_count += 1
            region, moved = move_inside_core(obj["region"], core, image_h)
            if moved:
                shifted.append({"box_id": box_id, "before": obj["region"], "after": region})
            raw_w = region["x2"] - region["x1"]
            raw_h = region["y2"] - region["y1"]
            padding = max(3, min(8, int(min(raw_w, raw_h) * 0.05)))
            box_w = raw_w - 2 * padding
            box_h = raw_h - 2 * padding
            role = role_for(box_id)
            font, fallback_reason = resolve_font(ROLE_FONT[role], text)
            font_path = str(get_font_path(font))
            chosen = None
            lines = []
            for size in ROLE_SIZES[role]:
                ok, wrapped = _fits(draw, text, box_w, box_h, font_path, size, 2)
                if ok:
                    chosen = size
                    lines = wrapped
                    break
            if chosen is None:
                failures.append({
                    "box_id": box_id, "page_index": page_index, "source_page": source_page,
                    "slice_index": slice_index, "region": region, "text": text,
                    "reason": "does_not_fit_at_28px",
                })
                continue
            obj["region"] = region
            obj["style"] = {
                "color": "auto", "font": font, "fontSize": chosen, "bold": role in ("skill", "shout"),
                "strokeWidth": 2, "strokeColor": "auto", "bgColor": "transparent", "cornerRadius": "0",
                "horizontalAlign": "center", "verticalAlign": "middle",
            }
            plan.append({
                "object_id": obj.get("id"), "box_id": box_id, "page_index": page_index,
                "source_page": source_page, "slice_index": slice_index, "role": role,
                "font": font, "font_size": chosen, "font_fallback_reason": fallback_reason,
                "line_count": len(lines), "region": region, "translation": text,
            })

    if active_count != 76:
        raise SystemExit(f"expected 76 translated objects, got {active_count}")
    (OUT / "typography-failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    if failures:
        raise SystemExit(f"{len(failures)} objects fail fixed 28px typography preflight")

    plan_by_object = {row["object_id"]: row for row in plan}
    rendered_lookup: dict[tuple[int, int], Path] = {}
    for page in pages:
        source_page = int(page.get("source_page") or 0)
        slice_index = int(page.get("slice_index") or 0)
        clean = WORK / f"clean_{source_page:03d}_{slice_index:02d}.png"
        image = Image.open(clean).convert("RGB")
        for obj in page.get("text_objects") or []:
            row = plan_by_object.get(obj.get("id"))
            if row is None:
                continue
            style = obj["style"]
            region = obj["region"]
            padding = max(3, min(8, int(min(region["x2"] - region["x1"], region["y2"] - region["y1"]) * 0.05)))
            image = render_text_in_box(
                image, obj["translation"],
                (region["x1"], region["y1"], region["x2"], region["y2"]),
                font_name=style["font"], font_size=style["fontSize"], padding=padding,
                fill="auto", is_bold=style["bold"], stroke_width=2, stroke_color="auto",
                bg_color="transparent", horizontal_align="center", vertical_align="middle",
            )
        path = RENDERED / f"{source_page:03d}_{slice_index:02d}.png"
        image.save(path)
        rendered_lookup[(source_page, slice_index)] = path

    grouped: dict[int, list[dict]] = defaultdict(list)
    for page in pages:
        grouped[int(page.get("source_page") or 0)].append(page)
    seams: list[dict] = []
    for source_page in range(17):
        owned = sorted(grouped[source_page], key=lambda p: int(p.get("slice_index") or 0))
        chunks = []
        expected = 0
        for n, page in enumerate(owned):
            slice_index = int(page.get("slice_index") or 0)
            core = page.get("stitch_core") or {}
            y1 = int(core.get("core_source_y1") or 0)
            y2 = int(core.get("core_source_y2") or 0)
            if y1 != expected:
                raise SystemExit(f"stitch ownership gap source {source_page}: {expected}->{y1}")
            arr = np.array(Image.open(rendered_lookup[(source_page, slice_index)]).convert("RGB"))
            core_y1 = int(core.get("core_y1") or 0)
            core_y2 = int(core.get("core_y2") or arr.shape[0])
            chunks.append(arr[core_y1:core_y2])
            expected = y2
            if n < len(owned) - 1:
                seams.append({"source_page": source_page, "y": y2})
        if not chunks:
            raise SystemExit(f"no slices for source page {source_page}")
        Image.fromarray(np.concatenate(chunks, axis=0)).save(FINAL / f"{source_page:03d}.png")

    proof_names = build_proofs(plan, rendered_lookup, seams)
    sizes = [row["font_size"] for row in plan]
    summary = {
        "checkpoint": "05-typography-render", "chapter_id": CID, "status": "REVIEW_REQUIRED",
        "active_story_objects": 76, "rendered_story_objects": 76, "rendered_slices": 68,
        "source_pages": 17, "autosize_active_objects": 0, "hard_min_font_size": 28,
        "font_size_min_px": min(sizes), "font_size_max_px": max(sizes),
        "font_fallback_count": sum(1 for row in plan if row.get("font_fallback_reason")),
        "font_counts": dict(Counter(row["font"] for row in plan)),
        "seam_shift_repairs": len(shifted), "proof_sheets": proof_names,
        "stitch_seams": len(seams), "next_action": "HUMAN_VISUAL_REVIEW",
    }
    (OUT / "translated-manifest-rendered.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "typography-plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "seam-shift-repairs.json").write_text(json.dumps(shifted, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "render-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "HUMAN_CHECKPOINT.txt").write_text(
        "CHECKPOINT: RENDER CANDIDATE\nSTOP for human visual review of all 17 pages, object proofs and stitch seams before final export.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
