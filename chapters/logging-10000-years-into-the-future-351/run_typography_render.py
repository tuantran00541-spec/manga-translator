from __future__ import annotations

import json
import math
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw

from app.parameters import RENDER_DEFAULT_PADDING, RENDER_PADDING_RATIO_MAX
from app.render.text_renderer import _fits, get_font_path, render_text_in_box

CHAPTER_ID = "c3513510"
OUT = Path("checkpoint/05-render")
PROOF = OUT / "proof"
RENDERED_SLICES = OUT / "rendered_slices"
FINAL_PAGES = OUT / "final_pages"
WORK = Path(f"data/processed/{CHAPTER_ID}")

# Typography is selected from the actual source composition + scene emotion.
# Normal speech/thought stays on Mac-dinh-3. Other fonts are semantic/emotional,
# never random per-line decoration.
DIALOGUE_NORMAL = {
    "box_e3b6ae51e9be4a2a",
    "box_4ff3e63fa43243a4", "box_07741edf43df4464", "box_4cb8ab0f6b7f4daa",
    "box_f4cbd538a9694c6f", "box_174bb4e22b3e4fee", "box_bb6dfb8c7c1e4e04",
    "box_c2aab41eba7d4dc0",
    "box_8e3985314c5946ff", "box_3247316c82c242c0", "box_31fb34d4442143f0",
    "box_a68523b0cffc4df5", "box_c64fae1997f341cc", "box_a6b14daa348e4d91", "box_f45d05491b4f43a0",
    "box_df60dc83681e4e11", "box_387e0f58850d4f1d", "box_8b2d6949b3394b5c", "box_ffc320d1ace04ad4",
    "box_6f4221a285054c0c", "box_f2c5ed0af9474b84",
    "box_27081357e3f34694", "box_2be39624a02c4658",
    "box_a8aa37118dff4c93", "box_e1fefd1f82084bca", "box_f16eacb3f56c49f6", "box_0dae2a64f2564a43",
}

# Panic / explosive shouting: the character is losing control, so use a hard impact face.
PANIC_SHOUT = {"box_e3b6ae51e9be4a2a", "box_2a0372b752a64600"}

# Darius' threat sequence is one emotional run: humiliation -> hatred -> vow to return.
# Keep one dark family across the whole run instead of changing font each bubble.
DARK_THREAT = {
    "box_df60dc83681e4e11", "box_387e0f58850d4f1d", "box_8b2d6949b3394b5c",
    "box_ffc320d1ace04ad4", "box_6f4221a285054c0c", "box_f2c5ed0af9474b84",
}

# Myth recollection is deliberately more storybook/flowing than ordinary dialogue.
MYTHIC = {
    "box_ef3b5f740102470a", "box_87ee6c1f744c411b",
    "box_b7a9cef01ad041b6", "box_e311034581154a2a",
}

# Poetic description of the arrow's beauty: flowing, not shouty.
LYRICAL_NARRATION = {
    "box_ad4ef7f93ab24ab3", "box_ba7f792c09fd4be7",
    "box_a99699fc3278422f", "box_f1b9b9850c574675",
}

# Dramatic free-text/narration punches that should carry more visual weight.
DRAMATIC_NARRATION = {
    "box_af2ae200e4af4d23", "box_66c683b9570b4165",
    "box_2df38511cc5b44bd", "box_bf1de16bd3bc4d57",
    "box_1d859ecaa9524d00", "box_dbca00c8c37644b0",
    "box_91d53e37e864422b",
}

SKILL_NAME = {"box_c58b5b3c29c64385"}
COSMIC_REVELATION = {"box_0dae2a64f2564a43"}

ROLE_FONT = {
    "dialogue": "Mac-dinh-3",
    "narration": "Mac-dinh-2",
    "panic_shout": "Granite",
    "dark_threat": "Shadow-fonts",
    "mythic": "Curves-Regular",
    "lyrical_narration": "Curves-Regular",
    "dramatic_narration": "Manga-fonts",
    "skill_name": "Skill-fonts-1",
    "cosmic_revelation": "Shadow-fonts",
}

ROLE_LADDERS = {
    "dialogue": [72, 68, 64, 60, 56, 52, 48, 44, 40, 38, 36, 34, 32, 30, 28],
    "narration": [64, 60, 56, 52, 48, 44, 40, 38, 36, 34, 32, 30, 28],
    "panic_shout": [88, 80, 72, 68, 64, 60, 56, 52, 48, 44, 40, 36, 32, 30, 28],
    "dark_threat": [76, 72, 68, 64, 60, 56, 52, 48, 44, 40, 36, 34, 32, 30, 28],
    "mythic": [68, 64, 60, 56, 52, 48, 44, 40, 38, 36, 34, 32, 30, 28],
    "lyrical_narration": [68, 64, 60, 56, 52, 48, 44, 40, 38, 36, 34, 32, 30, 28],
    "dramatic_narration": [76, 72, 68, 64, 60, 56, 52, 48, 44, 40, 36, 34, 32, 30, 28],
    "skill_name": [104, 96, 88, 80, 72, 64, 60, 56, 52, 48, 44, 40, 36, 32, 30, 28],
    "cosmic_revelation": [88, 80, 72, 68, 64, 60, 56, 52, 48, 44, 40, 36, 32, 30, 28],
}


def role_for(box_id: str) -> str:
    if box_id in SKILL_NAME:
        return "skill_name"
    if box_id in COSMIC_REVELATION:
        return "cosmic_revelation"
    if box_id in PANIC_SHOUT:
        return "panic_shout"
    if box_id in DARK_THREAT:
        return "dark_threat"
    if box_id in MYTHIC:
        return "mythic"
    if box_id in LYRICAL_NARRATION:
        return "lyrical_narration"
    if box_id in DRAMATIC_NARRATION:
        return "dramatic_narration"
    if box_id in DIALOGUE_NORMAL:
        return "dialogue"
    return "narration"


def font_cmap(font_name: str) -> set[int]:
    path = get_font_path(font_name)
    if not path.is_file():
        return set()
    tt = TTFont(str(path))
    cmap: set[int] = set()
    for table in tt["cmap"].tables:
        cmap.update(table.cmap.keys())
    tt.close()
    return cmap


def font_supports(font_name: str, text: str) -> bool:
    cmap = font_cmap(font_name)
    if not cmap:
        return False
    return all(ord(ch) in cmap for ch in text if ch not in "\n\r\t")


def resolve_font(preferred: str, text: str) -> tuple[str, str | None]:
    if font_supports(preferred, text):
        return preferred, None
    # User-designated ordinary dialogue font is the first semantic fallback.
    if preferred != "Mac-dinh-3" and font_supports("Mac-dinh-3", text):
        return "Mac-dinh-3", f"{preferred} missing Vietnamese glyph(s)"
    # Last-resort glyph safety only; never used as the chapter's visual default.
    if font_supports("BeVietnamPro-SemiBold", text):
        return "BeVietnamPro-SemiBold", f"{preferred} and Mac-dinh-3 missing Vietnamese glyph(s)"
    raise SystemExit(f"No Vietnamese-capable font for text: {text!r}")


def restore_clean_stack() -> None:
    base = Path("checkpoint/02-clean/processed")
    for src in base.iterdir():
        if src.is_file():
            shutil.copy2(src, WORK / src.name)
    for overlay_root in (
        Path("checkpoint/02b-repair/overlay/processed"),
        Path("checkpoint/02c-logo/overlay/processed"),
    ):
        for src in overlay_root.iterdir():
            if src.is_file():
                shutil.copy2(src, WORK / src.name)


def build_proofs(manifest: dict, rendered_lookup: dict[tuple[int, int], Path], plan: list[dict], stitch_seams: list[dict]) -> list[str]:
    plan_by_id = {row["object_id"]: row for row in plan}
    cells = []
    for page in manifest.get("pages") or []:
        source_page = int(page.get("source_page") or 0)
        slice_index = int(page.get("slice_index") or 0)
        im = Image.open(rendered_lookup[(source_page, slice_index)]).convert("RGB")
        for obj in page.get("text_objects") or []:
            if not isinstance(obj, dict) or obj.get("id") not in plan_by_id:
                continue
            row = plan_by_id[obj["id"]]
            r = obj["region"]
            margin = 45
            x1 = max(0, int(r["x1"]) - margin)
            y1 = max(0, int(r["y1"]) - margin)
            x2 = min(im.width, int(r["x2"]) + margin)
            y2 = min(im.height, int(r["y2"]) + margin)
            cells.append((row, im.crop((x1, y1, x2, y2))))

    proof_sheets = []
    cell_w, cell_h, per_sheet = 470, 380, 12
    for start in range(0, len(cells), per_sheet):
        batch = cells[start:start + per_sheet]
        cols = 3
        rows_n = math.ceil(len(batch) / cols)
        sheet = Image.new("RGB", (cols * cell_w, rows_n * cell_h), "white")
        d = ImageDraw.Draw(sheet)
        for n, (row, crop) in enumerate(batch):
            crop.thumbnail((cell_w - 20, 270))
            x = (n % cols) * cell_w + 10
            y = (n // cols) * cell_h + 10
            sheet.paste(crop, (x, y + 78))
            d.text((x, y), f"{row['box_id']} p{row['source_page']} s{row['slice_index']}", fill="black")
            d.text((x, y + 20), f"{row['role']} | {row['font']} {row['font_size']}px | lines={row['line_count']}", fill="black")
            if row.get("font_fallback_reason"):
                d.text((x, y + 40), "fallback: " + row["font_fallback_reason"], fill="black")
        name = f"object-proof-{start // per_sheet:02d}.jpg"
        sheet.save(PROOF / name, quality=92)
        proof_sheets.append(f"proof/{name}")

    thumbs = []
    for source_page in range(11):
        im = Image.open(FINAL_PAGES / f"{source_page:03d}.png").convert("RGB")
        im.thumbnail((320, 1100))
        thumbs.append((source_page, im.copy()))
    cols = 3
    cell_w2, cell_h2 = 340, 1140
    rows2 = math.ceil(len(thumbs) / cols)
    overview = Image.new("RGB", (cols * cell_w2, rows2 * cell_h2), "white")
    od = ImageDraw.Draw(overview)
    for n, (source_page, im) in enumerate(thumbs):
        x = (n % cols) * cell_w2 + 10
        y = (n // cols) * cell_h2 + 10
        od.text((x, y), f"source page {source_page:03d}", fill="black")
        overview.paste(im, (x, y + 24))
    overview.save(PROOF / "full-page-overview.jpg", quality=90)

    seam_cells = []
    for seam in stitch_seams:
        p, y = seam["source_page"], seam["y"]
        im = Image.open(FINAL_PAGES / f"{p:03d}.png").convert("RGB")
        seam_cells.append((p, y, im.crop((0, max(0, y - 120), im.width, min(im.height, y + 120)))))
    if seam_cells:
        sw, sh, cols = 500, 320, 2
        rowsn = math.ceil(len(seam_cells) / cols)
        seam_sheet = Image.new("RGB", (cols * sw, rowsn * sh), "white")
        sd = ImageDraw.Draw(seam_sheet)
        for n, (p, y, crop) in enumerate(seam_cells):
            crop.thumbnail((sw - 20, 250))
            x = (n % cols) * sw + 10
            yy = (n // cols) * sh + 10
            sd.text((x, yy), f"page {p:03d} seam y={y}", fill="black")
            seam_sheet.paste(crop, (x, yy + 30))
        seam_sheet.save(PROOF / "stitch-seams.jpg", quality=92)
    return proof_sheets


def main() -> None:
    for d in (OUT, PROOF, RENDERED_SLICES, FINAL_PAGES, WORK):
        d.mkdir(parents=True, exist_ok=True)

    tsum = json.loads(Path("checkpoint/04-translation/translation-review-summary.json").read_text(encoding="utf-8"))
    if tsum.get("status") != "PASS" or not tsum.get("editorial_pass") or tsum.get("active_untranslated_story_objects") != 0:
        raise SystemExit(f"translation gate is not approved: {tsum}")
    manifest = json.loads(Path("checkpoint/04-translation/translated-manifest.json").read_text(encoding="utf-8"))
    if len(manifest.get("pages") or []) != 40:
        raise SystemExit(f"expected 40 slices, got {len(manifest.get('pages') or [])}")

    restore_clean_stack()

    plan: list[dict] = []
    failures: list[dict] = []
    seam_clamps: list[dict] = []
    active_count = 0

    for page_index, page in enumerate(manifest.get("pages") or []):
        source_page = int(page.get("source_page") or 0)
        slice_index = int(page.get("slice_index") or 0)
        clean_path = WORK / f"clean_{source_page:03d}_{slice_index:02d}.png"
        if not clean_path.is_file():
            raise SystemExit(f"approved CLEAN slice missing: {clean_path}")
        page["clean"] = clean_path.as_posix()
        with Image.open(clean_path) as im:
            img_w, img_h = im.size
        core = page.get("stitch_core") or {}
        core_y1 = int(core.get("core_y1") or 0)
        core_y2 = int(core.get("core_y2") or img_h)

        for obj in page.get("text_objects") or []:
            if not isinstance(obj, dict) or obj.get("source_missing"):
                continue
            text = str(obj.get("translation") or "").strip()
            if not text:
                continue
            active_count += 1
            refs = [str(v) for v in (obj.get("source_boxes") or []) if isinstance(v, str)]
            if len(refs) != 1:
                raise SystemExit(f"active text object does not have exactly one source box: {obj.get('id')}")
            box_id = refs[0]
            role = role_for(box_id)
            preferred_font = ROLE_FONT[role]
            font_name, fallback_reason = resolve_font(preferred_font, text)
            font_path = get_font_path(font_name)

            region = dict(obj.get("region") or {})
            try:
                x1, y1, x2, y2 = [int(region[k]) for k in ("x1", "y1", "x2", "y2")]
            except Exception as exc:
                raise SystemExit(f"malformed active text region: {obj.get('id')}") from exc
            x1, x2 = max(0, x1), min(img_w, x2)
            y1, y2 = max(0, y1), min(img_h, y2)
            original_region = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

            top_cross = max(0, core_y1 - y1)
            bottom_cross = max(0, y2 - core_y2)
            if top_cross or bottom_cross:
                if max(top_cross, bottom_cross) > 32:
                    failures.append({"box_id": box_id, "reason": "unsafe_large_stitch_core_crossing", "top": top_cross, "bottom": bottom_cross})
                    continue
                y1 = max(y1, core_y1)
                y2 = min(y2, core_y2)
                seam_clamps.append({"box_id": box_id, "top_px": top_cross, "bottom_px": bottom_cross})
            if x2 <= x1 or y2 <= y1:
                failures.append({"box_id": box_id, "reason": "empty_render_region_after_clamp"})
                continue

            raw_w, raw_h = x2 - x1, y2 - y1
            pad = max(2, min(RENDER_DEFAULT_PADDING, int(min(raw_w, raw_h) * RENDER_PADDING_RATIO_MAX)))
            box_w, box_h = raw_w - pad * 2, raw_h - pad * 2
            ladder = ROLE_LADDERS[role]
            selected = None
            selected_lines = None
            probe = ImageDraw.Draw(Image.new("RGB", (max(2, raw_w), max(2, raw_h)), "white"))
            for size in ladder:
                ok, lines = _fits(probe, text, box_w, box_h, str(font_path), size, 1)
                if ok:
                    selected, selected_lines = size, lines
                    break
            if selected is None:
                failures.append({"box_id": box_id, "role": role, "font": font_name, "reason": "does_not_fit_at_28px", "region": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}, "text": text})
                continue

            obj["source_region_before_typography"] = original_region
            obj["region"] = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
            style = dict(obj.get("style") or {})
            style.update({
                "font": font_name,
                "fontSize": selected,
                "bold": role in {"panic_shout", "dramatic_narration", "skill_name", "cosmic_revelation"},
                "color": "auto",
                "strokeWidth": 1,
                "strokeColor": "auto",
                "bgColor": "transparent",
                "horizontalAlign": "center",
                "verticalAlign": "middle",
            })
            obj["style"] = style
            obj["typography_reviewed"] = True
            obj["font_size_fixed"] = True
            obj["typography_role"] = role
            plan.append({
                "object_id": obj.get("id"), "box_id": box_id, "page_index": page_index,
                "source_page": source_page, "slice_index": slice_index, "region": obj["region"],
                "role": role, "preferred_font": preferred_font, "font": font_name,
                "font_fallback_reason": fallback_reason, "font_size": selected,
                "bold": style["bold"], "stroke_width": 1, "wrapped_lines": selected_lines,
                "line_count": len(selected_lines or []), "translation": text,
            })

    if active_count != 71:
        raise SystemExit(f"expected 71 translated active objects, got {active_count}")
    if failures:
        (OUT / "typography-failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
        raise SystemExit(f"typography preflight failed for {len(failures)} object(s); see typography-failures.json")
    if len(plan) != 71 or min(row["font_size"] for row in plan) < 28:
        raise SystemExit("fixed typography plan invalid")

    rendered_lookup: dict[tuple[int, int], Path] = {}
    rendered_objects = 0
    for page in manifest.get("pages") or []:
        source_page = int(page.get("source_page") or 0)
        slice_index = int(page.get("slice_index") or 0)
        image = Image.open(Path(page["clean"])).convert("RGB")
        for obj in page.get("text_objects") or []:
            if not isinstance(obj, dict) or obj.get("source_missing"):
                continue
            text = str(obj.get("translation") or "").strip()
            if not text:
                continue
            style = obj.get("style") or {}
            fs = style.get("fontSize")
            if not isinstance(fs, int) or fs < 28:
                raise SystemExit(f"render refused non-fixed/subminimum font for {obj.get('id')}: {fs!r}")
            r = obj["region"]
            image = render_text_in_box(
                image, text,
                (int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"])),
                font_name=style["font"], font_size=fs, is_bold=bool(style.get("bold")),
                fill=style.get("color", "auto"), stroke_width=style.get("strokeWidth", 1),
                stroke_color=style.get("strokeColor", "auto"), bg_color=style.get("bgColor", "transparent"),
                horizontal_align=style.get("horizontalAlign", "center"), vertical_align=style.get("verticalAlign", "middle"),
            )
            rendered_objects += 1
        render_path = RENDERED_SLICES / f"render_{source_page:03d}_{slice_index:02d}.png"
        image.save(render_path)
        rendered_lookup[(source_page, slice_index)] = render_path
        page["rendered"] = render_path.as_posix()

    if rendered_objects != 71:
        raise SystemExit(f"expected to render 71 objects, rendered {rendered_objects}")

    grouped: dict[int, list[dict]] = defaultdict(list)
    for page in manifest.get("pages") or []:
        grouped[int(page.get("source_page") or 0)].append(page)
    if sorted(grouped) != list(range(11)):
        raise SystemExit(f"expected source pages 0..10, got {sorted(grouped)}")

    stitch_seams = []
    for source_page in sorted(grouped):
        pages = sorted(grouped[source_page], key=lambda p: int(p.get("slice_index") or 0))
        first_img = Image.open(rendered_lookup[(source_page, int(pages[0].get("slice_index") or 0))]).convert("RGB")
        width = first_img.width
        source_height = int((pages[0].get("stitch_core") or {}).get("source_height") or 0)
        if source_height <= 0:
            source_height = max(int((p.get("stitch_core") or {}).get("core_source_y2") or 0) for p in pages)
        canvas = Image.new("RGB", (width, source_height), "white")
        intervals = []
        for p in pages:
            core = p.get("stitch_core") or {}
            cy1, cy2 = int(core.get("core_y1") or 0), int(core.get("core_y2") or 0)
            sy1, sy2 = int(core.get("core_source_y1") or 0), int(core.get("core_source_y2") or 0)
            if cy2 <= cy1 or sy2 <= sy1 or (cy2 - cy1) != (sy2 - sy1):
                raise SystemExit(f"invalid stitch core on source page {source_page}: {core}")
            slice_index = int(p.get("slice_index") or 0)
            im = Image.open(rendered_lookup[(source_page, slice_index)]).convert("RGB")
            canvas.paste(im.crop((0, cy1, width, cy2)), (0, sy1))
            intervals.append((sy1, sy2))
        intervals.sort()
        cursor = 0
        for sy1, sy2 in intervals:
            if sy1 != cursor:
                raise SystemExit(f"stitch ownership gap/overlap page {source_page}: cursor={cursor}, next={sy1}")
            cursor = sy2
        if cursor != source_height:
            raise SystemExit(f"stitch ownership does not reach source height page {source_page}: {cursor}/{source_height}")
        canvas.save(FINAL_PAGES / f"{source_page:03d}.png")
        for _, seam_y in intervals[:-1]:
            if 0 < seam_y < source_height:
                stitch_seams.append({"source_page": source_page, "y": seam_y})

    proof_sheets = build_proofs(manifest, rendered_lookup, plan, stitch_seams)

    size_counts = Counter(row["font_size"] for row in plan)
    font_counts = Counter(row["font"] for row in plan)
    role_counts = Counter(row["role"] for row in plan)
    fallbacks = [row for row in plan if row.get("font_fallback_reason")]
    low_size = [row for row in plan if row["font_size"] <= 30]

    typography = {
        "checkpoint": "05-typography-preflight",
        "chapter_id": CHAPTER_ID,
        "policy": "scene_emotion_context_aware",
        "normal_dialogue_font": "Mac-dinh-3",
        "narration_font": "Mac-dinh-2",
        "autosize_active_objects": 0,
        "hard_min_font_size": 28,
        "font_size_counts": {str(k): v for k, v in sorted(size_counts.items(), reverse=True)},
        "font_counts": dict(font_counts),
        "role_counts": dict(role_counts),
        "font_fallback_count": len(fallbacks),
        "objects_at_or_below_30px": len(low_size),
        "seam_region_clamps": seam_clamps,
        "active_objects": 71,
        "status": "PASS" if not low_size else "PASS_REVIEW_LOW_SIZE",
    }
    render_summary = {
        "checkpoint": "05-render-candidate",
        "chapter_id": CHAPTER_ID,
        "status": "REVIEW_REQUIRED",
        "rendered_story_objects": rendered_objects,
        "rendered_slices": len(rendered_lookup),
        "source_pages": 11,
        "expected_source_pages": 11,
        "stitch_seam_count": len(stitch_seams),
        "proof_sheets": proof_sheets,
        "full_page_overview": "proof/full-page-overview.jpg",
        "stitch_seam_proof": "proof/stitch-seams.jpg" if stitch_seams else None,
        "next_action": "HUMAN_VISUAL_REVIEW",
    }
    (OUT / "typography-plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "typography-summary.json").write_text(json.dumps(typography, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "rendered-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "render-summary.json").write_text(json.dumps(render_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "HUMAN_CHECKPOINT.txt").write_text(
        "CHECKPOINT: HUMAN VISUAL REVIEW REQUIRED\nInspect object proof, full pages, character emotion/font consistency, and stitch seams. Technical render success is not FINAL.\n",
        encoding="utf-8",
    )

    candidate_zip = OUT / "Cao-Vo-Ha-Canh-Den-Mot-Van-Nam-Sau-Chapter-351-VI-CANDIDATE.zip"
    with zipfile.ZipFile(candidate_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(FINAL_PAGES.glob("*.png")):
            zf.write(p, p.name)

    print(json.dumps(typography, ensure_ascii=False, indent=2))
    print(json.dumps(render_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
