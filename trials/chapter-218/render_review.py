#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from app.routers.export import export_chapter, render_chapter

CHAPTER_ID = "c2182180"
EXPECTED_SLICES = 69
EXPECTED_SOURCE_PAGES = 15
EXPECTED_ACTIVE = 98
EXPECTED_TOMBSTONES = 125
WARNING_IDS = {
    "text_b478707b63c54344",
    "text_64b56e211cab46fb",
    "text_5562636cb2f34195",
    "text_ca9d8fbd11e94caa",
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_tree(src: Path, dst: Path) -> None:
    if not src.is_dir():
        raise SystemExit(f"missing directory: {src}")
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)


def validate_inputs(clean_root: Path, repair_root: Path, typeset_root: Path, clean_review_path: Path):
    clean_review = read_json(clean_review_path)
    if clean_review.get("status") != "PASS" or clean_review.get("chapter_id") != CHAPTER_ID:
        raise SystemExit(f"clean human gate is not approved: {clean_review}")
    human = clean_review.get("human_review") or {}
    for key in (
        "raw_vs_clean_full_sheets_reviewed",
        "object_sheets_reviewed",
        "stitch_core_ownership_checked",
    ):
        if human.get(key) is not True:
            raise SystemExit(f"clean human gate missing {key}")
    for key in (
        "story_text_blockers",
        "artwork_damage_blockers",
        "source_story_residue_blockers",
    ):
        if human.get(key) != 0:
            raise SystemExit(f"clean human gate blocker {key}={human.get(key)}")

    clean_manifest = clean_root / "processed" / "manifest.json"
    if not clean_manifest.is_file():
        raise SystemExit(f"missing base clean manifest: {clean_manifest}")
    repair_meta = read_json(repair_root / "clean-repair.json")
    if repair_meta.get("chapter_id") != CHAPTER_ID:
        raise SystemExit("repair chapter identity mismatch")
    if repair_meta.get("raw_vs_repaired_pixel_diff") != 0:
        raise SystemExit("approved clean repair is not pixel-exact RAW restore")
    if int(repair_meta.get("page_index", -1)) != 68:
        raise SystemExit("unexpected clean repair target")

    type_summary = read_json(typeset_root / "typeset-preflight-summary.json")
    if type_summary.get("status") != "PASS":
        raise SystemExit(f"typeset checkpoint not PASS: {type_summary}")
    expected = {
        "active_story_objects": EXPECTED_ACTIVE,
        "styled_story_objects": EXPECTED_ACTIVE,
        "font_size_auto_count": 0,
        "overflow_count": 0,
        "missing_glyph_count": 0,
    }
    for key, value in expected.items():
        if type_summary.get(key) != value:
            raise SystemExit(f"typeset {key}: expected {value}, got {type_summary.get(key)}")
    return clean_review, repair_meta, type_summary


def restore_layout(clean_root: Path, repair_root: Path, typeset_root: Path):
    raw_dst = Path("data/raw") / CHAPTER_ID
    processed_dst = Path("data/processed") / CHAPTER_ID
    output_dst = Path("data/output") / CHAPTER_ID
    copy_tree(clean_root / "raw", raw_dst)
    copy_tree(clean_root / "processed", processed_dst)
    reset_dir(output_dst)

    # Approved local repair: restore non-story promo/credit slice pixel-exact RAW.
    repaired = repair_root / "overlay" / "processed" / "clean_014_04.png"
    if not repaired.is_file():
        raise SystemExit(f"missing approved clean overlay: {repaired}")
    shutil.copy2(repaired, processed_dst / "clean_014_04.png")

    typeset_manifest_path = typeset_root / "processed-manifest-typeset.json"
    manifest = read_json(typeset_manifest_path)
    if manifest.get("chapter_id") != CHAPTER_ID:
        raise SystemExit("typeset chapter identity mismatch")
    pages = manifest.get("pages") or []
    if len(pages) != EXPECTED_SLICES:
        raise SystemExit(f"expected {EXPECTED_SLICES} slices, got {len(pages)}")

    active = 0
    tombstones = 0
    warnings = []
    for pi, page in enumerate(pages):
        if page.get("skipped"):
            raise SystemExit(f"unexpected skipped render slice: {pi}")
        clean = Path(str(page.get("clean") or ""))
        original = Path(str(page.get("original") or ""))
        if not clean.is_file():
            raise SystemExit(f"missing clean image page {pi}: {clean}")
        if not original.is_file():
            raise SystemExit(f"missing raw image page {pi}: {original}")
        for obj in page.get("text_objects") or []:
            if not isinstance(obj, dict):
                continue
            if obj.get("source_missing"):
                tombstones += 1
                continue
            active += 1
            oid = str(obj.get("id") or "")
            text = str(obj.get("translation") or "").strip()
            style = obj.get("style") or {}
            font_size = str(style.get("fontSize") or "").strip().lower()
            if not oid or not text:
                raise SystemExit(f"active render object incomplete page={pi} id={oid}")
            if font_size == "auto" or not font_size.isdigit() or int(font_size) <= 0:
                raise SystemExit(f"active render object has non-fixed size page={pi} id={oid}: {font_size}")
            if oid in WARNING_IDS:
                warnings.append({"id": oid, "page_index": pi, "font_size": int(font_size), "region": obj.get("region"), "text": text})

    if active != EXPECTED_ACTIVE:
        raise SystemExit(f"expected {EXPECTED_ACTIVE} active objects, got {active}")
    if tombstones != EXPECTED_TOMBSTONES:
        raise SystemExit(f"expected {EXPECTED_TOMBSTONES} tombstones, got {tombstones}")
    if {x["id"] for x in warnings} != WARNING_IDS:
        raise SystemExit(f"warning object set mismatch: {warnings}")

    manifest_path = processed_dst / "manifest.json"
    write_json(manifest_path, manifest)
    return manifest, warnings


def validate_export(manifest: dict, export_zip: Path):
    if not export_zip.is_file():
        raise SystemExit(f"production export missing: {export_zip}")
    groups = defaultdict(list)
    for pi, page in enumerate(manifest.get("pages") or []):
        groups[int(page.get("source_page"))].append((pi, page))
    if sorted(groups) != list(range(EXPECTED_SOURCE_PAGES)):
        raise SystemExit(f"unexpected source page set: {sorted(groups)}")

    expected_dims = {}
    for source_page, items in groups.items():
        items.sort(key=lambda x: int(x[1].get("slice_index", 0)))
        first_clean = Path(str(items[0][1].get("clean")))
        with Image.open(first_clean) as im:
            width = im.width
        source_heights = {
            int((p.get("stitch_core") or {}).get("source_height"))
            for _, p in items
            if (p.get("stitch_core") or {}).get("source_height") is not None
        }
        if source_heights:
            if len(source_heights) != 1:
                raise SystemExit(f"source {source_page} inconsistent source heights: {source_heights}")
            height = source_heights.pop()
        elif len(items) == 1:
            with Image.open(first_clean) as im:
                height = im.height
        else:
            raise SystemExit(f"source {source_page} missing stitch source height")
        expected_dims[source_page] = (width, height)

    actual = []
    with zipfile.ZipFile(export_zip) as z:
        names = sorted(name for name in z.namelist() if name.lower().endswith(".png"))
        if len(names) != EXPECTED_SOURCE_PAGES:
            raise SystemExit(f"expected {EXPECTED_SOURCE_PAGES} stitched pages, got {len(names)}")
        for idx, name in enumerate(names):
            with z.open(name) as src, Image.open(src) as im:
                dims = im.size
            if dims != expected_dims[idx]:
                raise SystemExit(f"stitched dimensions mismatch source={idx}: expected {expected_dims[idx]}, got {dims}")
            actual.append({"source_page": idx, "file": name, "width": dims[0], "height": dims[1]})
    return actual


def make_page_contact_sheet(export_zip: Path, out_path: Path):
    thumbs = []
    with zipfile.ZipFile(export_zip) as z:
        names = sorted(name for name in z.namelist() if name.lower().endswith(".png"))
        for name in names:
            with z.open(name) as src, Image.open(src) as im:
                img = im.convert("RGB")
                img.thumbnail((220, 620), Image.Resampling.LANCZOS)
                thumbs.append((name, img.copy()))
    cols = 3
    cell_w, cell_h = 245, 665
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for i, (name, img) in enumerate(thumbs):
        x = (i % cols) * cell_w
        y = (i // cols) * cell_h
        sheet.paste(img, (x + (cell_w - img.width) // 2, y + 22))
        draw.text((x + 6, y + 5), name, fill="black", font=font)
    sheet.save(out_path, quality=90)


def make_object_proofs(manifest: dict, proof_dir: Path):
    proof_dir.mkdir(parents=True, exist_ok=True)
    cards = []
    warning_cards = []
    font = ImageFont.load_default()
    for pi, page in enumerate(manifest.get("pages") or []):
        render_path = Path("data/output") / CHAPTER_ID / f"page_{pi:03d}.png"
        clean_path = Path(str(page.get("clean")))
        if not render_path.is_file():
            raise SystemExit(f"missing rendered slice {render_path}")
        with Image.open(render_path) as rim, Image.open(clean_path) as cim:
            rendered = rim.convert("RGB")
            clean = cim.convert("RGB")
            for obj in page.get("text_objects") or []:
                if not isinstance(obj, dict) or obj.get("source_missing"):
                    continue
                oid = str(obj.get("id") or "")
                region = obj.get("region") or {}
                try:
                    x1, y1, x2, y2 = [int(region[k]) for k in ("x1", "y1", "x2", "y2")]
                except Exception:
                    continue
                margin = 55
                bbox = (
                    max(0, x1 - margin), max(0, y1 - margin),
                    min(rendered.width, x2 + margin), min(rendered.height, y2 + margin),
                )
                rcrop = rendered.crop(bbox)
                rcrop.thumbnail((430, 300), Image.Resampling.LANCZOS)
                card = Image.new("RGB", (460, 350), "white")
                card.paste(rcrop, ((460 - rcrop.width)//2, 28 + (300 - rcrop.height)//2))
                d = ImageDraw.Draw(card)
                style = obj.get("style") or {}
                d.text((7, 6), f"p{pi:03d} {oid} size={style.get('fontSize')} font={style.get('font')}", fill="black", font=font)
                cards.append(card)

                if oid in WARNING_IDS:
                    ccrop = clean.crop(bbox)
                    ccrop.thumbnail((360, 300), Image.Resampling.LANCZOS)
                    r2 = rendered.crop(bbox)
                    r2.thumbnail((360, 300), Image.Resampling.LANCZOS)
                    wcard = Image.new("RGB", (760, 350), "white")
                    wcard.paste(ccrop, ((370-ccrop.width)//2, 28+(300-ccrop.height)//2))
                    wcard.paste(r2, (390+(370-r2.width)//2, 28+(300-r2.height)//2))
                    wd = ImageDraw.Draw(wcard)
                    wd.text((7, 6), f"{oid} size={style.get('fontSize')} CLEAN", fill="black", font=font)
                    wd.text((397, 6), "RENDER", fill="black", font=font)
                    warning_cards.append(wcard)

    if len(cards) != EXPECTED_ACTIVE:
        raise SystemExit(f"expected {EXPECTED_ACTIVE} object proof cards, got {len(cards)}")

    cols = 2
    per_sheet = 12
    for start in range(0, len(cards), per_sheet):
        chunk = cards[start:start+per_sheet]
        rows = (len(chunk) + cols - 1) // cols
        sheet = Image.new("RGB", (cols*460, rows*350), "white")
        for j, card in enumerate(chunk):
            sheet.paste(card, ((j%cols)*460, (j//cols)*350))
        sheet.save(proof_dir / f"objects-{start:03d}-{start+len(chunk)-1:03d}.jpg", quality=91)

    if len(warning_cards) != len(WARNING_IDS):
        raise SystemExit(f"expected {len(WARNING_IDS)} warning cards, got {len(warning_cards)}")
    warning_sheet = Image.new("RGB", (760, len(warning_cards)*350), "white")
    for i, card in enumerate(warning_cards):
        warning_sheet.paste(card, (0, i*350))
    warning_sheet.save(proof_dir / "small-status-clean-vs-render.jpg", quality=94)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-root", required=True)
    ap.add_argument("--repair-root", required=True)
    ap.add_argument("--typeset-root", required=True)
    ap.add_argument("--clean-review", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    clean_root = Path(args.clean_root)
    repair_root = Path(args.repair_root)
    typeset_root = Path(args.typeset_root)
    checkpoint = Path(args.output)
    reset_dir(checkpoint)

    clean_review, repair_meta, type_summary = validate_inputs(
        clean_root, repair_root, typeset_root, Path(args.clean_review)
    )
    manifest, warnings = restore_layout(clean_root, repair_root, typeset_root)

    result = render_chapter(CHAPTER_ID)
    stats = result.get("chapter_render") or {}
    if stats.get("total") != EXPECTED_SLICES or stats.get("skipped") != 0:
        raise SystemExit(f"unexpected render stats: {stats}")
    if stats.get("rendered") != EXPECTED_SLICES:
        raise SystemExit(f"expected fresh render of {EXPECTED_SLICES} slices, got {stats}")

    export_chapter(CHAPTER_ID)
    export_zip = Path("data/output") / CHAPTER_ID / f"chapter_{CHAPTER_ID}.zip"
    dims = validate_export(manifest, export_zip)

    shutil.copy2(export_zip, checkpoint / "Pick-Me-Up-Infinite-Gacha-Chapter-218-VI-RENDER-REVIEW.zip")
    shutil.copy2(typeset_root / "typography-plan.json", checkpoint / "typography-plan.json")
    rendered_manifest = read_json(Path("data/processed") / CHAPTER_ID / "manifest.json")
    write_json(checkpoint / "processed-manifest-rendered.json", rendered_manifest)

    proof_dir = checkpoint / "proof"
    proof_dir.mkdir()
    make_page_contact_sheet(export_zip, proof_dir / "final-pages-contact-sheet.jpg")
    make_object_proofs(rendered_manifest, proof_dir)

    size_hist = Counter()
    for page in rendered_manifest.get("pages") or []:
        for obj in page.get("text_objects") or []:
            if isinstance(obj, dict) and not obj.get("source_missing"):
                size_hist[int((obj.get("style") or {}).get("fontSize"))] += 1

    summary = {
        "checkpoint": "06-render-review",
        "chapter_id": CHAPTER_ID,
        "status": "TECHNICAL_PASS_PENDING_HUMAN_VISUAL_REVIEW",
        "source_clean": clean_review.get("approved_clean_stack"),
        "clean_local_repair": {
            "page_index": repair_meta.get("page_index"),
            "action": repair_meta.get("action"),
            "raw_vs_repaired_pixel_diff": repair_meta.get("raw_vs_repaired_pixel_diff"),
        },
        "typeset_status": type_summary.get("status"),
        "slices": EXPECTED_SLICES,
        "source_pages": EXPECTED_SOURCE_PAGES,
        "active_story_objects": EXPECTED_ACTIVE,
        "persistent_tombstones": EXPECTED_TOMBSTONES,
        "render_stats": stats,
        "font_size_auto_count": 0,
        "font_size_range": {
            "min": min(size_hist) if size_hist else None,
            "max": max(size_hist) if size_hist else None,
        },
        "font_size_histogram": {str(k): v for k, v in sorted(size_hist.items(), reverse=True)},
        "small_status_review_objects": warnings,
        "final_page_dimensions": dims,
        "proof": {
            "full_pages": "proof/final-pages-contact-sheet.jpg",
            "object_sheets_glob": "proof/objects-*.jpg",
            "small_status": "proof/small-status-clean-vs-render.jpg",
        },
        "next_action": "HUMAN_VISUAL_REVIEW_OBJECTS_AND_FULL_PAGES",
    }
    write_json(checkpoint / "render-review-summary.json", summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in {"font_size_histogram","final_page_dimensions","small_status_review_objects"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
