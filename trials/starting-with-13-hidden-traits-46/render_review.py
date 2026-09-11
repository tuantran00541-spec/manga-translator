#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from app.routers.export import export_chapter, render_chapter

CHAPTER_ID = "57d7b970"
EXPECTED_SLICES = 77
EXPECTED_SOURCE_PAGES = 22
EXPECTED_ACTIVE = 110
EXPECTED_TOMBSTONES = 147
SYSTEM_IDS = {
    "text_d847e89ab6ef448d",
    "text_6157d020e7f243be",
    "text_81122ec2d7714746",
    "text_9ab1ac919338430f",
    "text_15847a03ef9343b5",
    "text_8d32516791c14e75",
    "text_6546120bd57b4933",
    "text_b22061c66e774e72",
    "text_238aa6433e094783",
    "text_62d7cc543e6b438a",
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def reset_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_tree(src: Path, dst: Path):
    if not src.is_dir():
        raise SystemExit(f"missing directory: {src}")
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)


def validate_and_restore(clean_root: Path, repair_b: Path, repair_c: Path, typeset_root: Path, review_path: Path):
    review = read_json(review_path)
    if review.get("status") != "PASS" or review.get("chapter_id") != CHAPTER_ID:
        raise SystemExit("CLEAN human gate not PASS")
    human = review.get("human_review") or {}
    for key in ("story_text_blockers", "artwork_damage_blockers", "source_story_residue_blockers", "overlap_or_seam_blockers"):
        if human.get(key) != 0:
            raise SystemExit(f"CLEAN blocker {key}={human.get(key)}")
    repairs = review.get("local_repairs") or []
    if len(repairs) != 3 or any(r.get("raw_vs_repaired_pixel_diff") != 0 for r in repairs):
        raise SystemExit("CLEAN repair evidence incomplete")

    raw_dst = Path("data/raw") / CHAPTER_ID
    proc_dst = Path("data/processed") / CHAPTER_ID
    out_dst = Path("data/output") / CHAPTER_ID
    copy_tree(clean_root / "raw", raw_dst)
    copy_tree(clean_root / "processed", proc_dst)
    reset_dir(out_dst)

    overlays = [
        repair_b / "overlay/processed/clean_000_00.png",
        repair_b / "overlay/processed/clean_021_04.png",
        repair_c / "overlay/processed/clean_005_00.png",
    ]
    for src in overlays:
        if not src.is_file():
            raise SystemExit(f"missing approved overlay: {src}")
        shutil.copy2(src, proc_dst / src.name)

    ts = read_json(typeset_root / "typeset-summary.json")
    if ts.get("status") != "PASS":
        raise SystemExit(f"typeset not PASS: {ts}")
    for key, value in {
        "active_story_objects": EXPECTED_ACTIVE,
        "styled_objects": EXPECTED_ACTIVE,
        "auto_font_size_objects": 0,
        "glyph_missing_count": 0,
        "overflow_count": 0,
    }.items():
        if ts.get(key) != value:
            raise SystemExit(f"typeset {key}: expected {value}, got {ts.get(key)}")
    if ts.get("system_font") != "BeVietnamPro-SemiBold":
        raise SystemExit(f"unexpected system font: {ts.get('system_font')}")

    manifest = read_json(typeset_root / "processed-manifest-typeset.json")
    if manifest.get("chapter_id") != CHAPTER_ID:
        raise SystemExit("typeset identity mismatch")
    pages = manifest.get("pages") or []
    if len(pages) != EXPECTED_SLICES:
        raise SystemExit(f"expected {EXPECTED_SLICES} slices, got {len(pages)}")

    active = tomb = auto = 0
    sys_seen = set()
    for pi, page in enumerate(pages):
        if page.get("skipped"):
            raise SystemExit(f"unexpected skipped slice {pi}")
        for obj in page.get("text_objects") or []:
            if not isinstance(obj, dict):
                continue
            if obj.get("source_missing"):
                tomb += 1
                continue
            active += 1
            text = str(obj.get("translation") or "").strip()
            style = obj.get("style") or {}
            size = style.get("fontSize")
            if not text:
                raise SystemExit(f"active object without translation: {obj.get('id')}")
            if str(size).lower() == "auto" or not str(size).isdigit():
                auto += 1
            oid = str(obj.get("id") or "")
            if oid in SYSTEM_IDS:
                sys_seen.add(oid)
                if style.get("font") != "BeVietnamPro-SemiBold":
                    raise SystemExit(f"system object {oid} font={style.get('font')}")
    if active != EXPECTED_ACTIVE or tomb != EXPECTED_TOMBSTONES or auto != 0:
        raise SystemExit(f"manifest counts active={active} tomb={tomb} auto={auto}")
    if sys_seen != SYSTEM_IDS:
        raise SystemExit(f"system ID mismatch missing={sorted(SYSTEM_IDS-sys_seen)}")

    write_json(proc_dst / "manifest.json", manifest)
    return manifest, ts


def expected_dimensions(manifest: dict):
    groups = defaultdict(list)
    for page in manifest.get("pages") or []:
        groups[int(page.get("source_page"))].append(page)
    if sorted(groups) != list(range(EXPECTED_SOURCE_PAGES)):
        raise SystemExit(f"source page set mismatch: {sorted(groups)}")
    dims = {}
    for source_page, pages in groups.items():
        pages.sort(key=lambda p: int(p.get("slice_index", 0)))
        with Image.open(Path(str(pages[0]["clean"]))) as im:
            width = im.width
        heights = {int((p.get("stitch_core") or {}).get("source_height")) for p in pages if (p.get("stitch_core") or {}).get("source_height") is not None}
        if len(heights) != 1:
            raise SystemExit(f"source {source_page} source_height mismatch: {heights}")
        dims[source_page] = (width, heights.pop())
    return dims


def extract_and_validate_zip(export_zip: Path, manifest: dict, pages_dir: Path):
    reset_dir(pages_dir)
    expected = expected_dimensions(manifest)
    final_dims = []
    with zipfile.ZipFile(export_zip) as z:
        names = sorted(n for n in z.namelist() if n.lower().endswith(".png"))
        if names != [f"page_{i:03d}.png" for i in range(1, EXPECTED_SOURCE_PAGES + 1)]:
            raise SystemExit(f"unexpected export page order: {names}")
        for source_page, name in enumerate(names):
            target = pages_dir / name
            with z.open(name) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            with Image.open(target) as im:
                dims = im.size
            if dims != expected[source_page]:
                raise SystemExit(f"page {source_page} dims {dims} != {expected[source_page]}")
            final_dims.append({"source_page": source_page, "file": name, "width": dims[0], "height": dims[1]})
    return final_dims


def make_page_sheet(pages_dir: Path, out_path: Path):
    files = sorted(pages_dir.glob("page_*.png"))
    cols, cell_w, cell_h = 3, 250, 670
    rows = (len(files)+cols-1)//cols
    sheet = Image.new("RGB", (cols*cell_w, rows*cell_h), "white")
    draw = ImageDraw.Draw(sheet); font = ImageFont.load_default()
    for i, path in enumerate(files):
        with Image.open(path) as im:
            img = im.convert("RGB")
            img.thumbnail((225, 625), Image.Resampling.LANCZOS)
        x=(i%cols)*cell_w; y=(i//cols)*cell_h
        sheet.paste(img, (x+(cell_w-img.width)//2, y+28))
        draw.text((x+6,y+7), path.name, fill="black", font=font)
    sheet.save(out_path, quality=91)


def make_object_sheets(manifest: dict, proof_dir: Path):
    proof_dir.mkdir(parents=True, exist_ok=True)
    cards=[]; sys_cards=[]; font=ImageFont.load_default()
    for pi,page in enumerate(manifest.get("pages") or []):
        rpath=Path("data/output")/CHAPTER_ID/f"page_{pi:03d}.png"
        if not rpath.is_file():
            raise SystemExit(f"missing rendered slice {rpath}")
        with Image.open(rpath) as im:
            rendered=im.convert("RGB")
            for obj in page.get("text_objects") or []:
                if not isinstance(obj,dict) or obj.get("source_missing"):
                    continue
                reg=obj.get("region") or {}
                try: x1,y1,x2,y2=[int(reg[k]) for k in ("x1","y1","x2","y2")]
                except Exception: raise SystemExit(f"bad region {obj.get('id')}")
                m=45
                box=(max(0,x1-m),max(0,y1-m),min(rendered.width,x2+m),min(rendered.height,y2+m))
                crop=rendered.crop(box); crop.thumbnail((430,285),Image.Resampling.LANCZOS)
                card=Image.new("RGB",(460,330),"white")
                card.paste(crop,((460-crop.width)//2,32+(285-crop.height)//2))
                d=ImageDraw.Draw(card); st=obj.get("style") or {}; oid=str(obj.get("id") or "")
                d.text((6,7),f"p{pi:03d} {oid} {st.get('font')} {st.get('fontSize')}px",fill="black",font=font)
                cards.append(card)
                if oid in SYSTEM_IDS: sys_cards.append(card.copy())
    if len(cards)!=EXPECTED_ACTIVE or len(sys_cards)!=len(SYSTEM_IDS):
        raise SystemExit(f"proof count active={len(cards)} system={len(sys_cards)}")
    per=12; cols=2
    names=[]
    for start in range(0,len(cards),per):
        chunk=cards[start:start+per]; rows=(len(chunk)+1)//2
        sheet=Image.new("RGB",(920,rows*330),"white")
        for j,c in enumerate(chunk): sheet.paste(c,((j%2)*460,(j//2)*330))
        name=f"objects-{start:03d}-{start+len(chunk)-1:03d}.jpg"; sheet.save(proof_dir/name,quality=92); names.append(name)
    rows=(len(sys_cards)+1)//2
    sys_sheet=Image.new("RGB",(920,rows*330),"white")
    for j,c in enumerate(sys_cards): sys_sheet.paste(c,((j%2)*460,(j//2)*330))
    sys_sheet.save(proof_dir/"system-ui-review.jpg",quality=94)
    return names


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--clean-root",required=True)
    ap.add_argument("--repair-b",required=True)
    ap.add_argument("--repair-c",required=True)
    ap.add_argument("--typeset-root",required=True)
    ap.add_argument("--clean-review",required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()
    out=Path(args.output); reset_dir(out)

    manifest,ts=validate_and_restore(Path(args.clean_root),Path(args.repair_b),Path(args.repair_c),Path(args.typeset_root),Path(args.clean_review))
    result=render_chapter(CHAPTER_ID)
    stats=result.get("chapter_render") or {}
    if stats.get("total")!=EXPECTED_SLICES or stats.get("rendered")!=EXPECTED_SLICES or stats.get("skipped")!=0:
        raise SystemExit(f"unexpected render stats {stats}")
    export_chapter(CHAPTER_ID)
    export_zip=Path("data/output")/CHAPTER_ID/f"chapter_{CHAPTER_ID}.zip"
    if not export_zip.is_file(): raise SystemExit("production export ZIP missing")

    pages_dir=out/"pages"
    final_dims=extract_and_validate_zip(export_zip,manifest,pages_dir)
    candidate=out/"Starting-With-13-Hidden-Traits-Chapter-46-VI-RENDER-REVIEW.zip"
    shutil.copy2(export_zip,candidate)
    shutil.copy2(Path(args.typeset_root)/"typography-plan.json",out/"typography-plan.json")
    proof=out/"proof"; proof.mkdir(parents=True,exist_ok=True)
    make_page_sheet(pages_dir,proof/"full-pages-contact-sheet.jpg")
    object_sheets=make_object_sheets(manifest,proof)

    summary={
        "checkpoint":"06-render-review",
        "chapter_id":CHAPTER_ID,
        "status":"TECHNICAL_PASS_PENDING_HUMAN_VISUAL_REVIEW",
        "slices":EXPECTED_SLICES,
        "source_pages":EXPECTED_SOURCE_PAGES,
        "active_story_objects":EXPECTED_ACTIVE,
        "persistent_tombstones":EXPECTED_TOMBSTONES,
        "font_size_auto_count":0,
        "system_font":ts.get("system_font"),
        "render_stats":stats,
        "final_page_dimensions":final_dims,
        "candidate_zip":candidate.name,
        "page_contact_sheet":"proof/full-pages-contact-sheet.jpg",
        "object_sheets":[f"proof/{n}" for n in object_sheets],
        "system_ui_sheet":"proof/system-ui-review.jpg",
        "system_ui_ids":sorted(SYSTEM_IDS),
        "next_action":"HUMAN_VISUAL_REVIEW"
    }
    write_json(out/"render-review-summary.json",summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
