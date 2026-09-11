#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from collections import Counter
from pathlib import Path

from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw

from app.config import DEFAULT_FONT
from app.parameters import RENDER_DEFAULT_PADDING, RENDER_PADDING_RATIO_MAX
from app.render.text_renderer import _fits

EXPECTED_ACTIVE = 110
MIN_SIZE = 16
MAX_SIZE = 96
SOFT_WARN = 22

SYSTEM_UI_IDS = {
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

PRIMARY_CANDIDATES = ["Mac-dinh-3.ttf", "Manga-fonts.ttf", "Mac-dinh-2.ttf", "default.ttf"]
SYSTEM_CANDIDATES = ["BeVietnamPro-SemiBold.ttf", "Mac-dinh-3.ttf", "default.ttf"]

def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def cmap(path: Path):
    f=TTFont(str(path), lazy=True)
    try:
        return set((f.getBestCmap() or {}).keys())
    finally:
        f.close()

def visible_cps(texts):
    return {ord(ch) for t in texts for ch in t if not ch.isspace()}

def choose_font(font_dir: Path, names, texts, fallback=None):
    req=visible_cps(texts)
    checks=[]
    for name in names:
        p=font_dir/name
        if not p.is_file():
            checks.append({"font":name,"exists":False,"missing":len(req)})
            continue
        miss=req-cmap(p)
        checks.append({"font":name,"exists":True,"missing":len(miss)})
        if not miss:
            return p, checks
    if fallback and Path(fallback).is_file():
        p=Path(fallback); miss=req-cmap(p)
        checks.append({"font":p.name,"exists":True,"missing":len(miss),"fallback":True})
        if not miss: return p,checks
    return None,checks

def load_stats(text):
    words=[w for w in text.replace("\n"," ").split() if w]
    chars=sum(not c.isspace() for c in text)
    return len(words), chars, max([len(w) for w in words] or [0])

def ceiling_for(text, w, h):
    words, chars, longest=load_stats(text)
    if words <= 2: ratio=.72
    elif words <= 5: ratio=.60
    elif words <= 9: ratio=.50
    elif words <= 14: ratio=.42
    elif words <= 22: ratio=.35
    elif words <= 32: ratio=.30
    else: ratio=.26
    aspect=w/max(1,h)
    if aspect >= 2.4: ratio*=1.10
    elif aspect <= .75: ratio*=.90
    area=max(1,w*h)
    comfort=round(h*ratio)
    area_cap=round(math.sqrt(area)*.42)
    ceil=max(MIN_SIZE,min(MAX_SIZE,comfort,max(24,area_cap)))
    floor=max(MIN_SIZE,min(24,round(min(w,h)*.11)))
    floor=min(floor,ceil)
    return int(ceil),int(floor),{
        "word_count":words,"char_count":chars,"longest_word":longest,
        "box_aspect_ratio":round(aspect,3),"box_area":area,
        "text_density":round(chars/math.sqrt(area),4),
        "comfort_ceiling":int(ceil),"adaptive_floor":int(floor)
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--translation-root",required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()
    src=Path(args.translation_root); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    manifest=load(src/"processed-manifest-translated.json")
    ts=load(src/"translation-review-summary.json")
    failures=[]
    if ts.get("status")!="PASS" or ts.get("active_untranslated_story_objects")!=0:
        failures.append("translation checkpoint not PASS/complete")

    active=[]; tomb=[]
    for pi,page in enumerate(manifest.get("pages") or []):
        for obj in page.get("text_objects") or []:
            if not isinstance(obj,dict): continue
            (tomb if obj.get("source_missing") else active).append((pi,page,obj))
    if len(active)!=EXPECTED_ACTIVE:
        failures.append(f"expected {EXPECTED_ACTIVE} active objects, got {len(active)}")

    font_dir=DEFAULT_FONT.parent
    system_texts=[str(o.get("translation") or "") for _,_,o in active if o.get("id") in SYSTEM_UI_IDS]
    dialogue_texts=[str(o.get("translation") or "") for _,_,o in active if o.get("id") not in SYSTEM_UI_IDS]
    primary,primary_checks=choose_font(font_dir,PRIMARY_CANDIDATES,dialogue_texts,fallback=DEFAULT_FONT)
    if primary is None:
        primary=DEFAULT_FONT; failures.append("no primary font covers dialogue Vietnamese glyphs")
    system,system_checks=choose_font(font_dir,SYSTEM_CANDIDATES,system_texts,fallback=primary)
    if system is None:
        system=primary; failures.append("no system font covers system Vietnamese glyphs")

    font_cmaps={str(primary):cmap(primary),str(system):cmap(system)}
    draw=ImageDraw.Draw(Image.new("RGB",(8,8),"white"))
    overflow=[]; glyph_missing=[]; warnings=[]; plan={}; size_hist=Counter(); font_hist=Counter()

    for pi,page,obj in active:
        oid=str(obj.get("id")); text=str(obj.get("translation") or "").strip()
        reg=obj.get("region") or {}
        try: x1,y1,x2,y2=[int(reg[k]) for k in ("x1","y1","x2","y2")]
        except Exception:
            failures.append(f"{oid}: malformed region"); continue
        rw,rh=x2-x1,y2-y1
        if rw<=0 or rh<=0 or not text:
            failures.append(f"{oid}: invalid region/empty text"); continue
        fp=system if oid in SYSTEM_UI_IDS else primary
        missing=sorted({ord(c) for c in text if not c.isspace()}-font_cmaps[str(fp)])
        if missing:
            glyph_missing.append({"id":oid,"font":fp.name,"codepoints":missing})
            failures.append(f"{oid}: missing glyphs"); continue

        pad=max(2,min(RENDER_DEFAULT_PADDING,int(min(rw,rh)*RENDER_PADDING_RATIO_MAX)))
        bw,bh=rw-pad*2,rh-pad*2
        ceil,floor,stats=ceiling_for(text,bw,bh)
        picked=None; lines=None; stroke=2
        for size in range(ceil,floor-1,-1):
            ok,ls=_fits(draw,text,bw,bh,str(fp),size,stroke)
            if ok:
                picked=size; lines=ls; break
        if picked is None:
            for size in range(floor-1,MIN_SIZE-1,-1):
                ok,ls=_fits(draw,text,bw,bh,str(fp),size,stroke)
                if ok: picked=size; lines=ls; break
        if picked is None:
            overflow.append({"id":oid,"page_index":pi,"source_page":page.get("source_page"),
                             "slice_index":page.get("slice_index"),"region":reg,"text":text,"sizing":stats})
            failures.append(f"{oid}: cannot fit at >= {MIN_SIZE}px"); continue
        if picked<SOFT_WARN:
            warnings.append({"id":oid,"font_size":picked,"source_page":page.get("source_page"),
                             "slice_index":page.get("slice_index"),"text":text})
        style=obj.setdefault("style",{})
        style.update({
            "font":fp.stem,
            "fontSize":int(picked),
            "bold":False,
            "strokeWidth":stroke,
            "strokeColor":"auto",
            "color":"auto",
            "bgColor":"transparent",
            "horizontalAlign":"center",
            "verticalAlign":"middle",
        })
        size_hist[picked]+=1; font_hist[fp.stem]+=1
        plan[oid]={
            "font":fp.stem,"font_size":picked,"semantic":"system_ui" if oid in SYSTEM_UI_IDS else "dialogue",
            "source_page":page.get("source_page"),"slice_index":page.get("slice_index"),
            "region":reg,"text":text,"wrapped_lines":lines,"sizing":stats
        }

    auto_count=0
    for _,_,obj in active:
        if str((obj.get("style") or {}).get("fontSize")).lower()=="auto": auto_count+=1
    if auto_count: failures.append(f"{auto_count} active objects still use auto font size")
    if len(plan)!=len(active): failures.append(f"styled {len(plan)}/{len(active)} active objects")

    summary={
        "checkpoint":"05-typeset-preflight","chapter_id":manifest.get("chapter_id"),
        "status":"PASS" if not failures else "FAIL",
        "source_checkpoint":"04-translation-review",
        "active_story_objects":len(active),"styled_objects":len(plan),"tombstones":len(tomb),
        "auto_font_size_objects":auto_count,"glyph_missing_count":len(glyph_missing),
        "overflow_count":len(overflow),"readability_warning_count":len(warnings),
        "font_usage":dict(font_hist),"size_histogram":{str(k):v for k,v in sorted(size_hist.items())},
        "min_font_size":min(size_hist) if size_hist else None,
        "max_font_size":max(size_hist) if size_hist else None,
        "primary_font":primary.stem,"system_font":system.stem,
        "primary_font_checks":primary_checks,"system_font_checks":system_checks,
        "readability_warnings":warnings,"glyph_missing":glyph_missing,"overflow":overflow,
        "failures":failures,
        "next_action":"RENDER_REVIEW" if not failures else "LOCAL_TYPESET_REPAIR"
    }
    manifest["typography_review"]={"status":summary["status"],"auto_font_size_objects":auto_count,
                                   "overflow_count":len(overflow),"glyph_missing_count":len(glyph_missing)}
    dump(out/"processed-manifest-typeset.json",manifest)
    dump(out/"typography-plan.json",{"chapter_id":manifest.get("chapter_id"),"objects":plan})
    dump(out/"typeset-summary.json",summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    if failures: raise SystemExit(1)

if __name__=="__main__":
    main()
