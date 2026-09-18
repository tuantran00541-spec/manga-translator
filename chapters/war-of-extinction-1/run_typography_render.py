from __future__ import annotations

import json
import math
import shutil
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

from app.render.text_renderer import auto_detect_text_color, get_font_object, get_font_path
from app.routers.export import _core_range, _source_core_metadata, _validate_stitch_group, _stitch_png_to_file

CHAPTER_ID = "05efce51"
ROOT = Path("chapters/war-of-extinction-1")
OUT = Path("checkpoint/05-render")
PROOF = OUT / "proof"
RENDERED = OUT / "rendered_slices"
FINAL = OUT / "final_pages"
WORK = Path("data/processed") / CHAPTER_ID
RAW = Path("data/raw") / CHAPTER_ID

FONT_TEST = "Đắk Lắk, Nguyễn, phượng, khuỷu, quỹ, nghiễm, Ấ Ắ Ặ Ữ Ệ"
LADDER = [96, 90, 84, 78, 72, 66, 60, 56, 52, 48, 46, 44, 42, 40, 38, 36, 34, 32, 30, 28]
HARD_MIN = 28

SKILL_IDS = {
    "text_fc2019f421784cd5",
    "text_c291bdc565eb4954",
    "text_manual_p102_footwork",
    "text_e26baccf78af4e33",
}
SHOUT_IDS = {
    "text_4f05d727a97149fa", "text_7041a3527a084488", "text_30eaa646b03f4d06",
    "text_2ba20e5d637846f5", "text_1aa61677eb544b14", "text_cd9a74d423d3475f",
    "text_d641fb6daa444702", "text_629f955eaab14897", "text_c7137b4254eb4c0f",
    "text_5e23826bef30450e", "text_7d650a5906fc4213", "text_80a42b49311d47c0",
    "text_05f0f3640ef3416f", "text_5ee02c92f7ed45a3", "text_2a91d34416b64e82",
    "text_ac11bc24039d4ce6", "text_8fc2f3f6c6214727", "text_7c38ecd9830e496f",
    "text_25d31762d4ac4eb9",
}

def rect(r):
    return {k: int(r[k]) for k in ("x1", "y1", "x2", "y2")}

def clip_region(r, w, h):
    return {
        "x1": max(0, min(w, int(r["x1"]))),
        "y1": max(0, min(h, int(r["y1"]))),
        "x2": max(0, min(w, int(r["x2"]))),
        "y2": max(0, min(h, int(r["y2"]))),
    }

def expand_region(r, w, h, fx=0.08, fy=0.12, extra=4):
    rw=max(1,r["x2"]-r["x1"]); rh=max(1,r["y2"]-r["y1"])
    dx=int(round(rw*fx))+extra; dy=int(round(rh*fy))+extra
    return clip_region({"x1":r["x1"]-dx,"y1":r["y1"]-dy,"x2":r["x2"]+dx,"y2":r["y2"]+dy},w,h)

def inset_region(r, amount):
    a=max(0,int(amount))
    return {"x1":r["x1"]+a,"y1":r["y1"]+a,"x2":r["x2"]-a,"y2":r["y2"]-a}

def intersect_y(r, y1, y2):
    return {"x1":r["x1"],"y1":max(r["y1"],y1),"x2":r["x2"],"y2":min(r["y2"],y2)}

def valid_region(r, min_w=8, min_h=8):
    return r["x2"]-r["x1"]>=min_w and r["y2"]-r["y1"]>=min_h

@lru_cache(maxsize=None)
def font_cmap(font_name: str) -> frozenset[int]:
    p=get_font_path(font_name)
    if not p.is_file():
        return frozenset()
    tt=TTFont(str(p))
    cmap=set()
    for t in tt["cmap"].tables:
        cmap.update(t.cmap.keys())
    tt.close()
    return frozenset(cmap)

def missing_glyphs(font_name: str, text: str) -> list[str]:
    cmap=font_cmap(font_name)
    return sorted({ch for ch in text if ch not in "\n\r\t" and ord(ch) not in cmap})

def choose_font(role: str, text: str) -> tuple[str, str | None]:
    if role=="skill_name":
        special="Skill-fonts-1"
        if not missing_glyphs(special, FONT_TEST+"\n"+text):
            return special, None
        primary="Mac-dinh-3"
        miss=missing_glyphs(primary,FONT_TEST+"\n"+text)
        if miss:
            raise SystemExit(f"Mac-dinh-3 glyph gate failed for skill text {text!r}: {miss}")
        return primary, "Skill-fonts-1 rejected by full Vietnamese glyph gate; primary dialogue family selected"
    primary="Mac-dinh-3" if role in {"normal_dialogue","panic_shout"} else "Mac-dinh-2"
    miss=missing_glyphs(primary,FONT_TEST+"\n"+text)
    if miss:
        raise SystemExit(f"{primary} glyph gate failed for {text!r}: {miss}")
    return primary, None

def role_for(item: dict) -> str:
    oid=str(item["object_id"])
    if oid in SKILL_IDS:
        return "skill_name"
    sem=str(item.get("semantic_type") or "")
    if sem=="system_ui":
        return "system_ui"
    if sem=="free_text":
        return "free_text"
    if sem=="text":
        return "normal_narration"
    if oid in SHOUT_IDS:
        return "panic_shout"
    return "normal_dialogue"

def source_size_cap(item: dict, source_region: dict, role: str) -> float:
    src=str(item.get("source") or "")
    lines=max(1,len(src.splitlines()))
    h=max(1,source_region["y2"]-source_region["y1"])
    estimate=max(20.0,(h/lines)*0.88)
    mult={
        "normal_dialogue":1.25,
        "panic_shout":1.40,
        "normal_narration":1.20,
        "free_text":1.40,
        "skill_name":1.55,
        "system_ui":1.30,
    }[role]
    return min(96.0,max(28.0,estimate*mult))

def infer_bubble_body(image: Image.Image, source_region: dict, core_y1: int, core_y2: int):
    w,h=image.size
    sw=max(1,source_region["x2"]-source_region["x1"])
    sh=max(1,source_region["y2"]-source_region["y1"])
    dx=max(80,int(sw*0.85)); dy=max(90,int(sh*1.15))
    crop_box=(
        max(0,source_region["x1"]-dx),
        max(core_y1,source_region["y1"]-dy),
        min(w,source_region["x2"]+dx),
        min(core_y2,source_region["y2"]+dy),
    )
    if crop_box[2]<=crop_box[0] or crop_box[3]<=crop_box[1]:
        return None,"invalid_crop"
    gray=image.crop(crop_box).convert("L")
    arr=np.asarray(gray,dtype=np.uint8)
    binary=(arr>=225).astype(np.uint8)*255

    sx1=max(0,source_region["x1"]-crop_box[0])
    sy1=max(0,source_region["y1"]-crop_box[1])
    sx2=min(binary.shape[1],source_region["x2"]-crop_box[0])
    sy2=min(binary.shape[0],source_region["y2"]-crop_box[1])
    if sx2<=sx1 or sy2<=sy1:
        return None,"source_outside_crop"

    candidates=[]
    cx=(sx1+sx2)//2; cy=(sy1+sy2)//2
    candidates.append((cx,cy))
    for fy in (0.25,0.5,0.75):
        for fx in (0.25,0.5,0.75):
            candidates.append((int(sx1+(sx2-sx1)*fx),int(sy1+(sy2-sy1)*fy)))
    candidates=sorted(set(candidates),key=lambda p:int(arr[min(arr.shape[0]-1,max(0,p[1])),min(arr.shape[1]-1,max(0,p[0]))]),reverse=True)

    best=None
    for sx,sy in candidates:
        if not (0<=sx<binary.shape[1] and 0<=sy<binary.shape[0]) or binary[sy,sx]!=255:
            continue
        work=Image.fromarray(binary.copy(),mode="L")
        ImageDraw.floodfill(work,(sx,sy),128,thresh=0)
        comp=np.asarray(work)==128
        area=int(comp.sum())
        if best is None or area>best[0]:
            best=(area,comp)
    if best is None:
        return None,"no_white_seed"

    area,comp=best
    source_area=max(1,sw*sh)
    if area<max(250,int(source_area*0.75)):
        return None,"white_component_too_small"
    crop_area=comp.shape[0]*comp.shape[1]
    touches=bool(comp[0,:].any() or comp[-1,:].any() or comp[:,0].any() or comp[:,-1].any())
    if touches and area>crop_area*0.52:
        return None,"component_looks_like_page_background"

    counts=comp.sum(axis=1)
    max_row=int(counts.max()) if counts.size else 0
    if max_row<max(20,int(sw*0.55)):
        return None,"bubble_body_too_narrow"
    body_rows=np.where(counts>=max(12,int(max_row*0.45)))[0]
    if body_rows.size<3:
        return None,"bubble_body_rows_missing"

    # Prefer the contiguous dense-row run nearest the source center, dropping narrow tail rows.
    runs=[]
    start=prev=int(body_rows[0])
    for y in body_rows[1:]:
        y=int(y)
        if y==prev+1:
            prev=y
        else:
            runs.append((start,prev)); start=prev=y
    runs.append((start,prev))
    target_y=(sy1+sy2)/2
    run=min(runs,key=lambda rr:0 if rr[0]<=target_y<=rr[1] else min(abs(target_y-rr[0]),abs(target_y-rr[1])))
    ry1,ry2=run
    ys=np.arange(ry1,ry2+1)
    sub=comp[ys,:]
    yy,xx=np.where(sub)
    if xx.size<50:
        return None,"bubble_body_pixels_missing"
    xlo=int(np.percentile(xx,1.0)); xhi=int(np.percentile(xx,99.0))+1
    ylo=int(ry1); yhi=int(ry2)+1
    body={
        "x1":crop_box[0]+xlo,
        "y1":crop_box[1]+ylo,
        "x2":crop_box[0]+xhi,
        "y2":crop_box[1]+yhi,
    }
    body=clip_region(body,w,h)
    body=intersect_y(body,core_y1,core_y2)
    scx=(source_region["x1"]+source_region["x2"])/2
    scy=(source_region["y1"]+source_region["y2"])/2
    if not valid_region(body,40,30) or not(body["x1"]<=scx<=body["x2"] and body["y1"]<=scy<=body["y2"]):
        return None,"bubble_body_does_not_contain_source_center"
    if body["x2"]-body["x1"]<sw*0.92 or body["y2"]-body["y1"]<sh*0.80:
        return None,"bubble_body_smaller_than_source"
    return body,"clean_white_component"

def fallback_bubble_body(source_region: dict, w: int, h: int, core_y1: int, core_y2: int):
    sw=max(1,source_region["x2"]-source_region["x1"])
    sh=max(1,source_region["y2"]-source_region["y1"])
    body=clip_region({
        "x1":source_region["x1"]-int(sw*0.18)-12,
        "y1":source_region["y1"]-int(sh*0.28)-12,
        "x2":source_region["x2"]+int(sw*0.18)+12,
        "y2":source_region["y2"]+int(sh*0.28)+12,
    },w,h)
    return intersect_y(body,core_y1,core_y2)

def line_ink_bbox(draw, font, line, stroke):
    return draw.textbbox((0,0),line,font=font,stroke_width=stroke)

def ink_width(draw,font,line,stroke):
    b=line_ink_bbox(draw,font,line,stroke)
    return b[2]-b[0]

def balanced_wrap(draw, font, text_line: str, max_w: int, stroke: int):
    words=text_line.split()
    if not words:
        return [""],False
    if ink_width(draw,font,text_line,stroke)<=max_w:
        return [text_line],False
    for word in words:
        if ink_width(draw,font,word,stroke)>max_w:
            return None,False
    n=len(words)
    dp=[None]*(n+1)
    dp[n]=(0,[])
    for i in range(n-1,-1,-1):
        best=None
        for j in range(i+1,n+1):
            line=" ".join(words[i:j])
            w=ink_width(draw,font,line,stroke)
            if w>max_w:
                break
            rest=dp[j]
            if rest is None:
                continue
            slack=(max_w-w)/max(1,max_w)
            cost=slack*slack*1000
            count=j-i
            if count==1 and n>2:
                cost+=120 if j<n else 260
            if j==n and len(words[i])<=3 and n>2:
                cost+=400
            total=cost+rest[0]
            cand=(total,[line]+rest[1])
            if best is None or cand[0]<best[0]:
                best=cand
        dp[i]=best
    return (dp[0][1],True) if dp[0] else (None,False)

def layout_text(text: str, font_name: str, size: int, max_w: int, max_h: int, stroke: int):
    font=get_font_object(str(get_font_path(font_name)),size)
    canvas=Image.new("L",(max(8,max_w+40),max(8,max_h+40)),0)
    draw=ImageDraw.Draw(canvas)
    lines=[]; auto_split=False
    for explicit in str(text).splitlines() or [str(text)]:
        wrapped,split=balanced_wrap(draw,font,explicit.strip(),max_w,stroke)
        if wrapped is None:
            return None
        lines.extend(wrapped); auto_split=auto_split or split
    if not lines:
        return None

    accent_bbox=draw.textbbox((0,0),"ĐẮỮỆgỹ",font=font,stroke_width=stroke)
    accent_h=accent_bbox[3]-accent_bbox[1]
    line_height=max(int(math.ceil(size*1.10)),accent_h+max(2,int(size*0.06)))+stroke*2
    boxes=[draw.textbbox((0,0),ln,font=font,stroke_width=stroke) for ln in lines]
    rel_top=min(i*line_height+b[1] for i,b in enumerate(boxes))
    rel_bottom=max(i*line_height+b[3] for i,b in enumerate(boxes))
    rel_left=min(b[0] for b in boxes)
    rel_right=max(b[2] for b in boxes)
    width=max(b[2]-b[0] for b in boxes)
    height=rel_bottom-rel_top
    if width>max_w or height>max_h:
        return None
    return {
        "lines":lines,
        "line_height":line_height,
        "boxes":[list(b) for b in boxes],
        "ink_width":int(width),
        "ink_height":int(height),
        "rel_top":int(rel_top),
        "rel_bottom":int(rel_bottom),
        "auto_split":bool(auto_split),
    }

def choose_color_and_stroke(image: Image.Image, source_region: dict, role: str):
    box=(source_region["x1"],source_region["y1"],source_region["x2"],source_region["y2"])
    if role in {"normal_dialogue","panic_shout"}:
        crop=np.asarray(image.crop(box).convert("L"),dtype=np.uint8)
        median=float(np.median(crop)) if crop.size else 255.0
        if median<95:
            return (255,255,255),1,(0,0,0)
        return (16,16,16),0,(255,255,255)
    fill=auto_detect_text_color(image,box)
    lum=(fill[0]*299+fill[1]*587+fill[2]*114)/1000
    stroke=2 if role in {"free_text","skill_name"} else 1
    stroke_color=(0,0,0) if lum>128 else (255,255,255)
    return tuple(fill),stroke,stroke_color

def render_optical(image: Image.Image, plan: dict):
    draw=ImageDraw.Draw(image)
    font=get_font_object(str(get_font_path(plan["font"])),int(plan["font_size"]))
    r=plan["render_region"]
    cx=(r["x1"]+r["x2"])/2.0
    cy=(r["y1"]+r["y2"])/2.0
    lines=plan["wrapped_lines"]
    lh=int(plan["line_height"])
    stroke=int(plan["stroke_width"])
    boxes=[draw.textbbox((0,0),ln,font=font,stroke_width=stroke) for ln in lines]
    rel_top=min(i*lh+b[1] for i,b in enumerate(boxes))
    rel_bottom=max(i*lh+b[3] for i,b in enumerate(boxes))
    y_offset=cy-(rel_top+rel_bottom)/2.0

    bold_offsets=[(0,0)]
    if plan.get("bold"):
        bold_offsets=[(0,0),(-1,0),(1,0),(0,-1),(0,1)]

    abs_boxes=[]
    for i,(line,b) in enumerate(zip(lines,boxes)):
        x=cx-(b[0]+b[2])/2.0
        y=y_offset+i*lh
        for dx,dy in bold_offsets:
            draw.text((x+dx,y+dy),line,font=font,fill=tuple(plan["fill"]),
                      stroke_width=stroke,stroke_fill=tuple(plan["stroke_color"]))
        abs_boxes.append((x+b[0],y+b[1],x+b[2],y+b[3]))
    ink={
        "x1":int(math.floor(min(b[0] for b in abs_boxes))),
        "y1":int(math.floor(min(b[1] for b in abs_boxes))),
        "x2":int(math.ceil(max(b[2] for b in abs_boxes))),
        "y2":int(math.ceil(max(b[3] for b in abs_boxes))),
    }
    return image,ink

def proof_sheets(manifest, plan, rendered_lookup, base_lookup, seam_rows):
    PROOF.mkdir(parents=True,exist_ok=True)
    plan_by_id={r["object_id"]:r for r in plan}
    object_cells=[]
    for pi,page in enumerate(manifest.get("pages") or []):
        rendered=Image.open(rendered_lookup[pi]).convert("RGB")
        base=Image.open(base_lookup[pi]).convert("RGB")
        for obj in page.get("text_objects") or []:
            oid=str(obj.get("id") or "")
            if oid not in plan_by_id:
                continue
            row=plan_by_id[oid]
            rr=row["bubble_region"] if row.get("bubble_region") else row["source_text_region"]
            margin=55
            crop={
                "x1":max(0,rr["x1"]-margin),"y1":max(0,rr["y1"]-margin),
                "x2":min(base.width,rr["x2"]+margin),"y2":min(base.height,rr["y2"]+margin),
            }
            b=base.crop((crop["x1"],crop["y1"],crop["x2"],crop["y2"]))
            a=rendered.crop((crop["x1"],crop["y1"],crop["x2"],crop["y2"]))
            b.thumbnail((205,205),Image.Resampling.LANCZOS)
            a.thumbnail((205,205),Image.Resampling.LANCZOS)
            cell=Image.new("RGB",(450,285),"white")
            cell.paste(b,(5,55)); cell.paste(a,(230,55))
            d=ImageDraw.Draw(cell)
            d.text((5,5),f"{oid} p{pi:03d} role={row['role']}",fill="black")
            d.text((5,22),f"{row['font']} {row['font_size']}px lines={row['line_count']} margin={row.get('bubble_margin_em')}",fill="black")
            d.text((5,39),f"region={row['region_source']} auto_split={row['auto_split']} special={row.get('font_selection_note')}",fill="black")
            d.text((70,265),"CLEAN",fill="black"); d.text((295,265),"RENDER",fill="black")
            object_cells.append(cell)
    proof_names=[]
    per=12; cols=3; cw=450; ch=285
    for start in range(0,len(object_cells),per):
        batch=object_cells[start:start+per]
        rows=math.ceil(len(batch)/cols)
        sheet=Image.new("RGB",(cols*cw,rows*ch),"white")
        for i,c in enumerate(batch):
            sheet.paste(c,((i%cols)*cw,(i//cols)*ch))
        name=f"object-proof-{start//per:02d}.jpg"
        sheet.save(PROOF/name,quality=91)
        proof_names.append(name)

    # Full page overview in four compact sheets.
    full_names=[]
    pages=sorted(FINAL.glob("page_*.png"))
    per_page=10
    for start in range(0,len(pages),per_page):
        batch=pages[start:start+per_page]
        thumbs=[]
        for p in batch:
            im=Image.open(p).convert("RGB")
            im.thumbnail((420,620),Image.Resampling.LANCZOS)
            card=Image.new("RGB",(440,660),"white")
            d=ImageDraw.Draw(card); d.text((8,6),p.stem,fill="black")
            card.paste(im,((440-im.width)//2,28))
            thumbs.append(card)
        cols=2; rows=math.ceil(len(thumbs)/cols)
        sheet=Image.new("RGB",(880,rows*660),"white")
        for i,c in enumerate(thumbs):
            sheet.paste(c,((i%cols)*440,(i//cols)*660))
        name=f"full-page-overview-{start//per_page:02d}.jpg"
        sheet.save(PROOF/name,quality=88)
        full_names.append(name)

    seam_names=[]
    seam_cells=[]
    for s in seam_rows:
        p=FINAL/f"page_{s['source_page']+1:03d}.png"
        im=Image.open(p).convert("RGB")
        y=int(s["y"])
        crop=im.crop((0,max(0,y-130),im.width,min(im.height,y+130)))
        crop.thumbnail((620,240),Image.Resampling.LANCZOS)
        cell=Image.new("RGB",(650,285),"white")
        d=ImageDraw.Draw(cell); d.text((8,6),f"source {s['source_page']:02d} seam y={y}",fill="black")
        cell.paste(crop,((650-crop.width)//2,28))
        seam_cells.append(cell)
    per=12; cols=2; cw=650; ch=285
    for start in range(0,len(seam_cells),per):
        batch=seam_cells[start:start+per]
        rows=math.ceil(len(batch)/cols)
        sheet=Image.new("RGB",(cols*cw,rows*ch),"white")
        for i,c in enumerate(batch):
            sheet.paste(c,((i%cols)*cw,(i//cols)*ch))
        name=f"seam-proof-{start//per:02d}.jpg"
        sheet.save(PROOF/name,quality=90)
        seam_names.append(name)

    idx={"object_proofs":proof_names,"full_page_overviews":full_names,"seam_proofs":seam_names,
         "object_count":len(object_cells),"source_pages":len(pages),"seam_count":len(seam_rows)}
    (PROOF/"proof-index.json").write_text(json.dumps(idx,ensure_ascii=False,indent=2),encoding="utf-8")
    return idx

def main():
    for d in (OUT,PROOF,RENDERED,FINAL):
        d.mkdir(parents=True,exist_ok=True)

    tsum=json.loads(Path("checkpoint/04-translation/translation-review-summary.json").read_text(encoding="utf-8"))
    if tsum.get("status")!="PASS" or not tsum.get("editorial_pass") or tsum.get("active_untranslated_story_objects")!=0:
        raise SystemExit(f"translation gate is not PASS: {tsum}")
    manifest=json.loads(Path("checkpoint/04-translation/translated-manifest.json").read_text(encoding="utf-8"))
    tmap=json.loads((ROOT/"translation-map.json").read_text(encoding="utf-8"))
    entry={str(e["object_id"]):e for e in tmap.get("entries") or []}
    if len(entry)!=168 or len(manifest.get("pages") or [])!=139:
        raise SystemExit(f"unexpected manifest/map counts pages={len(manifest.get('pages') or [])} entries={len(entry)}")

    plan=[]
    failures=[]
    base_lookup={}
    rendered_lookup={}
    region_counts=Counter()
    role_counts=Counter()
    font_counts=Counter()
    size_counts=Counter()
    special_rejected=0
    seam_clamps=0
    auto_splits=0

    for pi,page in enumerate(manifest.get("pages") or []):
        original=Path(str(page.get("original") or ""))
        clean=Path(str(page.get("clean") or "")) if page.get("clean") else None
        base=original if page.get("skipped") else clean
        if base is None or not base.is_file():
            raise SystemExit(f"missing approved base image p{pi:03d}: {base}")
        base_lookup[pi]=base
        image=Image.open(base).convert("RGB")
        w,h=image.size
        core=page.get("stitch_core") or {}
        core_y1=int(core.get("core_y1") or 0)
        core_y2=int(core.get("core_y2") or h)
        boxes={str(b.get("id")):b for b in (page.get("boxes") or []) if isinstance(b,dict) and b.get("id")}

        for obj in page.get("text_objects") or []:
            if not isinstance(obj,dict) or obj.get("source_missing"):
                continue
            text=str(obj.get("translation") or "").strip()
            if not text:
                continue
            oid=str(obj.get("id") or "")
            if oid not in entry:
                failures.append({"object_id":oid,"reason":"missing_translation_map_entry"}); continue
            item=entry[oid]
            role=role_for(item)
            role_counts[role]+=1
            source_region=clip_region(rect(obj.get("region") or {}),w,h)
            source_region=intersect_y(source_region,core_y1,core_y2)
            if not valid_region(source_region,8,8):
                failures.append({"object_id":oid,"reason":"invalid_source_region","region":source_region}); continue

            if role in {"normal_dialogue","panic_shout"}:
                body,why=infer_bubble_body(image,source_region,core_y1,core_y2)
                if body is None:
                    body=fallback_bubble_body(source_region,w,h,core_y1,core_y2)
                    region_source="fallback_source_expansion:"+why
                else:
                    region_source=why
                if not valid_region(body,40,30):
                    failures.append({"object_id":oid,"reason":"invalid_bubble_body","body":body}); continue
            else:
                body=expand_region(source_region,w,h,0.025,0.045,4)
                body=intersect_y(body,core_y1,core_y2)
                region_source="display_source_region"

            font,note=choose_font(role,text)
            if note: special_rejected+=1
            font_counts[font]+=1
            fill,stroke,stroke_color=choose_color_and_stroke(image,source_region,role)
            bold=role=="panic_shout"
            cap=source_size_cap(item,source_region,role)
            ladder=[s for s in LADDER if s<=cap+0.5]
            if not ladder:
                ladder=[HARD_MIN]

            selected=None
            for size in ladder:
                margin=int(math.ceil(size*0.55)) if role in {"normal_dialogue","panic_shout"} else max(4,int(round(size*0.10)))
                safe=inset_region(body,margin)
                safe=intersect_y(safe,core_y1,core_y2)
                if not valid_region(safe,20,20):
                    continue
                layout=layout_text(text,font,size,safe["x2"]-safe["x1"],safe["y2"]-safe["y1"],stroke)
                if layout is None:
                    continue
                selected=(size,safe,layout,margin)
                break
            if selected is None:
                failures.append({"object_id":oid,"role":role,"font":font,"reason":"does_not_fit_at_28px","source_region":source_region,"bubble_region":body,"text":text})
                continue
            size,safe,layout,margin=selected
            size_counts[size]+=1
            auto_splits+=int(layout["auto_split"])

            # Predict final visible-ink bbox for breathing and clipping checks.
            cx=(safe["x1"]+safe["x2"])/2.0; cy=(safe["y1"]+safe["y2"])/2.0
            rel_top=layout["rel_top"]; rel_bottom=layout["rel_bottom"]
            yoff=cy-(rel_top+rel_bottom)/2.0
            font_obj=get_font_object(str(get_font_path(font)),size)
            probe=ImageDraw.Draw(Image.new("L",(10,10),0))
            abs_boxes=[]
            for i,line in enumerate(layout["lines"]):
                b=probe.textbbox((0,0),line,font=font_obj,stroke_width=stroke)
                x=cx-(b[0]+b[2])/2.0
                y=yoff+i*layout["line_height"]
                abs_boxes.append((x+b[0],y+b[1],x+b[2],y+b[3]))
            ink={
                "x1":int(math.floor(min(b[0] for b in abs_boxes))),
                "y1":int(math.floor(min(b[1] for b in abs_boxes))),
                "x2":int(math.ceil(max(b[2] for b in abs_boxes))),
                "y2":int(math.ceil(max(b[3] for b in abs_boxes))),
            }
            if ink["x1"]<safe["x1"] or ink["x2"]>safe["x2"] or ink["y1"]<safe["y1"] or ink["y2"]>safe["y2"]:
                failures.append({"object_id":oid,"reason":"predicted_ink_outside_render_region","ink":ink,"render_region":safe}); continue

            margin_em=None
            if role in {"normal_dialogue","panic_shout"}:
                margins=[ink["x1"]-body["x1"],body["x2"]-ink["x2"],ink["y1"]-body["y1"],body["y2"]-ink["y2"]]
                margin_em=round(min(margins)/max(1,size),3)
                if margin_em<0.45:
                    failures.append({"object_id":oid,"reason":"bubble_margin_below_hard_min","margin_em":margin_em,"bubble_region":body,"ink":ink}); continue

            if source_region["y1"]<core_y1 or source_region["y2"]>core_y2:
                seam_clamps+=1
            region_counts[region_source]+=1
            obj["source_text_region"]=source_region
            if role in {"normal_dialogue","panic_shout"}:
                obj["bubble_region"]=body
                obj["bubble_safe_region"]=safe
            obj["region"]=safe
            obj["style"]={
                "font":font,"fontSize":int(size),"bold":bold,
                "color":"#{:02x}{:02x}{:02x}".format(*fill),
                "strokeWidth":int(stroke),
                "strokeColor":"#{:02x}{:02x}{:02x}".format(*stroke_color),
                "bgColor":"transparent","horizontalAlign":"center","verticalAlign":"middle",
            }
            obj["typography_reviewed"]=True
            obj["font_size_fixed"]=True
            obj["typography_role"]=role
            row={
                "object_id":oid,"page_index":pi,"source_page":int(page.get("source_page") or 0),
                "slice_index":int(page.get("slice_index") or 0),"role":role,
                "source_text_region":source_region,"bubble_region":body if role in {"normal_dialogue","panic_shout"} else None,
                "bubble_safe_region":safe if role in {"normal_dialogue","panic_shout"} else None,
                "render_region":safe,"region_source":region_source,
                "font":font,"font_size":int(size),"font_selection_note":note,
                "font_fallback_reason":None,"bold":bold,"fill":list(fill),"stroke_width":int(stroke),
                "stroke_color":list(stroke_color),"wrapped_lines":layout["lines"],
                "line_count":len(layout["lines"]),"line_height":int(layout["line_height"]),
                "auto_split":bool(layout["auto_split"]),"compression_pct":0.0,
                "predicted_ink_bbox":ink,"bubble_margin_em":margin_em,
                "translation":text,
            }
            plan.append(row)

    if failures:
        (OUT/"typography-failures.json").write_text(json.dumps(failures,ensure_ascii=False,indent=2),encoding="utf-8")
        raise SystemExit(f"typography preflight failed for {len(failures)} object(s)")
    if len(plan)!=168:
        raise SystemExit(f"expected 168 typography rows, got {len(plan)}")
    if min(r["font_size"] for r in plan)<28:
        raise SystemExit("subminimum font size detected")
    if any(r["font_fallback_reason"] for r in plan):
        raise SystemExit("font fallback must be zero")

    plan_by_id={r["object_id"]:r for r in plan}
    rendered_objects=0
    for pi,page in enumerate(manifest.get("pages") or []):
        image=Image.open(base_lookup[pi]).convert("RGB")
        for obj in page.get("text_objects") or []:
            if not isinstance(obj,dict) or obj.get("source_missing"):
                continue
            oid=str(obj.get("id") or "")
            if oid not in plan_by_id:
                continue
            image,actual_ink=render_optical(image,plan_by_id[oid])
            plan_by_id[oid]["actual_ink_bbox"]=actual_ink
            rendered_objects+=1
        p=RENDERED/f"slice_{pi:03d}.png"
        image.save(p)
        rendered_lookup[pi]=p
        page["rendered"]=p.as_posix()
    if rendered_objects!=168:
        raise SystemExit(f"rendered object count {rendered_objects} != 168")

    groups=defaultdict(list)
    for pi,page in enumerate(manifest.get("pages") or []):
        source=int(page.get("source_page",pi))
        sr,sh=_source_core_metadata(page)
        groups[source].append({
            "page_index":pi,"slice_index":int(page.get("slice_index",0)),
            "core_range":_core_range(page),"source_core_range":sr,"source_height":sh,
            "path":rendered_lookup[pi],
        })
    if len(groups)!=40 or sorted(groups)!=list(range(40)):
        raise SystemExit(f"source page groups invalid: {sorted(groups)}")

    seam_rows=[]
    for source in sorted(groups):
        items=sorted(groups[source],key=lambda x:(x["slice_index"],x["page_index"]))
        validate=[{k:x[k] for k in ("slice_index","core_range","source_core_range","source_height")} for x in items]
        _validate_stitch_group(source,validate)
        cores=[x["core_range"] for x in items]
        out=FINAL/f"page_{source+1:03d}.png"
        _stitch_png_to_file([x["path"] for x in items],out,cores)
        source_intervals=sorted(x["source_core_range"] for x in items)
        for _,y in source_intervals[:-1]:
            seam_rows.append({"source_page":source,"y":int(y)})

    proof=proof_sheets(manifest,plan,rendered_lookup,base_lookup,seam_rows)
    typography={
        "checkpoint":"05-typography-preflight","chapter_id":CHAPTER_ID,"status":"PASS",
        "active_story_objects":168,"fixed_size_objects":168,"font_size_auto_objects":0,
        "font_fallback_count":0,"horizontal_compression_objects":0,"hard_min_font_size":28,
        "font_size_min_px":min(r["font_size"] for r in plan),"font_size_max_px":max(r["font_size"] for r in plan),
        "font_size_counts":{str(k):v for k,v in sorted(size_counts.items(),reverse=True)},
        "font_counts":dict(font_counts),"role_counts":dict(role_counts),
        "special_font_rejected_count":special_rejected,"bubble_region_sources":dict(region_counts),
        "auto_split_objects":auto_splits,"seam_clamped_source_regions":seam_clamps,
        "overflow_blockers":0,"clipping_blockers":0,"bubble_margin_blockers":0,"glyph_blockers":0,
        "compression_hard_cap_pct":6,"compression_used_pct":0,
        "normal_dialogue_font":"Mac-dinh-3","normal_narration_font":"Mac-dinh-2",
        "next_action":"HUMAN_VISUAL_REVIEW",
    }
    render={
        "checkpoint":"05-render-candidate","chapter_id":CHAPTER_ID,"status":"REVIEW_REQUIRED",
        "rendered_story_objects":168,"rendered_slices":139,"source_pages":40,
        "stitch_seam_count":len(seam_rows),"proof":proof,
        "next_action":"OBJECT_FULL_PAGE_SEAM_VISUAL_REVIEW"
    }
    (OUT/"typography-plan.json").write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT/"typography-summary.json").write_text(json.dumps(typography,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT/"render-summary.json").write_text(json.dumps(render,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT/"rendered-manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT/"HUMAN_CHECKPOINT.txt").write_text(
        "CHECKPOINT: HUMAN VISUAL REVIEW REQUIRED\nReview object proofs, all 40 full pages, and all stitch seams. Technical render PASS is not editorial PASS.\n",
        encoding="utf-8"
    )
    print(json.dumps(typography,ensure_ascii=False,indent=2))
    print(json.dumps(render,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
