from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.inpaint.lama_inpainter import Inpainter

CHAPTER_ID = "f1cd0121"


def sha(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def paths(root: Path, manifest: dict, page_index: int) -> tuple[Path, Path]:
    page = manifest["pages"][page_index]
    raw = root / "raw" / "sliced" / Path(str(page["original"])).name
    clean = root / "processed" / Path(str(page["clean"])).name
    if not raw.is_file() or not clean.is_file():
        raise FileNotFoundError(f"missing p{page_index:03d}: raw={raw} clean={clean}")
    return raw, clean


def p36_mask(image: np.ndarray) -> np.ndarray:
    # Human-reviewed SQUEE! immediately above a panel border.  The mask is
    # deliberately local; the structural border is reconstructed afterwards.
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[3145:min(3272, h), 20:min(245, w)] = 255
    return mask


def p38_mask(image: np.ndarray) -> np.ndarray:
    # SQUEEE! is dark blue on a flat white speech shape. Select blue ink only
    # inside the reviewed text footprint, excluding the surrounding costume and
    # most of the speech-shape outline. A small dilation includes antialiasing.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, np.array([103, 105, 35]), np.array([130, 255, 255]))
    yy, xx = np.indices(blue.shape)
    zone = (
        ((xx >= 172) & (xx <= 735) & (yy >= 2345) & (yy <= 2640))
        | ((xx >= 163) & (xx <= 305) & (yy >= 2308) & (yy < 2355))
    )
    mask = np.where(zone, blue, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    return cv2.dilate(mask, kernel, iterations=1)


def focused_card(raw: np.ndarray, before: np.ndarray, after: np.ndarray, page: int, bbox: tuple[int,int,int,int]) -> Image.Image:
    x1,y1,x2,y2=bbox
    pad=55
    x1=max(0,x1-pad); y1=max(0,y1-pad); x2=min(raw.shape[1],x2+pad); y2=min(raw.shape[0],y2+pad)
    ims=[]
    for arr in (raw,before,after):
        im=Image.fromarray(cv2.cvtColor(arr[y1:y2,x1:x2],cv2.COLOR_BGR2RGB))
        im.thumbnail((340,300),Image.Resampling.LANCZOS); ims.append(im)
    card=Image.new("RGB",(1050,345),"white")
    for i,im in enumerate(ims): card.paste(im,(i*350+(340-im.width)//2,25+(300-im.height)//2))
    d=ImageDraw.Draw(card); f=ImageFont.load_default()
    d.text((5,5),f"p{page:03d} late story-SFX repair",fill="black",font=f)
    for x,label in ((145,"RAW"),(490,"BEFORE"),(830,"AFTER")): d.text((x,325),label,fill="black",font=f)
    return card


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("root",type=Path); args=ap.parse_args()
    root=args.root.resolve(); mp=root/"processed"/"manifest.json"
    manifest=json.loads(mp.read_text(encoding="utf-8"))
    if manifest.get("chapter_id") != CHAPTER_ID: raise SystemExit("chapter mismatch")

    specs=[
        (36,(20,3145,245,3272),p36_mask,"SQUEE! story vocalization missed by CLEAN review"),
        (38,(155,2300,775,2650),p38_mask,"SQUEEE! story vocalization missed by CLEAN review"),
    ]
    inp=Inpainter(); ops=[]; cards=[]
    for page,bbox,mask_fn,reason in specs:
        raw_path,clean_path=paths(root,manifest,page)
        raw=cv2.imread(str(raw_path),cv2.IMREAD_COLOR); before=cv2.imread(str(clean_path),cv2.IMREAD_COLOR)
        if raw is None or before is None or raw.shape != before.shape: raise RuntimeError(f"bad p{page:03d}")
        mask=mask_fn(before)
        if not np.any(mask): raise RuntimeError(f"empty mask p{page:03d}")
        after=inp.inpaint_mask(before.copy(),mask,force_lama=True)
        if page == 36:
            # Source SFX crosses the panel top border. Rebuild only the short
            # segment hidden by the removed glyphs; existing pixels elsewhere
            # remain untouched.
            cv2.line(after,(99,3266),(245,3266),(0,0,0),3)
            cv2.line(after,(100,3266),(100,3272),(0,0,0),3)
        if np.array_equal(after,before): raise RuntimeError(f"no change p{page:03d}")
        cv2.imwrite(str(clean_path),after)
        ops.append({
            "page_index":page,"bbox":list(bbox),"reason":reason,"operation":"targeted_lama_mask",
            "mask_pixels":int(np.count_nonzero(mask)),"before_sha256_pixels":sha(before),
            "after_sha256_pixels":sha(after),"raw_sha256_pixels":sha(raw),"inpaint_metrics":inp.last_metrics(),
        })
        cards.append(focused_card(raw,before,after,page,bbox))

    proof=root/"late-clean-proof"; proof.mkdir(parents=True,exist_ok=True)
    sheet=Image.new("RGB",(1050,345*len(cards)),"white")
    for i,c in enumerate(cards): sheet.paste(c,(0,i*345))
    sheet.save(proof/"focused-late-story-sfx.jpg",quality=94)
    meta={
        "checkpoint":"03b-late-clean-repair","chapter_id":CHAPTER_ID,
        "created_at":datetime.now(timezone.utc).isoformat(),"source_ocr_run":34669709152,
        "repair_scope":"LOCAL_ONLY","raw_changed":False,"ocr_rerun_required":False,
        "touched_pages":[36,38],"operation_count":2,"operations":ops,
        "next_action":"STOP_FOR_HUMAN_LATE_CLEAN_REVIEW",
    }
    (root/"late-clean-repair.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
    (root/"HUMAN_CHECKPOINT_03B.txt").write_text(
        "CHECKPOINT: LATE CLEAN LOCAL REPAIR\nReview p036/p038 RAW vs BEFORE vs AFTER. RAW is unchanged, so approved OCR results remain valid.\n",
        encoding="utf-8")
    print(json.dumps(meta,ensure_ascii=False,indent=2))

if __name__ == "__main__": main()
