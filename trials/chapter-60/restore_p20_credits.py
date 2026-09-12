from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CHAPTER_ID = "f1cd0121"
PAGE = 20
BBOX = (245, 1515, 655, 1675)


def sha(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("root", type=Path); args = ap.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / "processed" / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("chapter_id") != CHAPTER_ID:
        raise SystemExit("chapter mismatch")
    page = manifest["pages"][PAGE]
    raw_path = root / "raw" / "sliced" / Path(str(page["original"])).name
    clean_path = root / "processed" / Path(str(page["clean"])).name
    raw = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
    before = cv2.imread(str(clean_path), cv2.IMREAD_COLOR)
    if raw is None or before is None or raw.shape != before.shape:
        raise RuntimeError("invalid p020 RAW/CLEAN")
    x1, y1, x2, y2 = BBOX
    after = before.copy()
    after[y1:y2, x1:x2] = raw[y1:y2, x1:x2]
    cv2.imwrite(str(clean_path), after)

    def crop(arr: np.ndarray) -> Image.Image:
        pad = 60
        xx1=max(0,x1-pad); yy1=max(0,y1-pad); xx2=min(arr.shape[1],x2+pad); yy2=min(arr.shape[0],y2+pad)
        im=Image.fromarray(cv2.cvtColor(arr[yy1:yy2,xx1:xx2],cv2.COLOR_BGR2RGB))
        im.thumbnail((410,260),Image.Resampling.LANCZOS)
        return im
    ims=[crop(raw),crop(before),crop(after)]
    card=Image.new("RGB",(1260,310),"white")
    for i,im in enumerate(ims): card.paste(im,(i*420+(410-im.width)//2,25+(260-im.height)//2))
    d=ImageDraw.Draw(card); f=ImageFont.load_default()
    d.text((5,5),"p020 production-credit RAW pixel restore",fill="black",font=f)
    for x,label in ((175,"RAW"),(590,"BEFORE"),(1010,"AFTER")): d.text((x,290),label,fill="black",font=f)
    proof=root/"late-clean-proof-03d"; proof.mkdir(parents=True,exist_ok=True)
    card.save(proof/"p020-credit-restore.jpg",quality=96)

    outside = np.ones(before.shape[:2], dtype=bool)
    outside[y1:y2,x1:x2] = False
    outside_unchanged = bool(np.array_equal(before[outside], after[outside]))
    restored_exact = bool(np.array_equal(raw[y1:y2,x1:x2], after[y1:y2,x1:x2]))
    meta={
        "checkpoint":"03d-credit-restore","chapter_id":CHAPTER_ID,"created_at":datetime.now(timezone.utc).isoformat(),
        "repair_scope":"ONE_RAW_PIXEL_RESTORE","raw_changed":False,"ocr_rerun_required":False,
        "page_index":PAGE,"bbox":list(BBOX),"reason":"restore ADAPTATION MISO / ORIGINAL MIRIP / ILLUSTRATION BYEOLBOM production credits removed by CLEAN",
        "restored_bbox_pixel_exact_with_raw":restored_exact,"outside_bbox_pixel_exact_with_03c":outside_unchanged,
        "before_sha256_pixels":sha(before),"after_sha256_pixels":sha(after),"raw_sha256_pixels":sha(raw),
        "next_action":"STOP_FOR_HUMAN_03D_CLEAN_REVIEW",
    }
    if not restored_exact or not outside_unchanged:
        raise RuntimeError(meta)
    (root/"late-clean-repair-03d.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
    (root/"HUMAN_CHECKPOINT_03D.txt").write_text(
        "CHECKPOINT: 03D CREDIT RESTORE\nReview p020 RAW vs BEFORE vs AFTER. Only the production-credit bbox was restored pixel-exact from RAW.\n",
        encoding="utf-8")
    print(json.dumps(meta,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
