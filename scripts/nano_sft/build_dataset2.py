#!/usr/bin/env python3
"""Build EN->VI SFT pairs for one Nano Machine chapter (v2).

VI images: ~/workspace/nano-vi/chNNN/*.webp (nettruyen15s)
EN images: ~/workspace/nano-machine-full/chapters/nano-machine-chNNN.zip
v2 fixes (v1 verdict: naive sequence pairing produces garbage):
- No tight geometry filter: OCR every box, filter on TEXT quality
  (letter ratio, min length, ad denylist) instead.
- Alignment by VISUAL page matching: each VI image is matched to its EN
  strip + y-offset with cv2.matchTemplate (downscaled grayscale, common
  width, NCC). Boxes pair within the matched slice by y-order; unpaired
  boxes are flagged, never forced. Promo pages match nothing -> skipped.
Usage: <venv python> build_dataset2.py <chapter> [--limit N] [--out DIR]
"""
import sys, os, json, re, subprocess, tempfile, zipfile, argparse
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
os.environ.setdefault("MANGA_PROCESSING_PIPELINE", "fast")
os.environ.setdefault("MANGA_ORT_PROVIDER", "openvino")
from PIL import Image
import numpy as np
import cv2
from app.detector.kiuyha_detector import KiuyhaTextDetector

def _find_model():
    for cand in [os.environ.get("KIUYHA_MODEL", ""),
                 os.path.join(REPO_ROOT, "models", "kiuyha-tuned-1023.onnx"),
                 "/home/hatch/workspace/manga-main/models/kiuyha-tuned-1023.onnx"]:
        if cand and os.path.exists(cand):
            return cand
    return os.path.join(REPO_ROOT, "models", "kiuyha-tuned-1023.onnx")

MODEL = _find_model()
VI_DIR = os.environ.get("NANO_VI_DIR", "/home/hatch/workspace/nano-vi")
EN_ZIPS = os.environ.get("NANO_EN_ZIPS", "/home/hatch/workspace/nano-machine-full/chapters")
AD_RE = re.compile(r"nettruyen|vitruyen|quảng cáo|theo dõi|link dự phòng|facebook|fb\.com|asurascans|asura|discord|patreon|\.com|support us|join us|đọc tiếp|chap mới|xem truyện", re.I)
MATCH_W = 360
MATCH_SCALE = 4

def garbage(text):
    if re.search(r"\d{2,}", text):
        return True
    if any(c in text for c in "¬`~|<>^_¿¡«»§¶†‡°"):
        return True
    digits = sum(1 for c in text if c.isdigit())
    if digits / max(1, len(text)) > 0.15:
        return True
    for w in re.findall(r"[A-Za-zÀ-ỹ]+", text):
        if len(w) < 3:
            continue
        transitions = sum(1 for a, b in zip(w, w[1:])
                          if a.isupper() != b.isupper())
        if transitions >= 2:
            return True
    return False

def letter_ratio(t):
    if not t:
        return 0.0
    letters = sum(1 for c in t if c.isalpha())
    return letters / max(1, len(t))

def ocr_text(crop_path, lang):
    r = subprocess.run(["tesseract", crop_path, "stdout", "-l", lang, "--psm", "6"],
                       capture_output=True, text=True, timeout=120)
    return " ".join(r.stdout.split())

def read_side(img_paths, lang, det):
    """Per image: list of {text, box} after OCR quality filter."""
    per_img = []
    with tempfile.TemporaryDirectory() as td:
        for ii, p in enumerate(img_paths):
            img = Image.open(p).convert("RGB")
            boxes = det.detect_slice(np.array(img))
            items = []
            for j, b in enumerate(boxes):
                x1, y1, x2, y2 = [int(v) for v in b[:4]]
                if (x2 - x1) < 20 or (y2 - y1) < 12:
                    continue
                cp = os.path.join(td, f"c{ii}_{j}.png")
                img.crop((x1, y1, x2, y2)).save(cp)
                try:
                    text = ocr_text(cp, lang)
                except Exception:
                    continue
                if len(text) < 3 or letter_ratio(text) < 0.5:
                    continue
                if AD_RE.search(text) or garbage(text):
                    continue
                items.append({"text": text, "box": [x1, y1, x2, y2]})
            items.sort(key=lambda it: ((it["box"][1] + it["box"][3]) / 2,
                                       (it["box"][0] + it["box"][2]) / 2))
            per_img.append(items)
    return per_img

def gray_small(path, width=MATCH_W):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    h, w = img.shape
    nw = width
    nh = max(1, int(h * width / w))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

def match_pages(vi_paths, en_paths):
    """Return {vi_idx: (en_idx, y0_in_en_fullres, score)}."""
    en_small = [gray_small(p) for p in en_paths]
    en_sizes = []
    for p in en_paths:
        im = Image.open(p)
        en_sizes.append(im.size)
    out = {}
    for vi_i, vp in enumerate(vi_paths):
        vt = gray_small(vp)
        if vt is None:
            continue
        best = (None, None, -1.0)
        vh, vw = vt.shape
        for en_i, et in enumerate(en_small):
            if et is None or et.shape[0] < vh or et.shape[1] < vw:
                continue
            res = cv2.matchTemplate(et, vt, cv2.TM_CCOEFF_NORMED)
            _, mx, _, ml = cv2.minMaxLoc(res)
            if mx > best[2]:
                ew, eh = en_sizes[en_i]
                scale = ew / MATCH_W
                best = (en_i, int(ml[1] * scale), mx)
        if best[0] is not None and best[2] >= 0.5:
            out[vi_i] = best
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chapter")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="/home/hatch/workspace/nano-vi/pairs")
    args = ap.parse_args()
    ch = args.chapter
    vi_paths = sorted(os.path.join(VI_DIR, f"ch{ch}", f)
                      for f in os.listdir(os.path.join(VI_DIR, f"ch{ch}"))
                      if f.endswith((".webp", ".jpg", ".png")))
    zf = zipfile.ZipFile(os.path.join(EN_ZIPS, f"nano-machine-ch{int(ch):03d}.zip"))
    tmpd = tempfile.mkdtemp(prefix=f"en-ch{ch}-")
    zf.extractall(tmpd)
    en_paths = sorted(os.path.join(tmpd, f) for f in os.listdir(tmpd)
                      if f.lower().endswith((".webp", ".jpg", ".png")))
    if args.limit:
        vi_paths, en_paths = vi_paths[:args.limit], en_paths[:args.limit]
    print(f"ch{ch}: VI {len(vi_paths)} imgs, EN {len(en_paths)} imgs", flush=True)
    det = KiuyhaTextDetector(MODEL)
    vi = read_side(vi_paths, "vie", det)
    print(f"ch{ch}: VI texts per img: {[len(x) for x in vi]}", flush=True)
    en = read_side(en_paths, "eng", det)
    print(f"ch{ch}: EN texts per img: {[len(x) for x in en]}", flush=True)
    matches = match_pages(vi_paths, en_paths)
    print(f"ch{ch}: page matches: {len(matches)}/{len(vi_paths)}", flush=True)
    for vi_i, (en_i, y0, sc) in sorted(matches.items()):
        print(f"  VI {os.path.basename(vi_paths[vi_i])} -> EN {os.path.basename(en_paths[en_i])} y0={y0} score={sc:.2f}", flush=True)
    pairs = []
    unpaired = 0
    vi_sizes = [Image.open(p).size for p in vi_paths]
    for vi_i, (en_i, y0, sc) in sorted(matches.items()):
        vw, vh = vi_sizes[vi_i]
        vi_items = vi[vi_i]
        en_items = [it for it in en[en_i]
                    if y0 <= (it["box"][1] + it["box"][3]) / 2 <= y0 + vh]
        used = set()
        for vi_it in vi_items:
            vc = (vi_it["box"][1] + vi_it["box"][3]) / 2 + y0
            best_j, best_d = None, 80.0
            for j, en_it in enumerate(en_items):
                if j in used:
                    continue
                ec = (en_it["box"][1] + en_it["box"][3]) / 2
                d = abs(ec - vc)
                if d < best_d:
                    best_j, best_d = j, d
            if best_j is None:
                unpaired += 1
                continue
            used.add(best_j)
            en_it = en_items[best_j]
            pairs.append({"en": en_it["text"], "vi": vi_it["text"],
                          "en_box": en_it["box"], "vi_box": vi_it["box"],
                          "en_img": os.path.basename(en_paths[en_i]),
                          "vi_img": os.path.basename(vi_paths[vi_i]),
                          "match_score": round(sc, 2)})
        unpaired += len(en_items) - len(used)
    os.makedirs(args.out, exist_ok=True)
    crops_dir = os.path.join(args.out, "crops", ch)
    os.makedirs(crops_dir, exist_ok=True)
    for p in pairs:
        ep = os.path.join(tmpd, p["en_img"])
        im = Image.open(ep).convert("RGB")
        x1, y1, x2, y2 = p["en_box"]
        cname = f"{ch}_{os.path.splitext(p['en_img'])[0]}_{x1}_{y1}.png"
        cpath = os.path.join(crops_dir, cname)
        im.crop((x1, y1, x2, y2)).save(cpath)
        p["en_crop"] = cpath
    outp = os.path.join(args.out, f"ch{ch}.jsonl")
    with open(outp, "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out, f"ch{ch}-debug.json"), "w") as f:
        json.dump({"vi": vi, "en": en,
                   "matches": {str(k): v for k, v in matches.items()}},
                  f, ensure_ascii=False)
    print(f"ch{ch}: PAIRS {len(pairs)} unpaired_boxes {unpaired} -> {outp}", flush=True)
    for p in pairs[:10]:
        print(f"  EN: {p['en']}\n  VI: {p['vi']}", flush=True)

if __name__ == "__main__":
    main()
