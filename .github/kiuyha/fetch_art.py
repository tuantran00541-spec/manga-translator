"""Fetch the Total-Text TRAIN split (curved / artistic words) and append it to an
existing kiuyha-data directory as extra training data.

Polygons are converted to axis-aligned YOLO boxes (class 0 = text), the same
label format build.py writes. Never touches the Test split: that is the
held-out art check used by evaluate.py.

Usage:
    python .github/kiuyha/fetch_art.py kiuyha-data [limit]

Needs Pillow (present once ultralytics is installed).
"""
import io
import json
import random
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

BASE = "https://huggingface.co/datasets/yunusserhat/Total-Text-Dataset/resolve/main"
API = "https://huggingface.co/api/datasets/yunusserhat/Total-Text-Dataset"
PREFIX = "tt-train"
WORKERS = 8


def fetch(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def parse_boxes(gt_text):
    """poly_gt lines -> (xmin, ymin, xmax, ymax); skips '#' don't-care regions."""
    boxes = []
    for line in gt_text.splitlines():
        xs = [int(v) for v in re.findall(r"-?\d+", (re.search(r"x: \[\[(.*?)\]\]", line) or [None, ""])[1])]
        ys = [int(v) for v in re.findall(r"-?\d+", (re.search(r"y: \[\[(.*?)\]\]", line) or [None, ""])[1])]
        trans = (re.search(r"transcriptions: \[u?'(.*?)'\]", line) or [None, "#"])[1]
        if xs and ys and "#" not in trans:
            boxes.append((min(xs), min(ys), max(xs), max(ys)))
    return boxes


def save(root, name, image, boxes):
    (root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(root / "images" / "train" / f"{name}.jpg", quality=random.randint(70, 95))
    w, h = image.size
    rows = [f"0 {(a + c) / 2 / w:.6f} {(b + d) / 2 / h:.6f} {(c - a) / w:.6f} {(d - b) / h:.6f}"
            for a, b, c, d in boxes if c - a > 2 and d - b > 2]
    (root / "labels" / "train" / f"{name}.txt").write_text("\n".join(rows))


def one(root, stem):
    try:
        raw = fetch(f"{BASE}/Images/Train/{stem}.jpg")
    except Exception:
        try:
            raw = fetch(f"{BASE}/Images/Train/{stem}.JPG")
        except Exception as exc:
            print(f"skip {stem}: image: {exc}", flush=True)
            return 0
    try:
        gt = fetch(f"{BASE}/txt_format/Train/poly_gt_{stem}.txt").decode()
    except Exception as exc:
        print(f"skip {stem}: gt: {exc}", flush=True)
        return 0
    boxes = parse_boxes(gt)
    if not boxes:
        return 0
    save(root, f"{PREFIX}-{stem}", Image.open(io.BytesIO(raw)).convert("RGB"), boxes)
    return 1


def main(out, limit=None):
    random.seed(0)
    root = Path(out)
    names = json.load(urllib.request.urlopen(API, timeout=60))["siblings"]
    stems = sorted({m.group(1) for i in names
                    if (m := re.fullmatch(r"Images/Train/(img\d+)\.(jpg|JPG)", i["rfilename"]))})
    if limit:
        stems = stems[:int(limit)]
    print(f"fetching {len(stems)} Total-Text train images", flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for n, got in enumerate(ex.map(lambda s: one(root, s), stems), 1):
            done += got
            if n % 100 == 0:
                print(f"{n}/{len(stems)} fetched, {done} kept", flush=True)
    stats_path = root / "stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        stats["train"] = stats.get("train", 0) + done
        stats["train_art_totaltext"] = done
        stats_path.write_text(json.dumps(stats, indent=1))
    print(json.dumps({"art_train_added": done}), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
