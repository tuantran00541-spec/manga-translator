"""Build the Kiuyha fine-tune set: our chapters (old boxes kept), drawn text on them, colour webtoon text, and held-out checks."""
import io
import json
import random
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import synth  # noqa: E402

HF = "https://huggingface.co/datasets"
TILE = 1024


def save(root, split, name, image, boxes):
    """Writes one image and its YOLO label file (class 0 = text)."""
    (root / "images" / split).mkdir(parents=True, exist_ok=True)
    (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(root / "images" / split / f"{name}.jpg", quality=random.randint(70, 95))
    w, h = image.size
    rows = [f"0 {(a + c) / 2 / w:.6f} {(b + d) / 2 / h:.6f} {(c - a) / w:.6f} {(d - b) / h:.6f}"
            for a, b, c, d in boxes if c - a > 2 and d - b > 2]
    (root / "labels" / split / f"{name}.txt").write_text("\n".join(rows))


def old_boxes(model, image, conf):
    """The current Kiuyha's boxes, kept as labels so the tuned model does not forget them."""
    result = model.predict(image, imgsz=1280, conf=conf, verbose=False)[0]
    return [tuple(int(v) for v in b) for b in result.boxes.xyxy.tolist()]


def tiles(image):
    """Square windows down a tall slice, skipping near-blank ones."""
    w, h = image.size
    side = min(w, TILE * 2)
    step = int(side * 0.75)
    for top in range(0, max(1, h - side // 3), step):
        crop = image.crop((0, top, w, min(h, top + side)))
        if np.asarray(crop.convert("L")).std() > 8:
            yield top, crop


def chapters(model, root, plan, stats):
    """Our chapters: each tile kept as is, plus a copy with drawn text; held-out series go to the checks."""
    from app.processing_pipeline_factory import build_processing_pipeline
    pipeline = build_processing_pipeline()
    for n, (url, split) in enumerate(plan):
        try:
            pages = pipeline.download_chapter(url, f"ky{n:03d}", workers=2)["pages"]
        except Exception as exc:  # one dead chapter must not stop the set
            print(f"skip {url}: {exc}", flush=True)
            continue
        slug = re.sub(r"[^a-z0-9]+", "-", url.split("/comics/")[-1].lower()).strip("-")
        for p, page in enumerate(pages):
            path = Path(page["original"])
            image = Image.open(path).convert("RGB")
            for top, crop in tiles(image):
                boxes = old_boxes(model, crop, 0.4)
                name = f"{slug}-{p:03d}-{top}"
                real, synth_split = ("train", "train") if split == "train" else ("real", "synth")
                save(root, real, name, crop, boxes)
                drawn, added = synth.draw_blocks(crop, boxes, random.randint(1, 4))
                save(root, synth_split, f"{name}-s", drawn, boxes + added)
                stats[real] = stats.get(real, 0) + 1
                stats[synth_split] = stats.get(synth_split, 0) + 1


def webtoon(model, root, shard, split, limit, stats):
    """Colour webtoon panels with one drawn word each and its true box (jhc90/webtoon_text_inpainting_data)."""
    import pyarrow.parquet as pq
    url = f"{HF}/jhc90/webtoon_text_inpainting_data/resolve/main/data/train-{shard:05d}-of-00050.parquet"
    local = Path(f"/tmp/webtoon-{shard}.parquet")
    if not local.exists():
        urllib.request.urlretrieve(url, local)
    table = pq.ParquetFile(local)
    count = 0
    for batch in table.iter_batches(batch_size=64, columns=["input_image", "metadata"]):
        for row in batch.to_pylist():
            if count >= limit:
                return
            image = Image.open(io.BytesIO(row["input_image"]["bytes"])).convert("RGB")
            box = tuple(row["metadata"]["bbox"])
            boxes = [b for b in old_boxes(model, image, 0.5) if synth._free(b, [box])] + [box]
            save(root, split, f"webtoon-{shard}-{count:05d}", image, boxes)
            count += 1
            stats[split] = stats.get(split, 0) + 1
    local.unlink()


def totaltext(root, stats):
    """Total-Text test photos of curved and artistic words, a check only: their boxes are per word, not per block."""
    base = f"{HF}/yunusserhat/Total-Text-Dataset/resolve/main"
    names = json.load(urllib.request.urlopen("https://huggingface.co/api/datasets/yunusserhat/Total-Text-Dataset"))["siblings"]
    for item in names:
        path = item["rfilename"]
        if not re.fullmatch(r"Images/Test/img\d+\.(jpg|JPG)", path):
            continue
        stem = path.rsplit("/", 1)[1].split(".")[0]
        try:
            gt = urllib.request.urlopen(f"{base}/txt_format/Test/poly_gt_{stem}.txt").read().decode()
            image = Image.open(io.BytesIO(urllib.request.urlopen(f"{base}/{path}").read())).convert("RGB")
        except Exception:
            continue
        boxes = []
        for line in gt.splitlines():
            xs = [int(v) for v in re.findall(r"-?\d+", (re.search(r"x: \[\[(.*?)\]\]", line) or [None, ""])[1])]
            ys = [int(v) for v in re.findall(r"-?\d+", (re.search(r"y: \[\[(.*?)\]\]", line) or [None, ""])[1])]
            if xs and ys and "#" not in (re.search(r"transcriptions: \[u?'(.*?)'\]", line) or [None, "#"])[1]:
                boxes.append((min(xs), min(ys), max(xs), max(ys)))
        if boxes:
            save(root, "art", f"tt-{stem}", image, boxes)
            stats["art"] = stats.get("art", 0) + 1


def main(weights, out, plan_file):
    from ultralytics import YOLO
    random.seed(0)
    root, stats = Path(out), {}
    model = YOLO(weights)
    plan = json.loads(Path(plan_file).read_text())
    chapters(model, root, [(c["url"], c["split"]) for c in plan["chapters"]], stats)
    for shard, split, limit in plan["webtoon"]:
        webtoon(model, root, shard, split, limit, stats)
    totaltext(root, stats)
    (root / "data.yaml").write_text(f"path: {root.resolve()}\ntrain: images/train\nval: images/synth\nnames:\n  0: text\n")
    (root / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
