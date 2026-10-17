"""Score the old and tuned Kiuyha on the held-out checks at the app's settings (1280, conf 0.25) and draw side-by-side samples."""
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SETS = {"synth": "held-out series with drawn text", "real": "held-out series as is (labels: old model, so recall = kept)",
        "webval": "colour webtoon words", "art": "Total-Text curved and artistic words (per-word boxes)"}


def labels(path, w, h):
    rows = [list(map(float, r.split()[1:])) for r in path.read_text().splitlines() if r.strip()]
    return [((x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h) for x, y, bw, bh in rows]


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def covered(g, preds, w, h):
    """Share of a true box covered by any predicted box, which forgives a block split into lines or words."""
    mask = np.zeros((h, w), bool)
    for p in preds:
        mask[int(max(0, p[1])):int(p[3]), int(max(0, p[0])):int(p[2])] = True
    area = mask[int(max(0, g[1])):int(g[3]), int(max(0, g[0])):int(g[2])]
    return area.mean() if area.size else 0.0


def score(model, folder):
    hits = cover = found = truth = predicted = 0
    seconds, per_image = 0.0, {}
    for image_path in sorted((folder / "images").glob("*.jpg")):
        image = Image.open(image_path).convert("RGB")
        gt = labels(folder / "labels" / f"{image_path.stem}.txt", *image.size)
        start = time.perf_counter()
        preds = model.predict(image, imgsz=1280, conf=0.25, verbose=False)[0].boxes.xyxy.tolist()
        seconds += time.perf_counter() - start
        truth, predicted = truth + len(gt), predicted + len(preds)
        hits += sum(any(iou(g, p) >= 0.5 for p in preds) for g in gt)
        found += sum(any(iou(p, g) >= 0.5 for g in gt) for p in preds)
        cover += sum(covered(g, preds, *image.size) >= 0.5 for g in gt)
        per_image[image_path.name] = preds
    n = len(per_image) or 1
    return {"images": len(per_image), "boxes": truth, "recall_iou50": round(hits / max(truth, 1), 4),
            "precision_iou50": round(found / max(predicted, 1), 4), "covered_recall": round(cover / max(truth, 1), 4),
            "ms_per_image": round(1000 * seconds / n, 1)}, per_image


def samples(folder, old, new, out, count=12):
    """Old boxes red, tuned boxes green, on the images where the two differ most."""
    out.mkdir(parents=True, exist_ok=True)
    ranked = sorted(old, key=lambda k: -abs(len(old[k]) - len(new[k])))[:count]
    for name in ranked:
        image = Image.open(folder / "images" / name).convert("RGB")
        draw = ImageDraw.Draw(image)
        for b in old[name]:
            draw.rectangle(b, outline=(230, 30, 30), width=4)
        for b in new[name]:
            draw.rectangle([b[0] + 3, b[1] + 3, b[2] - 3, b[3] - 3], outline=(20, 200, 60), width=4)
        image.thumbnail((900, 900))
        image.save(out / name, quality=80)


def main(data, out, old_weights, new_weights):
    from ultralytics import YOLO
    import torch
    torch.set_num_threads(4)
    data, out = Path(data), Path(out)
    old, new = YOLO(old_weights), YOLO(new_weights)
    report = {}
    for split, about in SETS.items():
        folder = data / "images" / split
        if not folder.exists():
            continue
        staged = data / "eval" / split
        staged.mkdir(parents=True, exist_ok=True)
        for kind in ("images", "labels"):
            link = staged / kind
            if not link.exists():
                link.symlink_to((data / kind / split).resolve())
        old_score, old_boxes = score(old, staged)
        new_score, new_boxes = score(new, staged)
        report[split] = {"about": about, "old": old_score, "tuned": new_score}
        samples(staged, old_boxes, new_boxes, out / "samples" / split)
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps(report, indent=1))
        print(split, json.dumps(report[split]), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
