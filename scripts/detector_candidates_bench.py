"""Try pretrained comic detectors (YOLO26 and others) on the same real slices.

For each candidate repo on Hugging Face the model card and weights are fetched,
then every slice is run at the model's usual 1024 input (the long side shrunk to
1024, like the app today), at a rectangle with the slice's shape and the same
pixel count as 1024x1024 (same cost, no padding), and at the slice's own
resolution in one pass (YOLO models are fully convolutional, so a .pt file
accepts any multiple of 32).

Measured per variant:

  seconds      total inference time over the slices (PyTorch, CPU)
  block_recall share of reference text blocks at least half covered by the
               variant's text regions (mask when the model gives one, else box)
  stray_area   share of the variant's text area farther than 20 px from any
               reference text
  classes      how many regions of each class it found

The reference is the app's own text segmenter run in 1.25x windows, so it leans
towards that model; the per-slice overlay sheets are the real verdict.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

CANDIDATES = [
    "ShadowB/Manga109-panel-balloon-text-yolov26-segmentation",
    "Kiuyha/Manga-Bubble-YOLO",
    "leoxs22/manga-panel-detector-yolo26n",
    "ogkalu/comic-text-segmenter-yolov8m",
    "ogkalu/comic-speech-bubble-detector-yolov8m",
]
REFERENCE_SCALE = 1.25
COLORS = {"text": (0, 0, 255), "balloon": (0, 200, 0), "frame": (255, 120, 0), "other": (0, 200, 255)}


def role(name: str) -> str:
    name = name.lower()
    if "bubble" in name or "balloon" in name:
        return "balloon"
    if "text" in name:
        return "text"
    if "frame" in name or "panel" in name:
        return "frame"
    return "other"


def fetch(repo: str, cards: Path) -> Path | None:
    from huggingface_hub import hf_hub_download, list_repo_files

    try:
        files = list_repo_files(repo)
    except Exception as exc:  # noqa: BLE001 - report and move on
        print(f"{repo}: cannot list files ({exc})", flush=True)
        return None
    (cards / f"{repo.replace('/', '__')}.files.txt").write_text("\n".join(files), encoding="utf-8")
    if "README.md" in files:
        card = Path(hf_hub_download(repo, "README.md"))
        (cards / f"{repo.replace('/', '__')}.md").write_text(card.read_text(encoding="utf-8"), encoding="utf-8")
    weights = [f for f in files if f.endswith(".pt")] or [f for f in files if f.endswith(".onnx")]
    if not weights:
        print(f"{repo}: no .pt or .onnx weights in {files}", flush=True)
        return None
    weights.sort(key=lambda f: ("best" not in f.lower(), "last" in f.lower(), len(f)))
    print(f"{repo}: using {weights[0]}", flush=True)
    return Path(hf_hub_download(repo, weights[0]))


def run_model(model, image: np.ndarray, imgsz):
    started = time.perf_counter()
    result = model.predict(image, imgsz=imgsz, conf=0.2, retina_masks=True, verbose=False)[0]
    elapsed = time.perf_counter() - started
    regions = []
    names = result.names
    boxes = result.boxes
    masks = None if result.masks is None else result.masks.data.cpu().numpy()
    for i in range(0 if boxes is None else len(boxes)):
        x1, y1, x2, y2 = (int(round(v)) for v in boxes.xyxy[i].tolist())
        name = str(names[int(boxes.cls[i])])
        mask = None
        if masks is not None and i < len(masks):
            full = masks[i]
            if full.shape != image.shape[:2]:
                full = cv2.resize(full, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
            mask = full > 0.5
        regions.append({"name": name, "role": role(name), "conf": float(boxes.conf[i]),
                        "box": (x1, y1, x2, y2), "mask": mask})
    return regions, elapsed


def text_area(shape, regions) -> np.ndarray:
    area = np.zeros(shape, bool)
    for r in regions:
        if r["role"] != "text":
            continue
        if r["mask"] is not None:
            area |= r["mask"]
        else:
            x1, y1, x2, y2 = r["box"]
            area[max(0, y1):y2, max(0, x1):x2] = True
    return area


def overlay(image: np.ndarray, regions, label: str) -> np.ndarray:
    canvas = image.copy()
    tint = canvas.copy()
    for r in regions:
        color = COLORS[r["role"]]
        x1, y1, x2, y2 = r["box"]
        if r["mask"] is not None:
            tint[r["mask"]] = color
            if r["role"] != "text":
                contours, _ = cv2.findContours(r["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(canvas, contours, -1, color, 4)
        elif r["role"] == "text":
            cv2.rectangle(tint, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3)
    text_like = [r for r in regions if r["role"] == "text"]
    if text_like:
        area = text_area(image.shape[:2], text_like)
        canvas[area] = (0.55 * canvas[area] + 0.45 * tint[area]).astype(np.uint8)
    scale = 300 / canvas.shape[1]
    small = cv2.resize(canvas, (300, max(1, int(canvas.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    small = cv2.copyMakeBorder(small, 34, 0, 0, 6, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    for row, part in enumerate((label[:34], label[34:68])):
        cv2.putText(small, part, (3, 13 + row * 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    return small


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--slices", type=int, default=16)
    parser.add_argument("--skip", type=int, default=1)
    args = parser.parse_args()
    cards = args.out / "cards"
    cards.mkdir(parents=True, exist_ok=True)

    import torch
    from ultralytics import YOLO

    from app.image_io import read_image
    from app.one_shot_cleanup import OneShotTextMaskDetector
    from app.processing_pipeline_factory import build_processing_pipeline

    torch.set_num_threads(max(1, torch.get_num_threads()))
    models = {}
    for repo in CANDIDATES:
        path = fetch(repo, cards)
        if path is None:
            continue
        try:
            models[repo.split("/")[-1]] = YOLO(str(path))
        except Exception as exc:  # noqa: BLE001
            print(f"{repo}: cannot load ({exc})", flush=True)

    pipeline = build_processing_pipeline()
    manifest = pipeline.download_chapter(args.url, "c1ea0002", workers=2)
    pages = manifest["pages"][args.skip:args.skip + args.slices]
    app_detector = OneShotTextMaskDetector()

    def app_regions(image, scale):
        app_detector.tile_scale = scale
        started = time.perf_counter()
        boxes, _ = app_detector.detect(image)
        elapsed = time.perf_counter() - started
        regions = []
        for b in boxes:
            mask = np.zeros(image.shape[:2], bool)
            if b.mask is not None and b.mask.shape == (b.y2 - b.y1, b.x2 - b.x1):
                mask[b.y1:b.y2, b.x1:b.x2] = b.mask > 127
            regions.append({"name": "text_comic", "role": "text", "conf": float(b.confidence),
                            "box": (b.x1, b.y1, b.x2, b.y2), "mask": mask})
        return regions, elapsed

    totals: dict[str, dict] = {}

    def add(name, regions, elapsed, ref, ref_blocks, near_ref):
        t = totals.setdefault(name, {"seconds": 0.0, "blocks": 0, "recalled": 0, "area": 0, "stray": 0, "classes": {}})
        t["seconds"] += elapsed
        area = text_area(ref.shape, regions)
        t["area"] += int(area.sum())
        t["stray"] += int((area & ~near_ref).sum())
        for block in ref_blocks:
            t["blocks"] += 1
            if (area & block).sum() >= 0.5 * block.sum():
                t["recalled"] += 1
        for r in regions:
            t["classes"][r["name"]] = t["classes"].get(r["name"], 0) + 1

    for number, page in enumerate(pages, start=1):
        full = read_image(Path(page["original"]))
        core = page.get("stitch_core") or {}
        image = np.ascontiguousarray(full[int(core.get("core_y1", 0)):int(core.get("core_y2", full.shape[0]))])
        h, w = image.shape[:2]
        reference, _ = app_regions(image, REFERENCE_SCALE)
        ref = text_area((h, w), reference)
        count, labels = cv2.connectedComponents(cv2.dilate(ref.astype(np.uint8), np.ones((15, 15), np.uint8)))
        ref_blocks = [(labels == i) & ref for i in range(1, count)]
        ref_blocks = [b for b in ref_blocks if b.sum() >= 150]
        near_ref = cv2.dilate(ref.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))) > 0

        tiles = [overlay(image, reference, "reference: app segmenter 1.25x windows")]
        current, elapsed = app_regions(image, 0.0)
        add("app_current", current, elapsed, ref, ref_blocks, near_ref)
        tiles.append(overlay(image, current, f"app current (onnx, 1 pass) {elapsed:.1f}s"))
        native = [int(math.ceil(h / 32) * 32), int(math.ceil(w / 32) * 32)]
        # A rectangle with the slice's shape and the same pixel count as 1024x1024:
        # the same cost as today, without spending two thirds of it on padding.
        fit = min(1.0, 1024 / math.sqrt(h * w))
        samecost = [int(math.ceil(h * fit / 32) * 32), int(math.ceil(w * fit / 32) * 32)]
        for name, model in models.items():
            for tag, imgsz in (("1024", 1024), ("samecost", samecost), ("native", native)):
                try:
                    regions, elapsed = run_model(model, image, imgsz)
                except Exception as exc:  # noqa: BLE001
                    print(f"{name} {tag}: failed ({exc})", flush=True)
                    continue
                key = f"{name}@{tag}"
                add(key, regions, elapsed, ref, ref_blocks, near_ref)
                tiles.append(overlay(image, regions, f"{key} {elapsed:.1f}s"))
        height = max(t.shape[0] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, height - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                 for t in tiles]
        cv2.imwrite(str(args.out / f"slice-{number:02d}.jpg"), np.concatenate(tiles, axis=1),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"slice {number}/{len(pages)} done", flush=True)

    report = {"url": args.url, "slices": len(pages), "reference_scale": REFERENCE_SCALE,
              "models": {name: str(getattr(m, "ckpt_path", "")) for name, m in models.items()},
              "names": {name: {int(k): v for k, v in m.names.items()} for name, m in models.items()},
              "variants": {}}
    for name, t in totals.items():
        report["variants"][name] = {
            "seconds": round(t["seconds"], 1),
            "block_recall": round(t["recalled"] / max(1, t["blocks"]), 3),
            "stray_area": round(t["stray"] / max(1, t["area"]), 3),
            "classes": t["classes"],
        }
    (args.out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
