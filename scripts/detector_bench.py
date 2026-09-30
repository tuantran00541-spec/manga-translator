"""Run the text detectors of other manga translators on the same slices and compare what each finds."""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
import types
from pathlib import Path

import cv2
import numpy as np

CHAPTERS = {
    "shadow-slave": "https://asurascans.com/comics/shadow-slave-05c7df14/chapter/1",
    "hero-cannot-rest": "https://asurascans.com/comics/the-hero-cannot-rest-3ec3b16f/chapter/1",
    "weapon-replicator": "https://asurascans.com/comics/the-academy-s-weapon-replicator-05c7df14/chapter/1",
    "mangadex": "https://mangadex.org/chapter/ef7c4c66-97da-460a-9409-2231a950c877/1",
}
# One vote per project; the other entries are variants reported alongside.
FAMILIES = {"ours": "ours", "comic-translate": "comic-translate", "ballons-ctd": "ctd", "koharu": "koharu"}
FOUND = 0.25  # share of a region a detector must cover to count as finding it


def _index(slices: Path) -> list[dict]:
    return json.loads((slices / "index.json").read_text())


def _save(out: Path, name: str, family: str, images: dict, seconds: float, status: str = "ok") -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"detector": name, "family": family, "status": status,
                               "seconds": round(seconds, 1), "images": images}))
    found = sum(len(v) for v in images.values())
    print(f"{name}: {found} boxes on {len(images)} slices in {seconds:.0f} s ({status})", flush=True)


def _run(slices: Path, out: Path, name: str, family: str, detect) -> None:
    images, started = {}, time.perf_counter()
    try:
        for entry in _index(slices):
            image = cv2.imread(str(slices / entry["file"]))
            images[entry["file"]] = [[int(b[0]), int(b[1]), int(b[2]), int(b[3]), round(float(b[4]), 3), b[5]]
                                     for b in detect(image)]
        _save(out, name, family, images, time.perf_counter() - started)
    except Exception as exc:  # a detector that cannot run is reported, not fatal
        _save(out, name, family, images, time.perf_counter() - started, f"error: {type(exc).__name__}: {exc}"[:400])
        raise


def cmd_slices(args) -> None:
    from app.manifest_utils import load_manifest_raw
    from app.processing_pipeline_factory import build_processing_pipeline

    pipeline = build_processing_pipeline()
    entries = []
    for number, (key, url) in enumerate(CHAPTERS.items()):
        chapter_id = f"b{number:07d}"
        try:
            pipeline.download_chapter(url, chapter_id, workers=2)
        except Exception as exc:
            print(f"{key}: download failed ({exc})", flush=True)
            continue
        for index, page in enumerate(load_manifest_raw(chapter_id)["pages"]):
            image = cv2.imread(str(page["original"]))
            if image is None:
                continue
            name = f"{key}/{index:03d}.jpg"
            (args.out / key).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.out / name), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            entries.append({"file": name, "chapter": key, "w": image.shape[1], "h": image.shape[0]})
        print(f"{key}: {sum(e['chapter'] == key for e in entries)} slices", flush=True)
    (args.out / "index.json").write_text(json.dumps(entries))


def cmd_ours(args) -> None:
    from app.config import KIUYHA_TEXT_MODEL
    from app.detector.kiuyha_detector import KiuyhaTextDetector

    detector = KiuyhaTextDetector(KIUYHA_TEXT_MODEL)
    _run(args.slices, args.out / "ours.json", "ours", "ours",
         lambda image: [(b.x1, b.y1, b.x2, b.y2, b.confidence, "text") for b in detector.text_boxes(image)])


def cmd_kiuyha_pt(args) -> None:
    from huggingface_hub import hf_hub_download, list_repo_files
    from ultralytics import YOLO

    for weights in sorted(f for f in list_repo_files("Kiuyha/Manga-Bubble-YOLO") if f.endswith(".pt")):
        model = YOLO(hf_hub_download("Kiuyha/Manga-Bubble-YOLO", weights))
        names = model.names

        def detect(image, model=model, names=names):
            result = model.predict(image, imgsz=1280, conf=0.25, verbose=False)[0]
            return [(*box, score, "text") for box, score, cls in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(),
                                                                     result.boxes.cls.tolist())
                    if "text" in str(names[int(cls)]).lower()]

        stem = Path(weights).stem.replace("/", "-")
        _run(args.slices, args.out / f"kiuyha-pt-{stem}.json", f"kiuyha-pt:{weights}", "ours", detect)


def cmd_rtdetr(args) -> None:
    import onnxruntime as ort
    from PIL import Image

    sys.path.insert(0, str(args.repo))
    from modules.detection.utils.slicer import ImageSlicer

    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    slicer = ImageSlicer(height_to_width_ratio_threshold=3.5, target_slice_ratio=3.0,
                         overlap_height_ratio=0.2, min_slice_height_ratio=0.7)
    kept: list = []

    def single(rgb):
        # As modules/detection/rtdetr_v2_onnx.py: a 640 square, labels 0 bubble, 1 and 2 text.
        pil = Image.fromarray(rgb)
        arr = np.transpose(np.asarray(pil.resize((640, 640)), dtype=np.float32) / 255.0, (2, 0, 1))[None]
        labels, boxes, scores = session.run(None, {"images": arr, "orig_target_sizes": np.array([[pil.width, pil.height]],
                                                                                                  dtype=np.int64)})[:3]
        bubbles, texts = [], []
        for label, box, score in zip(labels.reshape(-1), boxes.reshape(-1, 4), scores.reshape(-1)):
            if score < 0.3:
                continue
            (bubbles if int(label) == 0 else texts).append([int(v) for v in box])
            if int(label) != 0:
                kept.append(float(score))
        return (np.array(bubbles) if bubbles else np.array([]), np.array(texts) if texts else np.array([]))

    def detect(image):
        _, texts = slicer.process_slices_for_detection(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), single)
        return [(*box, 1.0, "text") for box in np.asarray(texts).reshape(-1, 4).tolist()]

    _run(args.slices, args.out / f"{args.name}.json", args.name, "comic-translate", detect)


def cmd_ctd(args) -> None:
    # Import the detector without ballontranslator.modules/__init__, which pulls in the Qt app.
    root = Path(args.repo) / "ballontranslator"
    sys.path.insert(0, str(args.repo))
    for package in ("ballontranslator.modules", "ballontranslator.modules.textdetector"):
        stub = types.ModuleType(package)
        stub.__path__ = [str(root / Path(*package.split(".")[1:]))]
        sys.modules[package] = stub
    inference = importlib.import_module("ballontranslator.modules.textdetector.ctd.inference")
    model = inference.TextDetector(str(args.model), detect_size=1024, device="cpu")

    def detect(image):
        blocks = model(image)[2]
        return [(*block.xyxy, 1.0, "text") for block in blocks]

    _run(args.slices, args.out / "ballons-ctd.json", "ballons-ctd", "ctd", detect)


def cmd_koharu(args) -> None:
    images, seconds = {}, 0.0
    for entry in _index(args.slices):
        path = args.json_dir / (entry["file"].replace("/", "__") + ".json")
        if not path.is_file():
            continue
        data = json.loads(path.read_text())
        seconds += float(data.get("seconds", 0))
        images[entry["file"]] = [[*map(int, d["bbox"]), round(float(d["score"]), 3),
                                  "sfx" if "onomatopoeia" in d["label"] else "text"]
                                 for d in data["detections"] if d["label"] in ("text", "onomatopoeia")]
    _save(args.out / "koharu.json", "koharu", "koharu", images, seconds,
          "ok" if images else "error: no koharu output")


def _mask(shape, boxes) -> np.ndarray:
    mask = np.zeros(shape, bool)
    for x1, y1, x2, y2, *_ in boxes:
        mask[max(0, y1):max(0, y2), max(0, x1):max(0, x2)] = True
    return mask


def _crop(image, box, finders, path: Path) -> None:
    x1, y1, x2, y2 = box
    h, w = image.shape[:2]
    pad = 40
    tile = image[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)].copy()
    cv2.rectangle(tile, (min(pad, x1), min(pad, y1)), (min(pad, x1) + x2 - x1, min(pad, y1) + y2 - y1), (0, 0, 255), 2)
    scale = min(1.0, 480 / max(tile.shape[:2]))
    tile = cv2.resize(tile, (max(1, int(tile.shape[1] * scale)), max(1, int(tile.shape[0] * scale))))
    tile = cv2.copyMakeBorder(tile, 24, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(tile, ", ".join(finders) or "-", (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.imwrite(str(path), tile, [cv2.IMWRITE_JPEG_QUALITY, 85])


def cmd_report(args) -> None:
    runs = {}
    for path in sorted(args.boxes.glob("*.json")):
        data = json.loads(path.read_text())
        runs[data["detector"]] = data
    voters = {data["family"]: name for name, data in runs.items()
              if name in FAMILIES and data["status"] == "ok"}
    args.out.mkdir(parents=True, exist_ok=True)
    stats = {name: {"family": d["family"], "status": d["status"], "seconds": d["seconds"], "found": 0,
                    "judged": 0, "hit": 0, "solo": 0} for name, d in runs.items()}
    rows = []
    for entry in _index(args.slices):
        image = cv2.imread(str(args.slices / entry["file"]))
        shape = image.shape[:2]
        masks = {name: _mask(shape, [b for b in d["images"].get(entry["file"], []) if b[5] == "text"])
                 for name, d in runs.items() if entry["file"] in d["images"]}
        union = np.zeros(shape, bool)
        for family, name in voters.items():
            union |= masks.get(name, np.zeros(shape, bool))
        count, labels, boxes, _ = cv2.connectedComponentsWithStats(
            cv2.dilate(union.astype(np.uint8), np.ones((9, 9), np.uint8)))
        for i in range(1, count):
            region = (labels == i) & union
            area = int(region.sum())
            if area < 200:
                continue
            found = {name for name, mask in masks.items() if (mask & region).sum() >= FOUND * area}
            families = {runs[name]["family"] for name in found if name in FAMILIES}
            x, y, w, h = boxes[i][:4]
            row = {"file": entry["file"], "chapter": entry["chapter"], "box": [int(x), int(y), int(x + w), int(y + h)],
                   "found": sorted(found), "families": sorted(families)}
            rows.append(row)
            for name in masks:
                stat = stats[name]
                stat["found"] += name in found
                others = families - {runs[name]["family"]}
                if len(others) >= 2:  # most of the other projects agree it is text
                    stat["judged"] += 1
                    stat["hit"] += name in found
                if name in found and not others:
                    stat["solo"] += 1
    for stat in stats.values():
        stat["recall_vs_others"] = round(stat["hit"] / stat["judged"], 3) if stat["judged"] else None
    summary = {"chapters": sorted({e["chapter"] for e in _index(args.slices)}), "slices": len(_index(args.slices)),
               "voters": voters, "regions": len(rows), "detectors": stats,
               "by_chapter": {}}
    for chapter in summary["chapters"]:
        part = [r for r in rows if r["chapter"] == chapter]
        summary["by_chapter"][chapter] = {
            name: {"found": sum(name in r["found"] for r in part),
                   "judged": sum(len(set(r["families"]) - {runs[name]["family"]}) >= 2 for r in part),
                   "hit": sum(name in r["found"] and len(set(r["families"]) - {runs[name]["family"]}) >= 2 for r in part)}
            for name in runs}
    # Crops: what ours missed that another project found, and what only one project found.
    ours = voters.get("ours")
    groups = {"missed-by-ours": [r for r in rows if ours and ours not in r["found"] and r["families"]]}
    for family, name in voters.items():
        groups[f"only-{family}"] = [r for r in rows if r["families"] == [family]]
    for group, members in groups.items():
        summary[f"{group}_count"] = len(members)
        folder = args.out / group
        folder.mkdir(exist_ok=True)
        step = max(1, len(members) // args.crops)
        for n, row in enumerate(members[::step][:args.crops]):
            image = cv2.imread(str(args.slices / row["file"]))
            _crop(image, row["box"], row["families"], folder / f"{n:03d}-{row['file'].replace('/', '-')[:-4]}.jpg")
    (args.out / "regions.json").write_text(json.dumps(rows))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "by_chapter"}, indent=1))


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("slices")
    p.add_argument("--out", type=Path, required=True)
    for name in ("ours", "kiuyha-pt", "rtdetr", "ctd", "koharu", "report"):
        p = sub.add_parser(name)
        p.add_argument("--slices", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--repo", type=Path)
        p.add_argument("--model", type=Path)
        p.add_argument("--name", default="comic-translate")
        p.add_argument("--json-dir", type=Path)
        p.add_argument("--boxes", type=Path)
        p.add_argument("--crops", type=int, default=40)
    args = parser.parse_args()
    {"slices": cmd_slices, "ours": cmd_ours, "kiuyha-pt": cmd_kiuyha_pt, "rtdetr": cmd_rtdetr, "ctd": cmd_ctd,
     "koharu": cmd_koharu, "report": cmd_report}[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
