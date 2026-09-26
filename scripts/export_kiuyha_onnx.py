"""Export Kiuyha/Manga-Bubble-YOLO to ONNX and check its boxes against the .pt model."""
from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np

KIUYHA_REPO = "Kiuyha/Manga-Bubble-YOLO"


def iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def matched(reference, candidate) -> int:
    return sum(1 for r in reference if any(iou(r, c) >= 0.9 for c in candidate))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--slices", type=int, default=16)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import onnxruntime as ort
    import torch
    from ultralytics import YOLO

    from app.detector.kiuyha_detector import KiuyhaTextDetector
    from app.image_io import read_image
    from app.processing_pipeline_factory import build_processing_pipeline

    from huggingface_hub import hf_hub_download, list_repo_files

    torch.set_num_threads(2)
    weights = sorted((f for f in list_repo_files(KIUYHA_REPO) if f.endswith(".pt")),
                     key=lambda f: ("best" not in f.lower(), len(f)))
    pt_path = Path(hf_hub_download(KIUYHA_REPO, weights[0]))
    pt_model = YOLO(str(pt_path))
    models = Path("models")
    models.mkdir(exist_ok=True)
    targets = {"kiuyha_text.onnx": {"dynamic": True}, "kiuyha_text_1280.onnx": {"dynamic": False}}
    report = {"source": str(pt_path.name), "exports": {}}
    for name, options in targets.items():
        exported = YOLO(str(pt_path)).export(format="onnx", imgsz=1280, opset=17, **options)
        shutil.copy(exported, models / name)
        session = ort.InferenceSession(str(models / name), providers=["CPUExecutionProvider"])
        report["exports"][name] = {
            "bytes": (models / name).stat().st_size,
            "inputs": [[i.name, list(i.shape)] for i in session.get_inputs()],
            "outputs": [[o.name, list(o.shape)] for o in session.get_outputs()],
        }
    print(json.dumps(report, indent=1), flush=True)

    manifest = build_processing_pipeline().download_chapter(args.url, "c1ea0004", workers=2)
    pages = manifest["pages"][1:1 + args.slices]
    detectors = {name: KiuyhaTextDetector(models / name) for name in targets}
    totals = {"pt_native": 0.0, "pt_1280": 0.0, "kiuyha_text.onnx": 0.0, "kiuyha_text_1280.onnx": 0.0}
    counts = {"pt_native": 0, "pt_1280": 0, "native_matched": 0, "native_onnx": 0, "static_matched": 0, "static_onnx": 0}
    for page in pages:
        full = read_image(Path(page["original"]))
        core = page.get("stitch_core") or {}
        image = np.ascontiguousarray(full[int(core.get("core_y1", 0)):int(core.get("core_y2", full.shape[0]))])
        h, w = image.shape[:2]
        runs = {}
        for key, imgsz in (("pt_native", [math.ceil(h / 32) * 32, math.ceil(w / 32) * 32]), ("pt_1280", 1280)):
            started = time.perf_counter()
            result = pt_model.predict(image, imgsz=imgsz, conf=0.25, verbose=False)[0]
            totals[key] += time.perf_counter() - started
            runs[key] = [tuple(v) for v in result.boxes.xyxy.tolist()]
        for name, detector in detectors.items():
            started = time.perf_counter()
            runs[name] = [b[:4] for b in detector.detect(image)]
            totals[name] += time.perf_counter() - started
        counts["pt_native"] += len(runs["pt_native"])
        counts["pt_1280"] += len(runs["pt_1280"])
        counts["native_onnx"] += len(runs["kiuyha_text.onnx"])
        counts["static_onnx"] += len(runs["kiuyha_text_1280.onnx"])
        counts["native_matched"] += matched(runs["pt_native"], runs["kiuyha_text.onnx"])
        counts["static_matched"] += matched(runs["pt_1280"], runs["kiuyha_text_1280.onnx"])
    report["parity"] = counts
    report["seconds"] = {k: round(v, 2) for k, v in totals.items()}
    report["slices"] = len(pages)
    (args.out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
