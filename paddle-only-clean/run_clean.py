#!/usr/bin/env python3
"""PaddleOCR-v6-only clean test on Nano ch337 (93 slices).

Feeds the reused Paddle det boxes (kiuyha-vs-paddle-ch337/paddle_boxes.json)
into the app's own Inpainter — no Kiuyha, no other detector.
Usage: run_clean.py [--smoke N | --all] [--out DIR]
"""
import argparse
import json
import os
import sys
import time

import cv2

sys.path.insert(0, os.environ.get("APP_WT", "/home/hatch/workspace/paddle-clean-wt"))

BOXES = "/home/hatch/workspace/nano-vi/kiuyha-vs-paddle-ch337/paddle_boxes.json"
DEFAULT_OUT = "/home/hatch/workspace/nano-vi/paddle-only-clean/out"


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke", type=int, help="clean first N slices only")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="skip slices whose output is already a valid PNG")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    from app.detector.boxes import BubbleBox
    from app.inpaint.lama_inpainter import Inpainter

    with open(BOXES, encoding="utf-8") as f:
        data = json.load(f)
    keys = sorted(data, key=lambda k: int(k))
    if args.smoke:
        keys = keys[: args.smoke]
    os.makedirs(args.out, exist_ok=True)

    inpainter = Inpainter()
    t_all = time.time()
    n_boxes_total = 0
    for i, key in enumerate(keys):
        entry = data[key]
        src = entry["original"]
        img = cv2.imread(src)
        if img is None:
            print(f"[{i}] slice {key}: MISSING {src}", flush=True)
            continue
        boxes = [
            BubbleBox(
                x1=int(round(b["x1"])), y1=int(round(b["y1"])),
                x2=int(round(b["x2"])), y2=int(round(b["y2"])),
                # MANUAL_CONFIDENCE_SENTINEL (1.0): the app's own manual-box
                # path. Paddle boxes carry no segmentation mask, so without
                # this build_mask() skips them as "non-authorized destructive
                # mask" (verified on smoke run 1: 0.0s/slice, nothing erased).
                # Kiuyha boxes get in via their text_segmenter ink mask;
                # Paddle has no such mask, so the manual rectangle-fallback
                # mask (rect + ink repair + adaptive dilate + paper trim)
                # is the app's only mask path for them. No app code changed.
                confidence=1.0,
                source_model="paddle-v6",
                class_name="text",
            )
            for b in entry["boxes"]
        ]
        n_boxes_total += len(boxes)
        t0 = time.time()
        out_path = os.path.join(args.out, os.path.basename(src))
        if args.resume and os.path.exists(out_path):
            with open(out_path, "rb") as fh:
                if fh.read(8) == b"\x89PNG\r\n\x1a\n" and os.path.getsize(out_path) > 1000:
                    print(f"[{i}] slice {key}: SKIP (done)", flush=True)
                    continue
        cleaned = inpainter.inpaint(img, boxes)
        dt = time.time() - t0
        cv2.imwrite(out_path, cleaned)
        print(f"[{i}] slice {key}: {len(boxes)} boxes, {dt:.1f}s -> {out_path}",
              flush=True)
    print(f"DONE: {len(keys)} slices, {n_boxes_total} boxes, "
          f"{time.time() - t_all:.1f}s total", flush=True)


if __name__ == "__main__":
    main()
