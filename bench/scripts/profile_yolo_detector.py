#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


def ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


class ProfiledYoloDetectorMixin:
    def _profile_init(self) -> None:
        self._profile = {
            "slices": [],
            "final_nms_ms": 0.0,
            "final_nms_input_boxes": 0,
            "final_nms_output_boxes": 0,
        }
        self._current_slice: dict[str, Any] | None = None

    def _new_slice(self, image, offset_x: int, offset_y: int) -> dict[str, Any]:
        row = {
            "index": len(self._profile["slices"]),
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "offset_x": int(offset_x),
            "offset_y": int(offset_y),
            "total_ms": 0.0,
            "preprocess_ms": 0.0,
            "ort_ms": 0.0,
            "postprocess_ms": 0.0,
            "mask_decode_ms": 0.0,
            "boxes_out": 0,
        }
        self._profile["slices"].append(row)
        return row

    def _detect_single_plain(self, image, offset_x: int, offset_y: int):
        row = self._new_slice(image, offset_x, offset_y)
        self._current_slice = row
        t0 = time.perf_counter()
        try:
            return super()._detect_single_plain(image, offset_x, offset_y)
        finally:
            row["total_ms"] = ms(t0)
            self._current_slice = None

    def _preprocess(self, image):
        t0 = time.perf_counter()
        try:
            return super()._preprocess(image)
        finally:
            if self._current_slice is not None:
                self._current_slice["preprocess_ms"] += ms(t0)

    def _postprocess(self, outputs, scale, pad, orig_w, orig_h):
        t0 = time.perf_counter()
        try:
            return super()._postprocess(outputs, scale, pad, orig_w, orig_h)
        finally:
            if self._current_slice is not None:
                self._current_slice["postprocess_ms"] += ms(t0)

    def _decode_mask(self, *args, **kwargs):
        t0 = time.perf_counter()
        try:
            return super()._decode_mask(*args, **kwargs)
        finally:
            if self._current_slice is not None:
                self._current_slice["mask_decode_ms"] += ms(t0)

    def _nms_boxes(self, boxes):
        t0 = time.perf_counter()
        result = super()._nms_boxes(boxes)
        elapsed = ms(t0)
        self._profile["final_nms_ms"] += elapsed
        self._profile["final_nms_input_boxes"] += len(boxes)
        self._profile["final_nms_output_boxes"] = len(result)
        return result

    def _post_slice_result(self, boxes):
        if self._profile["slices"]:
            self._profile["slices"][-1]["boxes_out"] = len(boxes)


class ProfiledYoloDetector(ProfiledYoloDetectorMixin):
    pass


def build_profiled_detector(repo: Path, model: Path, conf: float, threads: int, tta: bool):
    os.environ["MANGA_ORT_INTRA_OP_THREADS"] = str(max(1, threads))
    os.environ["ENABLE_TTA"] = "1" if tta else "0"
    sys.path.insert(0, str(repo.resolve()))

    from app.detector.bubble_detector import YoloDetector

    class _Detector(ProfiledYoloDetector, YoloDetector):
        def __init__(self, *args, **kwargs):
            YoloDetector.__init__(self, *args, **kwargs)
            self._profile_init()

        def _detect_single_plain(self, image, offset_x, offset_y):
            row = self._new_slice(image, offset_x, offset_y)
            self._current_slice = row
            t0 = time.perf_counter()
            try:
                result = YoloDetector._detect_single_plain(self, image, offset_x, offset_y)
                row["boxes_out"] = len(result)
                return result
            finally:
                row["total_ms"] = ms(t0)
                self._current_slice = None

        def _preprocess(self, image):
            t0 = time.perf_counter()
            try:
                return YoloDetector._preprocess(self, image)
            finally:
                if self._current_slice is not None:
                    self._current_slice["preprocess_ms"] += ms(t0)

        def _postprocess(self, outputs, scale, pad, orig_w, orig_h):
            t0 = time.perf_counter()
            try:
                return YoloDetector._postprocess(self, outputs, scale, pad, orig_w, orig_h)
            finally:
                if self._current_slice is not None:
                    self._current_slice["postprocess_ms"] += ms(t0)

        def _decode_mask(self, *args, **kwargs):
            t0 = time.perf_counter()
            try:
                return YoloDetector._decode_mask(self, *args, **kwargs)
            finally:
                if self._current_slice is not None:
                    self._current_slice["mask_decode_ms"] += ms(t0)

        def _nms_boxes(self, boxes):
            t0 = time.perf_counter()
            result = YoloDetector._nms_boxes(self, boxes)
            self._profile["final_nms_ms"] += ms(t0)
            self._profile["final_nms_input_boxes"] += len(boxes)
            self._profile["final_nms_output_boxes"] = len(result)
            return result

    return _Detector(model, conf, use_tta=tta)


def profile_image(detector, image_path: Path):
    import cv2

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"OpenCV failed to read {image_path}")

    detector._profile_init()
    t0 = time.perf_counter()
    boxes = detector.detect(image)
    total_ms = ms(t0)

    profile = detector._profile
    slices = profile["slices"]
    ort_total = max(
        0.0,
        sum(s["total_ms"] for s in slices)
        - sum(s["preprocess_ms"] for s in slices)
        - sum(s["postprocess_ms"] for s in slices),
    )
    post_no_mask = max(
        0.0,
        sum(s["postprocess_ms"] for s in slices) - sum(s["mask_decode_ms"] for s in slices),
    )
    return {
        "image": image_path.name,
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "total_ms": total_ms,
        "final_boxes": len(boxes),
        "slice_count": len(slices),
        "slice_total_ms": sum(s["total_ms"] for s in slices),
        "preprocess_total_ms": sum(s["preprocess_ms"] for s in slices),
        "ort_estimated_ms": ort_total,
        "postprocess_total_ms": sum(s["postprocess_ms"] for s in slices),
        "mask_decode_total_ms": sum(s["mask_decode_ms"] for s in slices),
        "postprocess_excluding_mask_ms": post_no_mask,
        "final_nms_ms": profile["final_nms_ms"],
        "final_nms_input_boxes": profile["final_nms_input_boxes"],
        "final_nms_output_boxes": profile["final_nms_output_boxes"],
        "slices": slices,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--conf", type=float, default=0.40)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("names", nargs="+", help="Image filenames")
    args = ap.parse_args()

    if not args.model.is_file():
        print(f"ERROR: missing model: {args.model}", file=sys.stderr)
        return 2

    print(f"[INIT] Loading production YoloDetector: {args.model}", flush=True)
    t0 = time.perf_counter()
    detector = build_profiled_detector(args.repo, args.model, args.conf, args.threads, args.tta)
    print(f"[INIT] Model ready: {ms(t0):.1f} ms | threads={max(1,args.threads)} | TTA={bool(args.tta)}", flush=True)

    results = []
    for i, name in enumerate(args.names, 1):
        path = args.images / name
        print(f"[{i}/{len(args.names)}] Profiling {path.name} ...", flush=True)
        result = profile_image(detector, path)
        results.append(result)
        print(
            f"    {result['width']}x{result['height']} | slices={result['slice_count']} | "
            f"total={result['total_ms']:.1f} ms | "
            f"pre={result['preprocess_total_ms']:.1f} | "
            f"ORT≈{result['ort_estimated_ms']:.1f} | "
            f"post={result['postprocess_total_ms']:.1f} | "
            f"mask={result['mask_decode_total_ms']:.1f} | "
            f"final_nms={result['final_nms_ms']:.1f} | boxes={result['final_boxes']}",
            flush=True,
        )
        for s in result["slices"]:
            print(
                f"      slice {s['index']:02d} @ y={s['offset_y']:5d} "
                f"{s['width']}x{s['height']} | total={s['total_ms']:.1f} ms | "
                f"pre={s['preprocess_ms']:.1f} | "
                f"ORT≈{max(0.0, s['total_ms']-s['preprocess_ms']-s['postprocess_ms']):.1f} | "
                f"post={s['postprocess_ms']:.1f} | mask={s['mask_decode_ms']:.1f} | "
                f"boxes={s['boxes_out']}",
                flush=True,
            )

    out = {
        "benchmark": "production_yolo_detector_profile",
        "runtime": {
            "provider": "CPUExecutionProvider",
            "intra_op_threads": max(1, args.threads),
            "tta": bool(args.tta),
            "conf_threshold": args.conf,
        },
        "model": str(args.model),
        "results": results,
        "note": "Diagnostic instrumentation only. Production detector code/model behavior is not changed; ORT time is estimated as slice total minus preprocess and postprocess.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[DONE] Profile written to: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
