#!/usr/bin/env python3
"""Compare 1024 vs specialized 640/512 text segmenters on real retry/residue ROIs.

This is a throwaway research harness. The 1024 production segmenter is the
teacher/reference; candidates must preserve its verified text evidence before
we consider any production routing change.
"""
from __future__ import annotations

import argparse
import contextvars
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import onnx

import app.detector.bubble_detector as yolo_mod
from app.detector.bubble_detector import YoloDetector
from app.detector.combined_detector import CombinedTextDetector
from app.detector.fast_residue_detector import FastResidueAdaptiveFocusCombinedTextDetector
from app.detector.recovery import SecondaryTextRecovery
from app.downloader.registry import ASURA_STATIC_ADAPTER
from app.model_contracts import validate_detector_session
from app.ort_utils import make_session
from app.parameters import TEXT_CONF_THRESHOLD
from app.optimized_pipeline import OptimizedChapterPipeline
from scripts.onnx_head_patch_probe import specialize


PHASES = ("grayscale_retry", "mser_promotion", "residue_verify")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def strip_value_info(path: Path) -> None:
    model = onnx.load(str(path), load_external_data=True)
    del model.graph.value_info[:]
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def prepare_inputs(chapter_url: str, raw_dir: Path):
    raw_dir.mkdir(parents=True, exist_ok=True)
    urls = ASURA_STATIC_ADAPTER.extract_image_urls(chapter_url)
    if len(urls) < 4:
        raise RuntimeError("chapter did not provide enough source images")
    chosen = sorted(set([1, len(urls) // 2, len(urls) - 2]))
    paths = ASURA_STATIC_ADAPTER.download_urls(
        [urls[i] for i in chosen],
        raw_dir,
        referer=chapter_url,
    )
    if len(paths) != len(chosen):
        raise RuntimeError("incomplete chapter input download")
    return sorted(paths)


def detector_for(path: Path, size: int):
    detector = object.__new__(YoloDetector)
    detector.model_path = str(path)
    detector.source_model = path.name
    detector.model_role = "text_segmenter"
    detector.session = make_session(path)
    detector.contract = validate_detector_session(
        detector.session,
        role="text_segmenter",
        configured_input_size=size,
    )
    detector.input_name = detector.contract.input_name
    detector.conf_threshold = TEXT_CONF_THRESHOLD
    detector.use_tta = False
    return detector


@contextmanager
def temporary_input_size(size: int):
    old = yolo_mod.INPUT_SIZE
    yolo_mod.INPUT_SIZE = int(size)
    try:
        yield
    finally:
        yolo_mod.INPUT_SIZE = old


def infer(detector, image, size):
    started = time.perf_counter()
    with temporary_input_size(size):
        boxes = detector._detect_single_plain(image, 0, 0)
    elapsed = (time.perf_counter() - started) * 1000.0
    return boxes, elapsed


def union_mask(shape, boxes):
    h, w = shape[:2]
    out = np.zeros((h, w), dtype=np.uint8)
    for box in boxes:
        if not box.verified_mask:
            continue
        x1 = max(0, min(w, int(box.x1)))
        y1 = max(0, min(h, int(box.y1)))
        x2 = max(x1, min(w, int(box.x2)))
        y2 = max(y1, min(h, int(box.y2)))
        if x2 <= x1 or y2 <= y1:
            continue
        mask = box.mask
        bw = max(1, int(box.x2) - int(box.x1))
        bh = max(1, int(box.y2) - int(box.y1))
        if mask.shape != (bh, bw):
            mask = cv2.resize(mask, (bw, bh), interpolation=cv2.INTER_NEAREST)
        sx1 = x1 - int(box.x1)
        sy1 = y1 - int(box.y1)
        sx2 = sx1 + (x2 - x1)
        sy2 = sy1 + (y2 - y1)
        part = mask[sy1:sy2, sx1:sx2]
        out[y1:y2, x1:x2] = np.maximum(out[y1:y2, x1:x2], part)
    return out


def iou(a, b):
    ix1=max(a.x1,b.x1); iy1=max(a.y1,b.y1)
    ix2=min(a.x2,b.x2); iy2=min(a.y2,b.y2)
    if ix2<=ix1 or iy2<=iy1:
        return 0.0
    inter=(ix2-ix1)*(iy2-iy1)
    aa=max(1,(a.x2-a.x1)*(a.y2-a.y1))
    bb=max(1,(b.x2-b.x1)*(b.y2-b.y1))
    return inter/float(aa+bb-inter)


def compare_one(image, teacher_boxes, candidate_boxes):
    teacher = [b for b in teacher_boxes if b.verified_mask]
    candidate = [b for b in candidate_boxes if b.verified_mask]
    tm = union_mask(image.shape, teacher)
    cm = union_mask(image.shape, candidate)
    t = tm > 0
    c = cm > 0
    inter = int(np.count_nonzero(t & c))
    tpx = int(np.count_nonzero(t))
    cpx = int(np.count_nonzero(c))
    matched = 0
    for base in teacher:
        if candidate and max(iou(base, alt) for alt in candidate) >= 0.30:
            matched += 1
    return {
        "teacher_boxes": len(teacher),
        "candidate_boxes": len(candidate),
        "teacher_mask_pixels": tpx,
        "candidate_mask_pixels": cpx,
        "mask_intersection_pixels": inter,
        "mask_recall_vs_teacher": round(inter / float(max(1, tpx)), 6),
        "mask_precision_vs_teacher": round(inter / float(max(1, cpx)), 6),
        "box_recall_iou30": round(matched / float(max(1, len(teacher))), 6),
        "teacher_positive": bool(tpx),
        "candidate_positive": bool(cpx),
    }


def capture_real_rois(pipeline, raw_paths, workers: int):
    phase = contextvars.ContextVar("quality_probe_phase", default="primary")
    captured = []
    original_forward = YoloDetector._detect_single_plain

    def wrap_phase(owner, method, name):
        original = getattr(owner, method)
        def wrapped(*args, **kwargs):
            current = phase.get()
            token = None
            if current == "primary":
                token = phase.set(name)
            try:
                return original(*args, **kwargs)
            finally:
                if token is not None:
                    phase.reset(token)
        setattr(owner, method, wrapped)
        return original

    originals = []
    originals.append((CombinedTextDetector, "_grayscale_text_retry",
                      wrap_phase(CombinedTextDetector, "_grayscale_text_retry", "grayscale_retry")))
    originals.append((CombinedTextDetector, "_focused_text_retry",
                      wrap_phase(CombinedTextDetector, "_focused_text_retry", "mser_promotion")))
    originals.append((FastResidueAdaptiveFocusCombinedTextDetector, "verify_post_inpaint_residue",
                      wrap_phase(FastResidueAdaptiveFocusCombinedTextDetector,
                                 "verify_post_inpaint_residue", "residue_verify")))

    def capture_forward(self, image, offset_x, offset_y):
        name = phase.get()
        if self.model_role == "text_segmenter" and name in PHASES:
            digest = hashlib.sha256(image.tobytes()).hexdigest()[:16]
            captured.append({
                "phase": name,
                "image": image.copy(),
                "source_shape": [int(image.shape[0]), int(image.shape[1])],
                "offset": [int(offset_x), int(offset_y)],
                "sha16": digest,
            })
        return original_forward(self, image, offset_x, offset_y)

    YoloDetector._detect_single_plain = capture_forward
    try:
        chapter_id = "onnxq" + hashlib.sha256(str(time.time_ns()).encode()).hexdigest()[:8]
        manifest = pipeline._build_chapter_from_raw_paths(
            chapter_id,
            raw_paths,
            source_url=None,
            workers=workers,
        )
        count = len(manifest["pages"])
        indices = sorted(set([
            min(1, count - 1),
            min(2, count - 1),
            count // 2,
            max(0, count - 2),
        ]))
        pipeline.process_pages(chapter_id, indices, workers=workers)
        return captured, indices, count
    finally:
        YoloDetector._detect_single_plain = original_forward
        for owner, method, original in originals:
            setattr(owner, method, original)


def main() -> int:
    p=argparse.ArgumentParser()
    p.add_argument("--chapter-url",
        default="https://asurascans.com/comics/killer-pietro-08677664/chapter/120")
    p.add_argument("--raw-dir", type=Path,
        default=Path("benchmark-results/onnx-quality-input"))
    p.add_argument("--output-dir", type=Path,
        default=Path("benchmark-results/onnx-quality"))
    p.add_argument("--workers", type=int, default=1)
    args=p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_paths=prepare_inputs(args.chapter_url,args.raw_dir)
    pipeline=OptimizedChapterPipeline()
    crops, indices, slice_count=capture_real_rois(pipeline, raw_paths, args.workers)
    if not crops:
        raise RuntimeError("no retry/residue ROI crops captured")

    candidates={1024: pipeline.detector.text_detector}
    for size in (640,512):
        path=args.output_dir/f"text_segmenter_{size}.onnx"
        specialize(Path("models/text_segmenter.onnx"),path,size)
        strip_value_info(path)
        candidates[size]=detector_for(path,size)

    report={
        "chapter_url":args.chapter_url,
        "indices":indices,
        "slice_count":slice_count,
        "crop_count":len(crops),
        "phase_counts":{},
        "candidates":{},
        "crops":[],
    }
    for item in crops:
        report["phase_counts"][item["phase"]]=report["phase_counts"].get(item["phase"],0)+1

    aggregate={size:{
        "calls":0,"elapsed_ms":0.0,"teacher_mask_pixels":0,"candidate_mask_pixels":0,
        "intersection_pixels":0,"teacher_boxes":0,"matched_boxes":0,
        "teacher_positive_crops":0,"candidate_positive_on_teacher_positive":0,
        "candidate_positive_on_teacher_negative":0,
    } for size in (640,512)}
    timings={size:[] for size in (1024,640,512)}

    for index,item in enumerate(crops):
        image=item["image"]
        teacher_boxes, teacher_ms=infer(candidates[1024],image,1024)
        timings[1024].append(teacher_ms)
        crop_row={k:v for k,v in item.items() if k!="image"}
        crop_row["index"]=index
        crop_row["teacher_ms"]=round(teacher_ms,3)
        crop_row["teacher_verified_boxes"]=len([b for b in teacher_boxes if b.verified_mask])
        crop_row["candidates"]={}
        for size in (640,512):
            boxes,elapsed=infer(candidates[size],image,size)
            timings[size].append(elapsed)
            cmp=compare_one(image,teacher_boxes,boxes)
            cmp["elapsed_ms"]=round(elapsed,3)
            crop_row["candidates"][str(size)]=cmp
            agg=aggregate[size]
            agg["calls"]+=1
            agg["elapsed_ms"]+=elapsed
            agg["teacher_mask_pixels"]+=cmp["teacher_mask_pixels"]
            agg["candidate_mask_pixels"]+=cmp["candidate_mask_pixels"]
            agg["intersection_pixels"]+=cmp["mask_intersection_pixels"]
            agg["teacher_boxes"]+=cmp["teacher_boxes"]
            if cmp["teacher_boxes"]:
                agg["matched_boxes"]+=round(cmp["box_recall_iou30"]*cmp["teacher_boxes"])
            if cmp["teacher_positive"]:
                agg["teacher_positive_crops"]+=1
                agg["candidate_positive_on_teacher_positive"]+=int(cmp["candidate_positive"])
            else:
                agg["candidate_positive_on_teacher_negative"]+=int(cmp["candidate_positive"])
        report["crops"].append(crop_row)

    report["timing"]={
        str(size):{
            "calls":len(values),
            "sum_ms":round(sum(values),3),
            "median_ms":round(float(np.median(values)),3),
            "mean_ms":round(float(np.mean(values)),3),
        } for size,values in timings.items()
    }
    for size,agg in aggregate.items():
        report["candidates"][str(size)]={
            **{k:(round(v,3) if isinstance(v,float) else v) for k,v in agg.items()},
            "weighted_mask_recall_vs_teacher":round(
                agg["intersection_pixels"]/float(max(1,agg["teacher_mask_pixels"])),6),
            "weighted_mask_precision_vs_teacher":round(
                agg["intersection_pixels"]/float(max(1,agg["candidate_mask_pixels"])),6),
            "box_recall_iou30":round(
                agg["matched_boxes"]/float(max(1,agg["teacher_boxes"])),6),
            "positive_crop_recall":round(
                agg["candidate_positive_on_teacher_positive"]/
                float(max(1,agg["teacher_positive_crops"])),6),
            "speedup_vs_1024_sum":round(
                report["timing"]["1024"]["sum_ms"]/
                float(max(1e-9,report["timing"][str(size)]["sum_ms"])),3),
        }

    write_json(args.output_dir/"report.json",report)
    print(json.dumps({
        "crop_count":report["crop_count"],
        "phase_counts":report["phase_counts"],
        "timing":report["timing"],
        "candidates":report["candidates"],
    },indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
