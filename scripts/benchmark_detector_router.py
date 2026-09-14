from __future__ import annotations

import gc
import hashlib
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

from app.detector.adaptive_focus_detector import (
    FOCUS_MAX_SOURCE_SIDE,
    AdaptiveFocusCombinedTextDetector,
    _adaptive_detect,
    _focus_text_detect,
)
from app.detector.bubble_detector import BubbleBox
from app.detector.combined_detector import CombinedTextDetector
from app.detector.sequential_fast_residue_detector import (
    SequentialFastResidueAdaptiveFocusCombinedTextDetector,
)
from app.image_io import read_image
from app.pipeline import ChapterPipeline
from app.parameters import DETECTOR_INPUT_SIZE, DETECTOR_TALL_IMAGE_FACTOR


CHAPTER_URL = "https://asurascans.com/comics/killer-pietro-08677664/chapter/120"
OUT = Path("benchmark-results/detector-router/report.json")
SAMPLE_COUNT = 16
ROUTER_MIN_PROPOSALS = 3
ROUTER_MAX_HEIGHT = 2 * FOCUS_MAX_SOURCE_SIDE


def _two_covering_tiles(height: int, width: int) -> list[tuple[int, int, int, int]]:
    height, width = int(height), int(width)
    if height <= 0 or width <= 0 or height > ROUTER_MAX_HEIGHT:
        return []
    if height <= FOCUS_MAX_SOURCE_SIDE:
        return [(0, 0, width, height)]
    first = (0, 0, width, min(height, FOCUS_MAX_SOURCE_SIDE))
    second_y1 = max(0, height - FOCUS_MAX_SOURCE_SIDE)
    second = (0, second_y1, width, height)
    return [first, second]


def _two_tile_text_detect(detector, image: np.ndarray):
    h, w = image.shape[:2]
    all_boxes: list[BubbleBox] = []
    tiles = _two_covering_tiles(h, w)
    for x1, y1, x2, y2 in tiles:
        crop = image[y1:y2, x1:x2]
        if crop.size:
            all_boxes.extend(detector._detect_single_plain(crop, x1, y1))
    boxes = detector._nms_boxes(all_boxes)
    return [
        detector._with_semantics(box)
        for box in detector._filter_invalid(boxes, w, h)
    ], tiles


class TwoTileRouterDetector(SequentialFastResidueAdaptiveFocusCombinedTextDetector):
    """Benchmark-only candidate that replaces full+focus with full-coverage tiles.

    It activates only on medium-height tall slices with at least three independent
    bubble/MSER proposals. Two 1344-high full-width tiles then cover every source
    pixel, so the candidate saves one 1024x1024 text-segmenter inference compared
    with the historical full-pass + two-focus pattern seen in the profiling data.
    All other pages execute the validated adaptive-focus detector unchanged.
    """

    def detect(self, image: np.ndarray, *, parallel: bool = False):
        started_at = time.perf_counter()
        h, w = image.shape[:2]

        bubble_started = time.perf_counter()
        bubble_boxes = _adaptive_detect(self._bubble_model, image)
        bubble_ms = (time.perf_counter() - bubble_started) * 1000.0

        proposal_started = time.perf_counter()
        recovery_boxes = self.recovery.detect(image, existing=bubble_boxes)
        proposal_ms = (time.perf_counter() - proposal_started) * 1000.0
        proposals = list(bubble_boxes) + list(recovery_boxes)

        tall = h > DETECTOR_INPUT_SIZE * DETECTOR_TALL_IMAGE_FACTOR
        use_router = bool(
            tall
            and h <= ROUTER_MAX_HEIGHT
            and len(proposals) >= ROUTER_MIN_PROPOSALS
        )

        text_started = time.perf_counter()
        if use_router:
            text_boxes, tiles = _two_tile_text_detect(self._text_model, image)
            focus_metrics = {
                "focus_proposals": len(proposals),
                "focus_uncovered_proposals": len(proposals),
                "focus_chip_calls": len(tiles),
                "focus_source_pixels": sum((x2-x1)*(y2-y1) for x1,y1,x2,y2 in tiles),
                "focus_tensor_pixels": len(tiles) * DETECTOR_INPUT_SIZE * DETECTOR_INPUT_SIZE,
                "focus_deferred_regions": 0,
                "focus_fallback_calls": 0,
            }
            deferred_boxes: list[BubbleBox] = []
        else:
            text_boxes, focus_metrics, deferred_boxes = _focus_text_detect(
                self._text_model,
                image,
                proposals,
            )
        text_ms = (time.perf_counter() - text_started) * 1000.0

        with self.bubble_detector.prefetched(image, bubble_boxes):
            with self.text_detector.prefetched(image, text_boxes):
                result = CombinedTextDetector.detect(self, image, parallel=False)
        if deferred_boxes:
            result.extend(deferred_boxes)

        metrics = dict(getattr(self._metrics_local, "value", {}) or {})
        metrics["bubble_model_ms"] = round(bubble_ms, 3)
        metrics["text_model_ms"] = round(text_ms, 3)
        metrics["focus_prefetch_mser_ms"] = round(proposal_ms, 3)
        metrics["focus_prefetch_proposals"] = len(recovery_boxes)
        metrics["mser_ms"] = round(float(metrics.get("mser_ms", 0.0)) + proposal_ms, 3)
        metrics.update(focus_metrics)
        metrics["router_two_tile_enabled"] = int(use_router)
        metrics["router_source_height"] = int(h)
        metrics["router_proposals"] = int(len(proposals))
        metrics["total_ms"] = round((time.perf_counter() - started_at) * 1000.0, 3)
        self._metrics_local.value = metrics
        return result


def _authority_mask(boxes: list[BubbleBox], h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    for box in boxes:
        if not bool(box.safe_to_inpaint) or not box.verified_mask:
            continue
        x1=max(0,min(w,int(box.x1))); y1=max(0,min(h,int(box.y1)))
        x2=max(x1,min(w,int(box.x2))); y2=max(y1,min(h,int(box.y2)))
        if x2<=x1 or y2<=y1:
            continue
        local = box.mask
        if local.shape != (box.y2-box.y1, box.x2-box.x1):
            local = cv2.resize(local, (box.x2-box.x1, box.y2-box.y1), interpolation=cv2.INTER_NEAREST)
        sx1=x1-int(box.x1); sy1=y1-int(box.y1)
        sx2=sx1+(x2-x1); sy2=sy1+(y2-y1)
        target=mask[y1:y2,x1:x2]
        mask[y1:y2,x1:x2] = np.maximum(target, local[sy1:sy2,sx1:sx2])
    return mask


def _review_signature(boxes: list[BubbleBox]) -> list[tuple]:
    return sorted(
        [
            (int(b.x1),int(b.y1),int(b.x2),int(b.y2),str(b.semantic_type),str(b.source_model),b.deferred_reason)
            for b in boxes if bool(b.needs_review)
        ],
        key=repr,
    )


def _run(detector, paths: dict[int, Path], indices: list[int], warmup_path: Path, label: str) -> dict:
    warm=read_image(warmup_path); detector.detect(warm, parallel=False); del warm
    elapsed=[]; rows=[]; authority={}; review={}; counts={}
    for index in indices:
        image=read_image(paths[index]); h,w=image.shape[:2]
        started=time.perf_counter(); boxes=detector.detect(image, parallel=False)
        elapsed.append((time.perf_counter()-started)*1000.0)
        metrics=detector.last_metrics(); rows.append(metrics)
        authority[str(index)] = hashlib.sha256(_authority_mask(boxes,h,w).tobytes()).hexdigest()
        review[str(index)] = _review_signature(boxes)
        counts[str(index)] = {
            "boxes":len(boxes),
            "safe":sum(bool(b.safe_to_inpaint) for b in boxes),
            "review":sum(bool(b.needs_review) for b in boxes),
        }
        del image,boxes
    return {
        "label":label,
        "wall_ms":round(sum(elapsed),3),
        "mean_ms":round(statistics.mean(elapsed),3),
        "median_ms":round(statistics.median(elapsed),3),
        "bubble_model_mean_ms":round(statistics.mean(float(r.get("bubble_model_ms") or 0) for r in rows),3),
        "text_model_mean_ms":round(statistics.mean(float(r.get("text_model_ms") or 0) for r in rows),3),
        "total_metric_mean_ms":round(statistics.mean(float(r.get("total_ms") or 0) for r in rows),3),
        "focus_chip_calls":int(sum(int(r.get("focus_chip_calls") or 0) for r in rows)),
        "router_pages":int(sum(bool(r.get("router_two_tile_enabled")) for r in rows)),
        "authority_hashes":authority,
        "review_signatures":review,
        "counts":counts,
    }


def _pct(before: float, after: float) -> float:
    return round((1.0-after/max(1.0,before))*100.0,2)


def main() -> None:
    OUT.parent.mkdir(parents=True,exist_ok=True)
    pipeline=ChapterPipeline()
    chapter_id=hashlib.sha256(f"detector-router-{time.time_ns()}".encode()).hexdigest()[:8]
    manifest=pipeline.download_chapter(CHAPTER_URL,chapter_id,workers=2)
    pages=manifest.get("pages",[]); total=len(pages)
    indices=sorted({round(i*(total-1)/(SAMPLE_COUNT-1)) for i in range(SAMPLE_COUNT)})
    paths={i:Path(pages[i]["original"]) for i in indices}
    warmup_index=next(i for i in range(total) if i not in set(indices)); warmup_path=Path(pages[warmup_index]["original"])

    control_detector=SequentialFastResidueAdaptiveFocusCombinedTextDetector()
    control=_run(control_detector,paths,indices,warmup_path,"control_full_plus_focus")
    del control_detector; gc.collect()

    candidate_detector=TwoTileRouterDetector()
    candidate=_run(candidate_detector,paths,indices,warmup_path,"candidate_two_tile_router")
    del candidate_detector; gc.collect()

    authority_mismatch=[k for k,v in control["authority_hashes"].items() if candidate["authority_hashes"].get(k)!=v]
    review_mismatch=[k for k,v in control["review_signatures"].items() if candidate["review_signatures"].get(k)!=v]
    count_mismatch=[k for k,v in control["counts"].items() if candidate["counts"].get(k)!=v]
    report={
        "chapter_url":CHAPTER_URL,"chapter_id":chapter_id,"total_slices":total,"sample_indices":indices,
        "router_gate":{"max_height":ROUTER_MAX_HEIGHT,"min_proposals":ROUTER_MIN_PROPOSALS},
        "control":control,"candidate":candidate,
        "speedup":{
            "wall_reduction_pct":_pct(control["wall_ms"],candidate["wall_ms"]),
            "text_model_reduction_pct":_pct(control["text_model_mean_ms"],candidate["text_model_mean_ms"]),
        },
        "quality":{
            "authority_mask_mismatch_pages":authority_mismatch,
            "review_signature_mismatch_pages":review_mismatch,
            "box_count_mismatch_pages":count_mismatch,
        },
    }
    OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("DETECTOR_ROUTER_AB="+json.dumps(report,ensure_ascii=False),flush=True)
    if authority_mismatch:
        raise RuntimeError("router changed destructive authority masks: "+json.dumps(authority_mismatch))


if __name__=="__main__":
    main()
