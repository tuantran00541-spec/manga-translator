from __future__ import annotations

import gc
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import time

import cv2
import numpy as np

from app.detector.adaptive_focus_detector import _adaptive_detect, _focus_text_detect
from app.detector.bubble_detector import BubbleBox
from app.detector.combined_detector import CombinedTextDetector
from app.detector.sequential_fast_residue_detector import (
    SequentialFastResidueAdaptiveFocusCombinedTextDetector,
)
from app.image_io import read_image
from app.parameters import DETECTOR_INPUT_SIZE, DETECTOR_TALL_IMAGE_FACTOR
from app.pipeline import ChapterPipeline

CHAPTER_URL = "https://asurascans.com/comics/killer-pietro-08677664/chapter/120"
OUT = Path("benchmark-results/bubble-uncertainty/report.json")
SAMPLE_COUNT = 16
AUDIT_BANDS = 10
AUDIT_WIDTH = 256


def _single_pass_bubble_detect(detector, image: np.ndarray) -> list[BubbleBox]:
    h, w = image.shape[:2]
    boxes = detector._detect_single_plain(image, 0, 0)
    return [detector._with_semantics(b) for b in detector._filter_invalid(boxes, w, h)]


def _gabor_scores(image: np.ndarray) -> list[float]:
    h, w = image.shape[:2]
    scale = min(1.0, AUDIT_WIDTH / float(max(1, w)))
    rw, rh = max(1, round(w * scale)), max(1, round(h * scale))
    small = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    response = np.zeros_like(gray)
    for theta in (0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4):
        kernel = cv2.getGaborKernel((9, 9), 2.5, theta, 5.0, 0.5, 0, ktype=cv2.CV_32F)
        response = np.maximum(response, np.abs(cv2.filter2D(gray, cv2.CV_32F, kernel)))
    edges = cv2.Canny((gray * 255).astype(np.uint8), 60, 160) > 0
    out = []
    for i in range(AUDIT_BANDS):
        y1, y2 = round(i * rh / AUDIT_BANDS), round((i + 1) * rh / AUDIT_BANDS)
        out.append(float(response[y1:y2].mean()) + 0.2 * float(edges[y1:y2].mean()))
    return out


def _pre_features(image: np.ndarray, recovery: list[BubbleBox]) -> dict[str, float]:
    h, w = image.shape[:2]
    scores = _gabor_scores(image)
    arr = np.asarray(scores, dtype=np.float32)
    median = float(np.median(arr)) if arr.size else 0.0
    mad = float(np.median(np.abs(arr - median))) if arr.size else 0.0
    floor = median + max(0.0025, 1.25 * mad)
    centers = sorted(((b.y1 + b.y2) * 0.5) / max(1.0, h) for b in recovery)
    boundaries = [0.0] + centers + [1.0]
    max_gap = max((boundaries[i + 1] - boundaries[i] for i in range(len(boundaries) - 1)), default=1.0)
    widths = [(b.x2 - b.x1) / max(1.0, w) for b in recovery]
    heights = [(b.y2 - b.y1) / max(1.0, h) for b in recovery]
    areas = [(b.x2 - b.x1) * (b.y2 - b.y1) / max(1.0, w * h) for b in recovery]
    uncovered = 0
    for i, score in enumerate(scores):
        y1, y2 = round(i * h / AUDIT_BANDS), round((i + 1) * h / AUDIT_BANDS)
        covered = any(max(0, min(b.y2, y2) - max(b.y1, y1)) > 0 for b in recovery)
        uncovered += int((not covered) and score >= floor)
    return {
        "height_ratio": h / float(DETECTOR_INPUT_SIZE),
        "recovery_count": float(len(recovery)),
        "max_gap": float(max_gap),
        "max_width": max(widths, default=0.0),
        "max_height": max(heights, default=0.0),
        "max_area": max(areas, default=0.0),
        "wide_count": float(sum(v >= 0.5 for v in widths)),
        "tall_count": float(sum(v >= 0.12 for v in heights)),
        "gabor_active": float(sum(v >= floor for v in scores)),
        "gabor_uncovered": float(uncovered),
        "gabor_peak_ratio": max(scores, default=0.0) / max(1e-6, median),
    }


def _predicates(features: dict[str, dict]) -> list[tuple[str, str, float]]:
    out = []
    for name in next(iter(features.values())).keys():
        values = sorted({float(row[name]) for row in features.values()})
        thresholds = values + [(a + b) * 0.5 for a, b in zip(values, values[1:])]
        for t in sorted(set(thresholds)):
            out.extend([(name, ">=", t), (name, "<=", t)])
    return out


def _match(row: dict, pred: tuple[str, str, float]) -> bool:
    name, op, t = pred
    return float(row[name]) >= t if op == ">=" else float(row[name]) <= t


def _select_safe_rule(features: dict[str, dict], unsafe: set[str]) -> dict:
    safe = set(features) - unsafe
    candidates = []
    preds = _predicates(features)
    for clause_len in (1, 2):
        for clause in itertools.combinations(preds, clause_len):
            covered = {k for k, row in features.items() if all(_match(row, p) for p in clause)}
            if covered and not (covered & unsafe):
                candidates.append((clause, covered))
    dedup = {}
    for clause, covered in candidates:
        key = tuple(sorted(covered, key=int))
        if key not in dedup or len(clause) < len(dedup[key][0]):
            dedup[key] = (clause, covered)
    candidates = list(dedup.values())
    best = ((), set())
    for count in (1, 2):
        for combo in itertools.combinations(candidates, count):
            covered = set().union(*(x[1] for x in combo))
            if len(covered) > len(best[1]):
                best = (tuple(x[0] for x in combo), covered)
    return {
        "clauses": [[{"feature": p[0], "op": p[1], "threshold": p[2]} for p in clause] for clause in best[0]],
        "fastpath_pages": sorted(best[1], key=int),
        "unsafe_pages": sorted(unsafe, key=int),
        "safe_pages": sorted(safe, key=int),
    }


def _rule_fast(row: dict, rule: dict) -> bool:
    for clause in rule.get("clauses", []):
        if all(_match(row, (p["feature"], p["op"], float(p["threshold"]))) for p in clause):
            return True
    return False


def _authority_mask(boxes: list[BubbleBox], h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), np.uint8)
    for b in boxes:
        if not b.safe_to_inpaint or not b.verified_mask:
            continue
        x1, y1, x2, y2 = max(0, b.x1), max(0, b.y1), min(w, b.x2), min(h, b.y2)
        if x2 <= x1 or y2 <= y1:
            continue
        local = b.mask
        expected = (b.y2 - b.y1, b.x2 - b.x1)
        if local.shape != expected:
            local = cv2.resize(local, (expected[1], expected[0]), interpolation=cv2.INTER_NEAREST)
        sx1, sy1 = x1 - b.x1, y1 - b.y1
        mask[y1:y2, x1:x2] = np.maximum(mask[y1:y2, x1:x2], local[sy1:sy1+y2-y1, sx1:sx1+x2-x1])
    return mask


def _review_signature(boxes: list[BubbleBox]) -> list[tuple]:
    return sorted((b.x1, b.y1, b.x2, b.y2, b.semantic_type, b.source_model, b.deferred_reason) for b in boxes if b.needs_review)


def _finish(detector, image, bubbles, recovery, started, bubble_ms, audit_ms, pre_mser_ms, fast):
    text_started = time.perf_counter()
    text_boxes, focus_metrics, deferred = _focus_text_detect(detector._text_model, image, list(bubbles) + list(recovery))
    text_ms = (time.perf_counter() - text_started) * 1000
    with detector.bubble_detector.prefetched(image, bubbles):
        with detector.text_detector.prefetched(image, text_boxes):
            result = CombinedTextDetector.detect(detector, image, parallel=False)
    result.extend(deferred)
    metrics = dict(getattr(detector._metrics_local, "value", {}) or {})
    metrics.update(focus_metrics)
    metrics.update({"bubble_model_ms": round(bubble_ms, 3), "text_model_ms": round(text_ms, 3),
                    "audit_ms": round(audit_ms, 3), "pregate_mser_ms": round(pre_mser_ms, 3),
                    "bubble_gate_fastpath": int(fast), "bubble_gate_fallback": int(not fast),
                    "total_ms": round((time.perf_counter() - started) * 1000, 3)})
    detector._metrics_local.value = metrics
    return result


class ProbeDetector(SequentialFastResidueAdaptiveFocusCombinedTextDetector):
    def detect(self, image: np.ndarray, *, parallel: bool = False):
        if image.shape[0] <= DETECTOR_INPUT_SIZE * DETECTOR_TALL_IMAGE_FACTOR:
            return super().detect(image, parallel=False)
        started = time.perf_counter()
        t = time.perf_counter(); pre = self.recovery.detect(image, existing=[]); pre_ms = (time.perf_counter()-t)*1000
        t = time.perf_counter(); audit = _pre_features(image, pre); audit_ms = (time.perf_counter()-t)*1000
        t = time.perf_counter(); bubbles = _single_pass_bubble_detect(self._bubble_model, image); bubble_ms = (time.perf_counter()-t)*1000
        recovery = self.recovery.detect(image, existing=bubbles)
        result = _finish(self, image, bubbles, recovery, started, bubble_ms, audit_ms, pre_ms, True)
        metrics = dict(self._metrics_local.value); metrics.update({f"audit_{k}": v for k, v in audit.items()}); self._metrics_local.value = metrics
        return result


class GatedDetector(SequentialFastResidueAdaptiveFocusCombinedTextDetector):
    def __init__(self, rule):
        super().__init__(); self.rule = rule
    def detect(self, image: np.ndarray, *, parallel: bool = False):
        if image.shape[0] <= DETECTOR_INPUT_SIZE * DETECTOR_TALL_IMAGE_FACTOR:
            return super().detect(image, parallel=False)
        started = time.perf_counter()
        t = time.perf_counter(); pre = self.recovery.detect(image, existing=[]); pre_ms = (time.perf_counter()-t)*1000
        t = time.perf_counter(); audit = _pre_features(image, pre); audit_ms = (time.perf_counter()-t)*1000
        fast = _rule_fast(audit, self.rule)
        t = time.perf_counter(); bubbles = _single_pass_bubble_detect(self._bubble_model, image) if fast else _adaptive_detect(self._bubble_model, image); bubble_ms = (time.perf_counter()-t)*1000
        recovery = self.recovery.detect(image, existing=bubbles)
        result = _finish(self, image, bubbles, recovery, started, bubble_ms, audit_ms, pre_ms, fast)
        metrics = dict(self._metrics_local.value); metrics.update({f"audit_{k}": v for k, v in audit.items()}); self._metrics_local.value = metrics
        return result


def _run(detector, paths, indices, warmup_path, label):
    detector.detect(read_image(warmup_path), parallel=False)
    elapsed, rows, authority, review, counts = [], {}, {}, {}, {}
    for index in indices:
        image = read_image(paths[index]); h, w = image.shape[:2]
        t = time.perf_counter(); boxes = detector.detect(image, parallel=False); elapsed.append((time.perf_counter()-t)*1000)
        rows[str(index)] = dict(detector.last_metrics())
        authority[str(index)] = hashlib.sha256(_authority_mask(boxes, h, w).tobytes()).hexdigest()
        review[str(index)] = _review_signature(boxes)
        counts[str(index)] = {"boxes": len(boxes), "safe": sum(b.safe_to_inpaint for b in boxes), "review": sum(b.needs_review for b in boxes)}
    return {"label": label, "wall_ms": round(sum(elapsed),3), "mean_ms": round(statistics.mean(elapsed),3),
            "median_ms": round(statistics.median(elapsed),3), "rows": rows, "authority_hashes": authority,
            "review_signatures": review, "counts": counts}


def _mismatch(a, b):
    return {"authority": [k for k,v in a["authority_hashes"].items() if b["authority_hashes"].get(k)!=v],
            "review": [k for k,v in a["review_signatures"].items() if b["review_signatures"].get(k)!=v],
            "counts": [k for k,v in a["counts"].items() if b["counts"].get(k)!=v]}


def _pct(a,b): return round((1-b/max(1.0,a))*100,2)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pipeline = ChapterPipeline(); chapter_id = hashlib.sha256(f"bubble-pregate-{time.time_ns()}".encode()).hexdigest()[:8]
    manifest = pipeline.download_chapter(CHAPTER_URL, chapter_id, workers=2); pages = manifest["pages"]; total = len(pages)
    indices = sorted({round(i*(total-1)/(SAMPLE_COUNT-1)) for i in range(SAMPLE_COUNT)})
    paths = {i: Path(pages[i]["original"]) for i in indices}; warm = next(i for i in range(total) if i not in set(indices)); warm_path = Path(pages[warm]["original"])
    control = _run(SequentialFastResidueAdaptiveFocusCombinedTextDetector(), paths, indices, warm_path, "control")
    gc.collect(); probe = _run(ProbeDetector(), paths, indices, warm_path, "probe"); gc.collect()
    probe_quality = _mismatch(control, probe); unsafe = set().union(*map(set, probe_quality.values()))
    features = {k: {n[6:]:v for n,v in row.items() if n.startswith("audit_") and n!="audit_ms"} for k,row in probe["rows"].items() if any(n.startswith("audit_") for n in row)}
    rule = _select_safe_rule(features, unsafe)
    candidate = _run(GatedDetector(rule), paths, indices, warm_path, "candidate"); quality = _mismatch(control, candidate)
    tall_rows = [r for r in candidate["rows"].values() if "bubble_gate_fastpath" in r]
    report = {"chapter_url": CHAPTER_URL, "chapter_id": chapter_id, "sample_indices": indices,
              "audit": {"rule": rule, "features": features}, "control": control, "probe": probe,
              "probe_quality": probe_quality, "candidate": candidate, "candidate_quality": quality,
              "speedup": {"probe_wall_reduction_pct": _pct(control["wall_ms"],probe["wall_ms"]),
                          "candidate_wall_reduction_pct": _pct(control["wall_ms"],candidate["wall_ms"]),
                          "fastpath_pages": sum(int(r.get("bubble_gate_fastpath",0)) for r in tall_rows),
                          "fallback_pages": sum(int(r.get("bubble_gate_fallback",0)) for r in tall_rows),
                          "audit_mean_ms": round(statistics.mean(float(r.get("audit_ms",0)) for r in tall_rows),3),
                          "pregate_mser_mean_ms": round(statistics.mean(float(r.get("pregate_mser_ms",0)) for r in tall_rows),3)}}
    OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); print("BUBBLE_UNCERTAINTY_AB="+json.dumps(report),flush=True)
    bad=set().union(*map(set,quality.values()))
    if bad: raise RuntimeError("pre-gate strict mismatch: "+json.dumps(sorted(bad,key=int)))

if __name__ == "__main__": main()
