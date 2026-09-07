from __future__ import annotations

"""Branch-only PP-OCRv6 proposal gate for adaptive_focus_v2.

The gate never grants erase authority. It contributes cheap text-line proposals
only; the existing YOLO text segmenter still has to verify a region before it
can become a destructive inpaint mask.
"""

import os
from pathlib import Path
import sys
import threading
import time
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import detector_experiment as exp

_GATE_MODEL_NAME = os.getenv(
    "MANGA_DETECTOR_FOCUS_GATE_MODEL", "PP-OCRv6_small_det"
).strip() or "PP-OCRv6_small_det"
_GATE_LIMIT = max(
    320, int(os.getenv("MANGA_DETECTOR_FOCUS_GATE_LIMIT", "1024"))
)
_GATE_CPU_THREADS = max(
    1, int(os.getenv("MANGA_DETECTOR_FOCUS_GATE_THREADS", "2"))
)
_GATE_SCORE_MIN = max(
    0.0, min(1.0, float(os.getenv("MANGA_DETECTOR_FOCUS_GATE_SCORE", "0.0")))
)
_FOCUS_CHIP_MAX_HEIGHT = max(
    exp.bubble_mod.INPUT_SIZE,
    int(os.getenv("MANGA_DETECTOR_FOCUS_CHIP_MAX_HEIGHT", "1344")),
)
_FOCUS_CHIP_OVERLAP = max(
    0, min(256, int(os.getenv("MANGA_DETECTOR_FOCUS_CHIP_OVERLAP", "96")))
)

_gate_model = None
_gate_create_lock = threading.RLock()
_gate_predict_lock = threading.RLock()
_patched = False


def _payload(result: Any) -> dict[str, Any]:
    value = getattr(result, "json", result)
    if callable(value):
        value = value()
    if isinstance(value, str):
        import json

        value = json.loads(value)
    if not isinstance(value, dict):
        value = dict(value)
    inner = value.get("res", value)
    return inner if isinstance(inner, dict) else value


def _get_gate_model():
    global _gate_model
    if _gate_model is not None:
        return _gate_model
    with _gate_create_lock:
        if _gate_model is not None:
            return _gate_model
        from paddleocr import TextDetection

        _gate_model = TextDetection(
            model_name=_GATE_MODEL_NAME,
            device="cpu",
            enable_mkldnn=True,
            cpu_threads=_GATE_CPU_THREADS,
        )
        return _gate_model


def _poly_boxes(outputs, width: int, height: int):
    boxes = []
    for output in outputs:
        data = _payload(output)
        raw_polys = data.get("dt_polys")
        raw_scores = data.get("dt_scores")
        polys = list(raw_polys) if raw_polys is not None else []
        scores = list(raw_scores) if raw_scores is not None else []
        for index, poly in enumerate(polys):
            try:
                arr = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
            except Exception:
                continue
            if arr.size == 0 or not np.all(np.isfinite(arr)):
                continue
            score = 1.0
            if index < len(scores):
                try:
                    score = float(scores[index])
                except (TypeError, ValueError):
                    score = 1.0
            if score < _GATE_SCORE_MIN:
                continue
            x1 = max(0, min(width, int(np.floor(arr[:, 0].min()))))
            y1 = max(0, min(height, int(np.floor(arr[:, 1].min()))))
            x2 = max(0, min(width, int(np.ceil(arr[:, 0].max()))))
            y2 = max(0, min(height, int(np.ceil(arr[:, 1].max()))))
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            boxes.append(
                exp.bubble_mod.BubbleBox(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=max(0.0, min(1.0, score)),
                    source_model="ppocrv6_focus_gate",
                    class_id=0,
                    class_name="text_gate",
                    semantic_type="text",
                    mask_source="none",
                    safe_to_inpaint=False,
                    ocr_eligible=False,
                    needs_review=False,
                )
            )
    return boxes


def _gate_proposals(image):
    height, width = image.shape[:2]
    threshold = int(
        exp.bubble_mod.INPUT_SIZE * exp.bubble_mod.DETECTOR_TALL_IMAGE_FACTOR
    )
    if height <= threshold:
        return [], 0.0, None

    started = time.perf_counter()
    try:
        model = _get_gate_model()
        with _gate_predict_lock:
            outputs = model.predict(
                input=image,
                batch_size=1,
                limit_side_len=_GATE_LIMIT,
                limit_type="max",
            )
        boxes = _poly_boxes(outputs, width, height)
        return boxes, (time.perf_counter() - started) * 1000.0, None
    except Exception as exc:
        return [], (time.perf_counter() - started) * 1000.0, repr(exc)


def _bounded_focus_bands(
    height: int,
    proposals,
    *,
    max_chips: int = exp._FOCUS_MAX_CHIPS,
    merge_gap: int = exp._FOCUS_MERGE_GAP,
):
    """Use the chip count as a soft target without destroying text scale.

    V2 forced every page down to at most two bands; on this chapter that made
    a 4026px "focus" crop which was squeezed back into a 1024px model input.
    V2.1 only merges bands when the merged span stays near detector-native
    scale. Far-apart evidence is allowed to use extra chips rather than lose
    small text.
    """
    proposals = list(proposals or [])
    if not proposals:
        return []

    bands = [
        [int(start), int(end)]
        for start, end in exp._original_plan_focus_bands(
            height,
            proposals,
            max_chips=max(1, len(proposals)),
            merge_gap=merge_gap,
        )
    ]

    soft_target = max(1, int(max_chips))
    while len(bands) > soft_target:
        candidates = []
        for index in range(len(bands) - 1):
            merged_start = bands[index][0]
            merged_end = bands[index + 1][1]
            merged_height = merged_end - merged_start
            if merged_height > _FOCUS_CHIP_MAX_HEIGHT:
                continue
            gap = max(0, bands[index + 1][0] - bands[index][1])
            candidates.append((gap, merged_height, index))
        if not candidates:
            break
        _, _, merge_at = min(candidates)
        bands[merge_at][1] = bands[merge_at + 1][1]
        del bands[merge_at + 1]

    bounded = []
    for start, end in bands:
        size = int(end - start)
        if size <= _FOCUS_CHIP_MAX_HEIGHT:
            bounded.append((int(start), int(end)))
            continue
        windows = exp.plan_adaptive_windows(
            size,
            tile_max=_FOCUS_CHIP_MAX_HEIGHT,
            overlap=_FOCUS_CHIP_OVERLAP,
        )
        bounded.extend((int(start + a), int(start + b)) for a, b in windows)
    return bounded


def _v21_combined_detect(self, image, *, parallel: bool = False):
    """V2 proposal cache plus a detector-only PP-OCR line gate."""
    bubble_started = time.perf_counter()
    bubble_boxes = exp._adaptive_detect(self.bubble_detector, image)
    bubble_ms = (time.perf_counter() - bubble_started) * 1000.0

    proposal_started = time.perf_counter()
    recovery_boxes = self.recovery.detect(image, existing=bubble_boxes)
    proposal_ms = (time.perf_counter() - proposal_started) * 1000.0

    gate_boxes, gate_ms, gate_error = _gate_proposals(image)

    context = {
        "image_id": id(image),
        "shape": tuple(image.shape[:2]),
        "bubble_boxes": list(bubble_boxes),
        "bubble_cache_used": False,
        "proposals": list(bubble_boxes) + list(recovery_boxes) + list(gate_boxes),
    }
    exp._focus_local.value = context
    try:
        result = exp._original_combined_detect(self, image, parallel=False)
    finally:
        exp._focus_local.value = None

    metrics = dict(getattr(self._metrics_local, "value", {}) or {})
    metrics["bubble_model_ms"] = round(bubble_ms, 3)
    metrics["focus_prefetch_mser_ms"] = round(proposal_ms, 3)
    metrics["focus_prefetch_proposals"] = len(recovery_boxes)
    metrics["focus_gate_ms"] = round(gate_ms, 3)
    metrics["focus_gate_proposals"] = len(gate_boxes)
    metrics["mser_ms"] = round(float(metrics.get("mser_ms", 0.0)) + proposal_ms, 3)
    metrics["total_ms"] = round(
        float(metrics.get("total_ms", 0.0)) + bubble_ms + proposal_ms + gate_ms,
        3,
    )
    self._metrics_local.value = metrics

    with exp._lock:
        exp._stats["v2"]["combined_invocations"] += 1
        exp._stats["v2"]["bubble_prefetch_ms"] += bubble_ms
        exp._stats["v2"]["proposal_prefetch_ms"] += proposal_ms
        exp._stats["v2"]["proposal_prefetch_boxes"] += len(recovery_boxes)
        v21 = exp._stats.setdefault(
            "v21",
            {
                "gate_model": _GATE_MODEL_NAME,
                "gate_limit_side_len": _GATE_LIMIT,
                "gate_cpu_threads": _GATE_CPU_THREADS,
                "gate_runs": 0,
                "gate_ms": 0.0,
                "gate_proposals": 0,
                "gate_failures": 0,
                "gate_last_error": None,
            },
        )
        if image.shape[0] > int(
            exp.bubble_mod.INPUT_SIZE * exp.bubble_mod.DETECTOR_TALL_IMAGE_FACTOR
        ):
            v21["gate_runs"] += 1
            v21["gate_ms"] += gate_ms
            v21["gate_proposals"] += len(gate_boxes)
            if gate_error:
                v21["gate_failures"] += 1
                v21["gate_last_error"] = gate_error
    return result


def install_focus_v21() -> bool:
    """Install before detector_experiment.install_from_env()."""
    global _patched
    if _patched:
        return True
    if exp._EXPERIMENT != "adaptive_focus_v2":
        return False
    if not hasattr(exp, "_original_plan_focus_bands"):
        exp._original_plan_focus_bands = exp.plan_focus_bands
    exp._stats["focus_gate_patch"] = "ppocrv6_bounded_v21"
    exp._stats["focus_chip_max_height"] = _FOCUS_CHIP_MAX_HEIGHT
    exp._stats["focus_chip_overlap"] = _FOCUS_CHIP_OVERLAP
    exp.plan_focus_bands = _bounded_focus_bands
    exp._v2_combined_detect = _v21_combined_detect
    _patched = True
    return True


def _self_test() -> None:
    if not hasattr(exp, "_original_plan_focus_bands"):
        exp._original_plan_focus_bands = exp.plan_focus_bands

    class Result:
        json = {
            "res": {
                "dt_polys": [
                    [[10, 20], [110, 20], [110, 55], [10, 55]],
                    [[0, 0], [2, 0], [2, 2], [0, 2]],
                ],
                "dt_scores": [0.9, 0.9],
            }
        }

    boxes = _poly_boxes([Result()], 200, 100)
    assert len(boxes) == 1
    box = boxes[0]
    assert (box.x1, box.y1, box.x2, box.y2) == (10, 20, 110, 55)
    assert box.semantic_type == "text"
    assert not box.safe_to_inpaint

    tall = [
        exp.bubble_mod.BubbleBox(0, 100, 100, 500, 0.9, semantic_type="text"),
        exp.bubble_mod.BubbleBox(0, 3000, 100, 3400, 0.9, semantic_type="text"),
    ]
    bands = _bounded_focus_bands(4096, tall, max_chips=1, merge_gap=64)
    assert bands
    assert max(end - start for start, end in bands) <= _FOCUS_CHIP_MAX_HEIGHT


if __name__ == "__main__":
    _self_test()
    print(
        {
            "status": "pass",
            "patch": "ppocrv6_bounded_v21",
            "model": _GATE_MODEL_NAME,
            "limit_side_len": _GATE_LIMIT,
            "threads": _GATE_CPU_THREADS,
        }
    )
