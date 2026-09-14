from __future__ import annotations

import gc
import hashlib
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np

from app.detector.bubble_detector import BubbleBox
from app.detector.sequential_fast_residue_detector import (
    SequentialFastResidueAdaptiveFocusCombinedTextDetector,
)
from app.parameters import DETECTOR_INPUT_SIZE, DETECTOR_LETTERBOX_VALUE
from app.pipeline import ChapterPipeline
import scripts.benchmark_focus_atlas as base


OUT = Path("benchmark-results/focus-atlas-v3/report.json")
SEAM_GUARD = 24
EDGE_GUARD = 6
SCALE_PROFILES = (0.90, 0.80, 0.70, 0.60, 0.50)
ACTIVE_MIN_SCALE = 1.0


def _best_layout(a: dict, b: dict) -> tuple[str, float]:
    size = int(DETECTOR_INPUT_SIZE)
    aw, ah = int(a["resized_w"]), int(a["resized_h"])
    bw, bh = int(b["resized_w"]), int(b["resized_h"])
    vertical = min(
        1.0,
        size / float(max(1, aw, bw)),
        (size - SEAM_GUARD) / float(max(1, ah + bh)),
    )
    horizontal = min(
        1.0,
        size / float(max(1, ah, bh)),
        (size - SEAM_GUARD) / float(max(1, aw + bw)),
    )
    return (
        ("vertical", float(vertical))
        if vertical >= horizontal
        else ("horizontal", float(horizontal))
    )


def _scaled_slot(slot: dict, factor: float) -> dict:
    x1, y1, x2, y2 = (int(v) for v in slot["chip"])
    source_w = max(1, x2 - x1)
    source_h = max(1, y2 - y1)
    rw = max(1, int(round(float(slot["resized_w"]) * factor)))
    rh = max(1, int(round(float(slot["resized_h"]) * factor)))
    return {
        "chip": (x1, y1, x2, y2),
        "resized_w": rw,
        "resized_h": rh,
        "scale_x": rw / float(source_w),
        "scale_y": rh / float(source_h),
        "relative_scale": float(factor),
    }


def _build_pair_atlas_scaled(
    image: np.ndarray,
    chip_a: tuple[int, int, int, int],
    chip_b: tuple[int, int, int, int],
):
    a0 = base._slot_for_chip(chip_a)
    b0 = base._slot_for_chip(chip_b)
    orientation, factor = _best_layout(a0, b0)
    if factor + 1e-9 < float(ACTIVE_MIN_SCALE):
        return None, None

    a, b = _scaled_slot(a0, factor), _scaled_slot(b0, factor)
    size = int(DETECTOR_INPUT_SIZE)
    aw, ah = int(a["resized_w"]), int(a["resized_h"])
    bw, bh = int(b["resized_w"]), int(b["resized_h"])

    if orientation == "vertical":
        total = ah + SEAM_GUARD + bh
        if total > size or max(aw, bw) > size:
            return None, None
        top = max(0, (size - total) // 2)
        origins = (
            (max(0, (size - aw) // 2), top),
            (max(0, (size - bw) // 2), top + ah + SEAM_GUARD),
        )
    else:
        total = aw + SEAM_GUARD + bw
        if total > size or max(ah, bh) > size:
            return None, None
        left = max(0, (size - total) // 2)
        origins = (
            (left, max(0, (size - ah) // 2)),
            (left + aw + SEAM_GUARD, max(0, (size - bh) // 2)),
        )

    atlas = np.full(
        (size, size, 3),
        DETECTOR_LETTERBOX_VALUE,
        dtype=np.uint8,
    )
    slots = []
    for slot, (ax, ay) in zip((a, b), origins):
        sx1, sy1, sx2, sy2 = slot["chip"]
        crop = image[sy1:sy2, sx1:sx2]
        if crop.size == 0:
            return None, None
        rw, rh = int(slot["resized_w"]), int(slot["resized_h"])
        if ax < 0 or ay < 0 or ax + rw > size or ay + rh > size:
            return None, None
        atlas[ay:ay + rh, ax:ax + rw] = cv2.resize(
            crop,
            (rw, rh),
            interpolation=cv2.INTER_LINEAR,
        )
        slots.append(
            {
                **slot,
                "atlas_rect": (ax, ay, ax + rw, ay + rh),
                "orientation": orientation,
            }
        )
    return atlas, slots


def _map_atlas_boxes_strict(
    boxes: list[BubbleBox],
    slots: list[dict],
) -> tuple[list[BubbleBox], bool]:
    mapped: list[BubbleBox] = []
    for box in boxes:
        cx = (float(box.x1) + float(box.x2)) * 0.5
        cy = (float(box.y1) + float(box.y2)) * 0.5
        owners = [
            slot
            for slot in slots
            if int(slot["atlas_rect"][0]) <= cx <= int(slot["atlas_rect"][2])
            and int(slot["atlas_rect"][1]) <= cy <= int(slot["atlas_rect"][3])
        ]
        if len(owners) != 1:
            return [], False
        slot = owners[0]
        ax1, ay1, ax2, ay2 = (int(v) for v in slot["atlas_rect"])
        if (
            int(box.x1) < ax1 + EDGE_GUARD
            or int(box.y1) < ay1 + EDGE_GUARD
            or int(box.x2) > ax2 - EDGE_GUARD
            or int(box.y2) > ay2 - EDGE_GUARD
        ):
            return [], False

        sx1, sy1, sx2, sy2 = (int(v) for v in slot["chip"])
        scale_x = max(1e-9, float(slot["scale_x"]))
        scale_y = max(1e-9, float(slot["scale_y"]))
        local_x1 = int(math.floor((int(box.x1) - ax1) / scale_x))
        local_y1 = int(math.floor((int(box.y1) - ay1) / scale_y))
        local_x2 = int(math.ceil((int(box.x2) - ax1) / scale_x))
        local_y2 = int(math.ceil((int(box.y2) - ay1) / scale_y))
        crop_w, crop_h = sx2 - sx1, sy2 - sy1
        local_x1 = max(0, min(crop_w, local_x1))
        local_y1 = max(0, min(crop_h, local_y1))
        local_x2 = max(local_x1, min(crop_w, local_x2))
        local_y2 = max(local_y1, min(crop_h, local_y2))
        if local_x2 <= local_x1 or local_y2 <= local_y1:
            return [], False

        mapped_mask = None
        if box.mask is not None:
            mapped_mask = cv2.resize(
                box.mask,
                (local_x2 - local_x1, local_y2 - local_y1),
                interpolation=cv2.INTER_NEAREST,
            )
        mapped.append(
            base.replace(
                box,
                x1=sx1 + local_x1,
                y1=sy1 + local_y1,
                x2=sx1 + local_x2,
                y2=sy1 + local_y2,
                mask=mapped_mask,
            )
        )
    return mapped, True


def _quality(control: dict, candidate: dict) -> dict:
    authority = [
        k for k, value in control["authority_hashes"].items()
        if candidate["authority_hashes"].get(k) != value
    ]
    review = [
        k for k, value in control["review_signatures"].items()
        if candidate["review_signatures"].get(k) != value
    ]
    counts = [
        k for k, value in control["counts"].items()
        if candidate["counts"].get(k) != value
    ]
    return {
        "authority_mask_mismatch_pages": authority,
        "review_signature_mismatch_pages": review,
        "box_count_mismatch_pages": counts,
        "exact": not authority and not review and not counts,
    }


def main() -> None:
    global ACTIVE_MIN_SCALE

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pipeline = ChapterPipeline()
    chapter_id = hashlib.sha256(
        f"focus-atlas-v3-{time.time_ns()}".encode()
    ).hexdigest()[:8]
    manifest = pipeline.download_chapter(base.CHAPTER_URL, chapter_id, workers=2)
    pages = manifest.get("pages", [])
    total = len(pages)
    indices = sorted({
        round(i * (total - 1) / (base.SAMPLE_COUNT - 1))
        for i in range(base.SAMPLE_COUNT)
    })
    paths = {i: Path(pages[i]["original"]) for i in indices}
    warmup_index = next(i for i in range(total) if i not in set(indices))
    warmup_path = Path(pages[warmup_index]["original"])

    control_detector = SequentialFastResidueAdaptiveFocusCombinedTextDetector()
    control = base._run(
        control_detector,
        paths,
        indices,
        warmup_path,
        "control_independent_focus",
    )
    del control_detector
    gc.collect()

    profiles = []
    for min_scale in SCALE_PROFILES:
        ACTIVE_MIN_SCALE = float(min_scale)
        detector = base.FocusAtlasDetector()
        candidate = base._run(
            detector,
            paths,
            indices,
            warmup_path,
            f"scale_aware_atlas_{min_scale:.2f}",
        )
        del detector
        gc.collect()

        quality = _quality(control, candidate)
        speedup = {
            "wall_reduction_pct": base._pct(
                control["wall_ms"], candidate["wall_ms"]
            ),
            "text_model_reduction_pct": base._pct(
                control["text_model_mean_ms"],
                candidate["text_model_mean_ms"],
            ),
            "focus_call_reduction_pct": base._pct(
                float(control["focus_chip_calls"]),
                float(candidate["atlas_model_calls"]),
            ) if control["focus_chip_calls"] else 0.0,
        }
        profiles.append({
            "min_relative_scale": float(min_scale),
            "candidate": candidate,
            "speedup": speedup,
            "quality": quality,
            "promotion_eligible": bool(
                quality["exact"] and candidate["atlas_saved_calls"] > 0
            ),
        })

    eligible = [p for p in profiles if p["promotion_eligible"]]
    best = max(
        eligible,
        key=lambda p: (
            float(p["speedup"]["wall_reduction_pct"]),
            int(p["candidate"]["atlas_saved_calls"]),
            float(p["min_relative_scale"]),
        ),
        default=None,
    )
    report = {
        "chapter_url": base.CHAPTER_URL,
        "chapter_id": chapter_id,
        "total_slices": total,
        "sample_indices": indices,
        "seam_guard": SEAM_GUARD,
        "edge_guard": EDGE_GUARD,
        "scale_profiles": list(SCALE_PROFILES),
        "control": control,
        "profiles": profiles,
        "best_exact_profile": (
            {
                "min_relative_scale": best["min_relative_scale"],
                "speedup": best["speedup"],
                "atlas_pairs": best["candidate"]["atlas_pairs"],
                "atlas_saved_calls": best["candidate"]["atlas_saved_calls"],
            }
            if best is not None else None
        ),
    }
    OUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("FOCUS_ATLAS_V3=" + json.dumps(report, ensure_ascii=False), flush=True)


base._build_pair_atlas = _build_pair_atlas_scaled
base._map_atlas_boxes = _map_atlas_boxes_strict


if __name__ == "__main__":
    main()
