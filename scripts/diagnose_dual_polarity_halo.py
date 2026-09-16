from __future__ import annotations

import json

import cv2
import numpy as np

import app.inpaint.fast_lama_inpainter as fl
from app.detector.bubble_detector import BubbleBox
from app.detector.mask_builder import build_mask
from app.inpaint.fast_lama_inpainter import FastInpainter


def speech_box(x1, y1, x2, y2, mask):
    return BubbleBox(
        x1, y1, x2, y2, 0.95, mask,
        source_model="text_segmenter.onnx",
        semantic_type="speech_bubble",
        mask_source="text_segmenter",
        safe_to_inpaint=True,
        ocr_eligible=True,
        needs_review=False,
        source_role="text_segmenter",
    )


def local_inputs(image, box):
    inp = FastInpainter()
    h, w = image.shape[:2]
    pad = fl._BUBBLE_FASTPATH_PAD
    x1 = max(0, int(box.x1) - pad)
    y1 = max(0, int(box.y1) - pad)
    x2 = min(w, int(box.x2) + pad)
    y2 = min(h, int(box.y2) + pad)
    crop = image[y1:y2, x1:x2]
    local_box = inp._local_box(box, x1, y1)
    authority_mask = build_mask((y2 - y1, x2 - x1), [local_box], crop)
    return inp, crop, authority_mask


def diagnose(name, image, box):
    inp, crop, authority_mask = local_inputs(image, box)
    authority = authority_mask > 127
    ys, xs = np.nonzero(authority)
    data = {
        "name": name,
        "authority_pixels": int(xs.size),
        "refine_result": None,
    }
    if xs.size < 64:
        data["exit"] = "authority_lt_64"
        print("HALO_DIAG=" + json.dumps(data))
        return

    bx1, bx2 = int(xs.min()), int(xs.max()) + 1
    by1, by2 = int(ys.min()), int(ys.max()) + 1
    tight_area = max(1, (bx2 - bx1) * (by2 - by1))
    data["occupancy"] = float(xs.size) / float(tight_area)

    ring = inp._mask_ring(authority_mask, fl._BUBBLE_FASTPATH_RING)
    data["ring_pixels"] = int(np.count_nonzero(ring))
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    gray_f = gray.astype(np.float32, copy=False)
    ring_gray = gray_f[ring]
    data["ring_std"] = float(ring_gray.std()) if ring_gray.size else None
    edges = cv2.Canny(gray, 64, 128, L2gradient=True) > 0
    data["ring_edge_density"] = float(edges[ring].mean()) if np.any(ring) else None
    if crop.ndim == 3 and crop.shape[2] == 3:
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        ring_ab = lab[ring, 1:3].astype(np.float32, copy=False)
        data["ring_ab_std_max"] = float(np.max(ring_ab.std(axis=0))) if ring_ab.size else None

    background = float(np.median(ring_gray))
    inside = gray_f[authority]
    low = float(np.percentile(inside, 10.0))
    high = float(np.percentile(inside, 90.0))
    dark_contrast = background - low
    light_contrast = high - background
    data.update({
        "background": background,
        "inside_p10": low,
        "inside_p90": high,
        "dark_contrast": dark_contrast,
        "light_contrast": light_contrast,
    })

    core_delta = max(fl._STROKE_REFINE_CORE_DELTA_MIN, data["ring_std"] * 3.0)
    weak_delta = max(fl._STROKE_REFINE_WEAK_DELTA_MIN, data["ring_std"] * 1.5)
    if dark_contrast >= light_contrast:
        strong = authority & (gray_f <= background - core_delta)
        weak = authority & (gray_f <= background - weak_delta)
        data["polarity"] = "dark"
    else:
        strong = authority & (gray_f >= background + core_delta)
        weak = authority & (gray_f >= background + weak_delta)
        data["polarity"] = "light"
    data["strong_pixels"] = int(np.count_nonzero(strong))
    data["weak_pixels"] = int(np.count_nonzero(weak))

    count, labels, stats, _ = cv2.connectedComponentsWithStats(weak.astype(np.uint8), 8)
    keep = np.zeros_like(authority_mask, dtype=np.uint8)
    kept_components = 0
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) < 2:
            continue
        component = labels == label
        if np.any(strong & component):
            keep[component] = 255
            kept_components += 1
    data["kept_components"] = kept_components
    data["keep_pixels"] = int(np.count_nonzero(keep))

    halo_delta = max(
        fl._STROKE_REFINE_HALO_DELTA_MIN,
        data["ring_std"] * fl._STROKE_REFINE_HALO_STD_SCALE,
    )
    halo_candidate = authority & (np.abs(gray_f - background) >= halo_delta)
    data["halo_delta"] = float(halo_delta)
    data["halo_candidate_pixels"] = int(np.count_nonzero(halo_candidate))
    grown = keep > 127
    kernel = np.ones((3, 3), dtype=np.uint8)
    growth = []
    for _ in range(fl._STROKE_REFINE_HALO_RADIUS):
        adjacent = cv2.dilate(grown.astype(np.uint8), kernel, iterations=1) > 0
        grown = grown | (halo_candidate & adjacent)
        growth.append(int(np.count_nonzero(grown)))
    data["grown_pixels_by_step"] = growth
    fringe = cv2.dilate(grown.astype(np.uint8), kernel, iterations=1)
    refined = np.where(authority, fringe, 0).astype(np.uint8)
    data["refined_pixels_manual"] = int(np.count_nonzero(refined))
    data["refined_fraction_manual"] = data["refined_pixels_manual"] / float(max(1, data["authority_pixels"]))

    actual = inp._refine_dense_smooth_stroke_mask(crop, authority_mask)
    data["refine_result"] = None if actual is None else int(np.count_nonzero(actual > 127))
    if actual is not None:
        painted = inp._bubble_gradient_fill_from_authority_ring(crop, actual, authority_mask)
        data["authority_ring_fill"] = painted is not None
    else:
        data["authority_ring_fill"] = None
    print("HALO_DIAG=" + json.dumps(data, sort_keys=True))


def case_plain():
    h, w = 220, 320
    yy, xx = np.mgrid[:h, :w]
    background = np.empty((h, w, 3), dtype=np.uint8)
    background[..., 0] = np.clip(218 + xx * 0.035 + yy * 0.020, 0, 255)
    background[..., 1] = np.clip(222 + xx * 0.032 + yy * 0.018, 0, 255)
    background[..., 2] = np.clip(228 + xx * 0.028 + yy * 0.015, 0, 255)
    image = background.copy()
    dense = np.zeros((120, 220), dtype=np.uint8)
    dense[8:112, 8:212] = 255
    box = speech_box(50, 45, 270, 165, dense)
    image[75:86, 85:235] = 5
    image[105:117, 100:220] = 8
    image[134:146, 125:200] = 3
    return image, box


def case_outline():
    h, w = 220, 320
    yy, xx = np.mgrid[:h, :w]
    background = np.empty((h, w, 3), dtype=np.uint8)
    background[..., 0] = np.clip(210 + xx * 0.040 + yy * 0.022, 0, 255)
    background[..., 1] = np.clip(216 + xx * 0.036 + yy * 0.020, 0, 255)
    background[..., 2] = np.clip(224 + xx * 0.030 + yy * 0.017, 0, 255)
    image = background.copy()
    dense = np.zeros((120, 220), dtype=np.uint8)
    dense[8:112, 8:212] = 255
    box = speech_box(50, 45, 270, 165, dense)
    core = np.zeros((h, w), dtype=np.uint8)
    core[76:84, 84:236] = 255
    core[106:114, 102:220] = 255
    core[136:144, 126:202] = 255
    outline = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    outline_only = (outline > 0) & (core == 0)
    image[outline_only] = 252
    image[core > 0] = 4
    return image, box


for name, factory in (("plain", case_plain), ("outline", case_outline)):
    image, box = factory()
    diagnose(name, image, box)
