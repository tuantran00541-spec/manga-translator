from __future__ import annotations

import sys
import types
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if "onnxruntime" not in sys.modules:
    sys.modules["onnxruntime"] = types.ModuleType("onnxruntime")

from app.inpaint.lama_inpainter import Inpainter


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def chromatic_smart_fill_check():
    mask = np.zeros((64, 64), np.uint8)
    mask[24:40, 24:40] = 255
    neutral = np.full((64, 64, 3), 100, np.uint8)
    check(Inpainter._smart_fill_color(neutral, mask) is not None, "neutral Smart Fill regressed")

    chromatic = np.empty((64, 64, 3), np.uint8)
    color_a = np.array([0, 0, 200], np.uint8)
    color_b = np.array([0, 102, 0], np.uint8)
    checker = (np.indices((64, 64)).sum(axis=0) % 2) == 0
    chromatic[checker] = color_a
    chromatic[~checker] = color_b
    gray = cv2.cvtColor(chromatic, cv2.COLOR_BGR2GRAY)
    check(float(gray.std()) < 1.5, "chromatic fixture is not grayscale-flat")
    check(Inpainter._smart_fill_color(chromatic, mask) is None, "equal-luminance chromatic artwork accepted by Smart Fill")
    print("phase6 chromatic Smart Fill negative PASS")


def mask_support_and_feather_check():
    thin = np.zeros((64, 64), np.uint8)
    thin[31, 31] = 255
    reduced = Inpainter._resize_mask_preserve_support(thin, 8, 8)
    check(np.any(reduced > 127), "one-pixel glyph support vanished during downscale")

    painter = Inpainter()
    painter._ensure_session = lambda: None
    painter._lama_fill_single = lambda crop, mask: np.full_like(crop, 220)
    original = np.full((40, 40, 3), 20, np.uint8)
    mask = np.zeros((40, 40), np.uint8)
    mask[15:25, 15:25] = 255
    output = painter._lama_fill(original.copy(), original.copy(), mask, (0, 0, 40, 40), feather=True)
    check(np.all(output[mask > 127] == 220), "manual feather leaked original glyph core")
    check(np.array_equal(output[:8, :8], original[:8, :8]), "manual feather changed pixels outside its bounded margin")
    print("phase6 support-preserving resize/opaque feather PASS")


chromatic_smart_fill_check()
mask_support_and_feather_check()
print("backend mask safety sanity: phase 6 deterministic subset PASS")
print("NOTE: stroke-mask promotion remains blocked until a real human-reviewed truth-mask manifest exists")
