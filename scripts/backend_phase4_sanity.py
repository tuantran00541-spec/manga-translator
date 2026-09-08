from __future__ import annotations

import sys
import threading
import types
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if "onnxruntime" not in sys.modules:
    sys.modules["onnxruntime"] = types.ModuleType("onnxruntime")

from app.detector.adaptive_focus_detector import (
    FOCUS_MAX_CHIPS,
    FOCUS_MAX_SOURCE_SIDE,
    _proposal_has_full_text,
    plan_focus_chips,
)
from app.detector.bubble_detector import BubbleBox
from app.detector.recovery import SecondaryTextRecovery


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def scheduling_checks():
    giant = BubbleBox(0, 0, 900, 4096, 0.8, semantic_type="free_text")
    chips, deferred = plan_focus_chips(4096, 900, [giant])
    check(1 <= len(chips) <= FOCUS_MAX_CHIPS, "focus call budget")
    check(bool(deferred), "giant proposal must expose deferred budget regions")
    check(all(max(x2-x1, y2-y1) <= FOCUS_MAX_SOURCE_SIDE for x1,y1,x2,y2 in chips), "giant proposal defeated source resolution bound")

    proposal = BubbleBox(100, 100, 600, 600, 0.8, semantic_type="speech_bubble")
    tiny = BubbleBox(320, 320, 330, 330, 0.8, semantic_type="text")
    check(not _proposal_has_full_text(proposal, [tiny]), "tiny presence incorrectly marked complete")
    dialogue = BubbleBox(120, 120, 360, 260, 0.8, semantic_type="speech_bubble")
    text = BubbleBox(180, 155, 300, 225, 0.8, semantic_type="text")
    check(_proposal_has_full_text(dialogue, [text]), "ordinary dialogue completeness regressed")

    image = np.full((4096, 900, 3), 128, np.uint8)
    image[2500:2750, 120:780] = (255, 0, 255)
    cv2.putText(image, "SFX", (180, 2680), cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 255, 0), 16, cv2.LINE_AA)
    fallback, fallback_deferred = plan_focus_chips(4096, 900, [], fallback_image=image)
    check(len(fallback) == 1 and not fallback_deferred, "proposal-independent fallback budget")
    check(any(y1 < 2750 and y2 > 2500 for _x1,y1,_x2,y2 in fallback), "chromatic free-text signal not selected")
    print("phase4 focus budget/completeness/fallback checks PASS")


def extraction_cache_check():
    recovery = SecondaryTextRecovery()
    calls = {"count": 0}
    class FakeMSER:
        def detectRegions(self, gray):
            calls["count"] += 1
            return [], np.array([[10, 10, 12, 8], [25, 10, 12, 8], [40, 10, 12, 8]], dtype=np.int32)
    recovery._mser = FakeMSER()
    recovery._mser_lock = threading.Lock()
    image = np.zeros((120, 160, 3), np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    first = recovery._extract_primitives(image, gray)
    second = recovery._extract_primitives(image, gray)
    check(calls["count"] == 1, "MSER primitives extracted more than once")
    check(np.array_equal(first, second), "cached primitives changed")
    other = image.copy()
    recovery._extract_primitives(other, gray)
    check(calls["count"] == 2, "cache ownership leaked across source images")
    print("phase4 MSER primitive ownership/cache check PASS")


scheduling_checks()
extraction_cache_check()
print("backend adaptive scheduling sanity: phase 4 PASS")
