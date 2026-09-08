from pathlib import Path
import re


def replace_once(source: str, old: str, new: str, label: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(f"{label}: expected one exact match, got {source.count(old)}")
    return source.replace(old, new, 1)


# F09: Smart Fill must reject chromatic structure that grayscale can hide.
parameters_path = Path("app/parameters.py")
parameters = parameters_path.read_text(encoding="utf-8")
parameters = replace_once(
    parameters,
    '''SMART_FILL_BLACK_EDGE_DENSITY_MAX = _env_float(
    "MANGA_SMART_FILL_BLACK_EDGE_DENSITY_MAX", 0.002, minimum=0.0, maximum=1.0
)
''',
    '''SMART_FILL_BLACK_EDGE_DENSITY_MAX = _env_float(
    "MANGA_SMART_FILL_BLACK_EDGE_DENSITY_MAX", 0.002, minimum=0.0, maximum=1.0
)
SMART_FILL_CHROMA_STD_MAX = _env_float(
    "MANGA_SMART_FILL_CHROMA_STD_MAX", 12.0, minimum=0.0, maximum=128.0
)
''',
    "smart-fill chroma parameter",
)
parameters_path.write_text(parameters, encoding="utf-8")

lama_path = Path("app/inpaint/lama_inpainter.py")
lama = lama_path.read_text(encoding="utf-8")
lama = replace_once(
    lama,
    "    SMART_FILL_CANNY_LOW,\n",
    "    SMART_FILL_CANNY_LOW,\n    SMART_FILL_CHROMA_STD_MAX,\n",
    "lama parameter import",
)
lama = replace_once(
    lama,
    '''        ring_gray = gray[ring]
        context_gray = gray[context]
        ring_pixels = crop[ring]
        context_std = float(context_gray.std())

        edges = cv2.Canny(
''',
    '''        ring_gray = gray[ring]
        context_gray = gray[context]
        ring_pixels = crop[ring]
        context_std = float(context_gray.std())

        # Grayscale flatness is not color flatness. Equal-luminance artwork can
        # have almost zero gray variance while carrying strong chromatic edges.
        # Smart Fill is allowed only when both the clean ring and wider context
        # are chromatically stable in CIELAB a/b channels.
        chroma_safe = True
        if crop.ndim == 3 and crop.shape[2] == 3:
            lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
            ring_ab = lab[ring, 1:3].astype(np.float32, copy=False)
            context_ab = lab[context, 1:3].astype(np.float32, copy=False)
            ring_chroma_std = float(np.max(ring_ab.std(axis=0))) if ring_ab.size else 0.0
            context_chroma_std = float(np.max(context_ab.std(axis=0))) if context_ab.size else 0.0
            chroma_safe = bool(
                ring_chroma_std <= SMART_FILL_CHROMA_STD_MAX
                and context_chroma_std <= SMART_FILL_CHROMA_STD_MAX
            )

        edges = cv2.Canny(
''',
    "smart-fill chroma guard",
)
for anchor in (
    "            white_ratio >= SMART_FILL_WHITE_RATIO_MIN\n",
    "            black_ratio >= SMART_FILL_BLACK_RATIO_MIN\n",
    "            SMART_FILL_MIDTONE_MIN <= median_gray <= SMART_FILL_MIDTONE_MAX\n",
):
    lama = replace_once(
        lama,
        anchor,
        "            chroma_safe\n            and " + anchor.lstrip(),
        f"chroma condition {anchor.strip()}",
    )

# F11: any source support pixel must survive model-space downscale.
insert_before = '''    def _lama_fill_single(self, crop: np.ndarray, local_mask: np.ndarray) -> np.ndarray:
'''
resize_helper = '''    @staticmethod
    def _resize_mask_preserve_support(
        mask: np.ndarray,
        width: int,
        height: int,
    ) -> np.ndarray:
        """Resize a binary mask without dropping thin glyph support.

        Downscaling uses area coverage and promotes any positive contribution;
        upscaling remains nearest-neighbour so no interpolated authority is
        invented between disconnected source pixels.
        """
        width, height = max(1, int(width)), max(1, int(height))
        source = (mask > 127).astype(np.uint8) * 255
        src_h, src_w = source.shape[:2]
        if (src_w, src_h) == (width, height):
            return source
        if width < src_w or height < src_h:
            coverage = cv2.resize(
                source.astype(np.float32),
                (width, height),
                interpolation=cv2.INTER_AREA,
            )
            return (coverage > 0.0).astype(np.uint8) * 255
        return cv2.resize(source, (width, height), interpolation=cv2.INTER_NEAREST)

'''
lama = replace_once(lama, insert_before, resize_helper + insert_before, "support resize helper")
lama = lama.replace(
    "mask_resized = cv2.resize(local_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)",
    "mask_resized = self._resize_mask_preserve_support(local_mask, new_w, new_h)",
)
if lama.count("_resize_mask_preserve_support(local_mask, new_w, new_h)") != 2:
    raise RuntimeError("expected dynamic and fixed mask resize replacements")
lama = replace_once(
    lama,
    '''        if feather:
            alpha = (local_mask > 127).astype(np.float32)
            k = MANUAL_FEATHER_RADIUS * 2 + 1
            alpha = cv2.GaussianBlur(alpha, (k, k), 0)
            alpha = np.clip(alpha, 0.0, 1.0)[:, :, None]
            blended = painted.astype(np.float32) * alpha + original_crop.astype(np.float32) * (1.0 - alpha)
''',
    '''        if feather:
            core = local_mask > 127
            alpha = core.astype(np.float32)
            if MANUAL_FEATHER_RADIUS > 0:
                k = MANUAL_FEATHER_RADIUS * 2 + 1
                feathered = cv2.GaussianBlur(alpha, (k, k), 0)
                # Feather only the explicit margin; approved glyph support stays
                # fully opaque so text cannot ghost back through the composite.
                alpha = np.where(core, 1.0, feathered)
            alpha = np.clip(alpha, 0.0, 1.0)[:, :, None]
            blended = painted.astype(np.float32) * alpha + original_crop.astype(np.float32) * (1.0 - alpha)
''',
    "opaque feather core",
)
lama_path.write_text(lama, encoding="utf-8")

# F08: MSER/shape heuristics remain review evidence, never independent erase authority.
recovery_path = Path("app/detector/recovery.py")
recovery = recovery_path.read_text(encoding="utf-8")
recovery = replace_once(
    recovery,
    '''            safe = bool(
                MSER_SAFE_MASK_RATIO_MIN <= ratio <= MSER_SAFE_MASK_RATIO_MAX
                and not self._mask_component_spans_crop(mask)
                and page_ratio <= MSER_SAFE_PAGE_AREA_RATIO_MAX
                and len(cluster) >= MSER_SAFE_CLUSTER_MIN_REGIONS
            )
            if safe:
                candidate = replace(
                    candidate,
                    mask=mask,
                    mask_source="opencv_mser",
                    safe_to_inpaint=True,
                    ocr_eligible=True,
                    needs_review=False,
                    confidence=MSER_SAFE_CONFIDENCE,
                )
''',
    '''            review_mask_valid = bool(
                MSER_SAFE_MASK_RATIO_MIN <= ratio <= MSER_SAFE_MASK_RATIO_MAX
                and not self._mask_component_spans_crop(mask)
                and page_ratio <= MSER_SAFE_PAGE_AREA_RATIO_MAX
                and len(cluster) >= MSER_SAFE_CLUSTER_MIN_REGIONS
            )
            if review_mask_valid:
                candidate = replace(
                    candidate,
                    mask=mask,
                    mask_source="opencv_mser_review",
                    safe_to_inpaint=False,
                    ocr_eligible=True,
                    needs_review=True,
                    confidence=MSER_SAFE_CONFIDENCE,
                )
''',
    "MSER authority revoke",
)
recovery = replace_once(
    recovery,
    '''        verification_set = existing + out
        if verification_set and all(
            box.safe_to_inpaint and not box.needs_review for box in verification_set
        ):
            residual = self._residual_line_candidates(
                np.asarray(boxes), (h, w), verification_set
            )
''',
    '''        # Review-only MSER proposals must not suppress additional review
        # recovery. Residual lines are permitted when independent existing
        # evidence is already verified; they still never gain erase authority.
        verification_set = existing + out
        if existing and all(
            box.safe_to_inpaint and not box.needs_review for box in existing
        ):
            residual = self._residual_line_candidates(
                np.asarray(boxes), (h, w), verification_set
            )
''',
    "MSER residual review policy",
)
recovery_path.write_text(recovery, encoding="utf-8")

mask_path = Path("app/detector/mask_builder.py")
mask_builder = mask_path.read_text(encoding="utf-8")
mask_builder = replace_once(
    mask_builder,
    '''AUTO_DESTRUCTIVE_MASK_SOURCES = frozenset(
    {"text_segmenter", "bubble_flat_contrast", "opencv_mser"}
)
''',
    '''# MSER recovery remains review evidence until independent text evidence or
# an explicit user action authorizes cleanup. It is intentionally absent here.
AUTO_DESTRUCTIVE_MASK_SOURCES = frozenset(
    {"text_segmenter", "bubble_flat_contrast"}
)
''',
    "destructive source set",
)
mask_path.write_text(mask_builder, encoding="utf-8")

# The supplied bubble model has detection-only authority. Make the old flat-
# bubble mask fallback explicitly disabled for that role instead of leaving an
# apparently-live branch that requires a mask the contract cannot provide.
combined_path = Path("app/detector/combined_detector.py")
combined = combined_path.read_text(encoding="utf-8")
combined = replace_once(
    combined,
    '''        if (
            box.semantic_type != "speech_bubble"
''',
    '''        if box.source_role == "bubble_detector":
            return None
        if (
            box.semantic_type != "speech_bubble"
''',
    "flat bubble detection-only disable",
)
combined_path.write_text(combined, encoding="utf-8")

# Keep the static destructive-authority gate aligned with the stronger runtime contract.
authority_path = Path("scripts/inpaint_authority_sanity.py")
authority = authority_path.read_text(encoding="utf-8")
authority = replace_once(
    authority,
    '''        '"bubble_flat_contrast"',
        '"opencv_mser"',
        "def is_destructive_box_authorized(box: BubbleBox) -> bool:",
''',
    '''        '"bubble_flat_contrast"',
        "MSER recovery remains review evidence",
        "def is_destructive_box_authorized(box: BubbleBox) -> bool:",
''',
    "authority source markers",
)
authority_path.write_text(authority, encoding="utf-8")

sanity = r'''from __future__ import annotations

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

from app.detector.bubble_detector import BubbleBox
from app.detector.combined_detector import CombinedTextDetector
from app.detector.mask_builder import AUTO_DESTRUCTIVE_MASK_SOURCES
from app.detector.recovery import SecondaryTextRecovery
from app.inpaint.lama_inpainter import Inpainter


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def mser_authority_check():
    check("opencv_mser" not in AUTO_DESTRUCTIVE_MASK_SOURCES, "MSER still listed as destructive authority")
    recovery = SecondaryTextRecovery()
    primitives = np.array(
        [[30, 35, 12, 12], [44, 35, 12, 12], [58, 35, 12, 12]],
        dtype=np.int32,
    )
    recovery._extract_primitives = lambda image, gray: primitives
    def seed(crop):
        mask = np.zeros(crop.shape[:2], np.uint8)
        mask[2::4, 2::4] = 255
        return mask
    recovery._seed_mask = seed
    image = np.full((160, 240, 3), 180, np.uint8)
    result = recovery.detect(image, existing=[])
    check(bool(result), "MSER review fixture produced no evidence")
    check(all(not box.safe_to_inpaint for box in result), "MSER independently gained erase authority")
    check(all(box.needs_review for box in result), "MSER evidence stopped requiring review")
    check(all(box.ocr_eligible for box in result), "MSER review evidence lost OCR eligibility")
    print("phase6 MSER review-only authority PASS")


def chromatic_smart_fill_check():
    mask = np.zeros((64, 64), np.uint8)
    mask[24:40, 24:40] = 255
    neutral = np.full((64, 64, 3), 100, np.uint8)
    check(Inpainter._smart_fill_color(neutral, mask) is not None, "neutral Smart Fill regressed")

    # BGR red-ish and green-ish values both convert to gray ~=60 in OpenCV, so
    # grayscale variance/edges alone cannot see this chromatic artwork pattern.
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


def flat_bubble_contract_check():
    image = np.full((100, 140, 3), 255, np.uint8)
    mask = np.full((50, 90), 255, np.uint8)
    proposal = BubbleBox(
        20, 20, 110, 70, 0.95, mask,
        source_model="bubble_yolo.onnx",
        class_name="text_bubble",
        semantic_type="speech_bubble",
        source_role="bubble_detector",
    )
    recovered = CombinedTextDetector._flat_bubble_text_fallback(image, proposal, 140, 100)
    check(recovered is None, "detection-only bubble proposal gained synthetic destructive mask")
    print("phase6 detection-only flat-bubble fallback disabled PASS")


mser_authority_check()
chromatic_smart_fill_check()
mask_support_and_feather_check()
flat_bubble_contract_check()
print("backend mask safety sanity: phase 6 deterministic subset PASS")
print("NOTE: stroke-mask promotion remains blocked until a real human-reviewed truth-mask manifest exists")
'''
Path("scripts/backend_phase6_sanity.py").write_text(sanity, encoding="utf-8")
print("phase6 safety patch applied")
