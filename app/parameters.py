from __future__ import annotations

"""Central runtime/quality tuning for Manga Translator.

This module is the single source of truth for tunable numeric/boolean parameters.
Defaults intentionally preserve the behavior that existed before this file was
introduced. Environment overrides use the ``MANGA_*`` prefix and are clamped to
safe ranges so a typo cannot silently create nonsensical geometry or unbounded
resource use.

Security limits stay in :mod:`app.security`; filesystem/model paths stay in
:mod:`app.config`. Values such as pixel channel maxima (255), binary-mask cutoffs
(127), HTTP status codes and array indices are invariants, not tuning knobs.
"""

import os
from numbers import Number
from typing import Final


_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return bool(default)
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    return bool(default)


def _env_int(
    name: str,
    default: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
    odd: bool = False,
) -> int:
    raw = os.getenv(name, "").strip()
    try:
        value = int(raw) if raw else int(default)
    except ValueError:
        value = int(default)
    if minimum is not None:
        value = max(int(minimum), value)
    if maximum is not None:
        value = min(int(maximum), value)
    if odd and value % 2 == 0:
        candidate = value + 1
        if maximum is not None and candidate > maximum:
            candidate = value - 1
        value = max(int(minimum or 1), candidate)
    return int(value)


def _env_float(
    name: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    raw = os.getenv(name, "").strip()
    try:
        value = float(raw) if raw else float(default)
    except ValueError:
        value = float(default)
    if minimum is not None:
        value = max(float(minimum), value)
    if maximum is not None:
        value = min(float(maximum), value)
    return float(value)


# ---------------------------------------------------------------------------
# Detector / segmentation
# ---------------------------------------------------------------------------

BUBBLE_DESTRUCTIVE_CONF_THRESHOLD = _env_float(
    "MANGA_BUBBLE_DESTRUCTIVE_CONF", 0.40, minimum=0.0, maximum=1.0
)
# Proposal confidence is intentionally separate from destructive authority.
# v0.2 historically hid this as min(BUBBLE_CONF_THRESHOLD, 0.12).
BUBBLE_PROPOSAL_CONF_THRESHOLD = _env_float(
    "MANGA_BUBBLE_PROPOSAL_CONF", 0.12, minimum=0.0, maximum=1.0
)
TEXT_CONF_THRESHOLD = _env_float(
    "MANGA_TEXT_CONF", 0.20, minimum=0.0, maximum=1.0
)
BUBBLE_IOU_THRESHOLD = _env_float(
    "MANGA_BUBBLE_NMS_IOU", 0.30, minimum=0.0, maximum=1.0
)
DETECTOR_FINAL_NMS_IOU = _env_float(
    "MANGA_DETECTOR_FINAL_NMS_IOU", 0.35, minimum=0.0, maximum=1.0
)
DETECTOR_STABLE_ID_IOU_MIN = _env_float(
    "MANGA_DETECTOR_STABLE_ID_IOU_MIN", 0.50, minimum=0.0, maximum=1.0
)
DETECTOR_INPUT_SIZE = _env_int(
    "MANGA_DETECTOR_INPUT_SIZE", 1024, minimum=256, maximum=4096
)
DETECTOR_WINDOW_OVERLAP = _env_int(
    "MANGA_DETECTOR_WINDOW_OVERLAP", 200, minimum=0, maximum=2048
)
DETECTOR_TALL_IMAGE_FACTOR = _env_float(
    "MANGA_DETECTOR_TALL_IMAGE_FACTOR", 1.50, minimum=1.0, maximum=4.0
)
DETECTOR_MAX_BOX_WIDTH_RATIO = _env_float(
    "MANGA_DETECTOR_MAX_BOX_WIDTH_RATIO", 0.97, minimum=0.1, maximum=1.0
)
DETECTOR_MAX_BOX_AREA_RATIO = _env_float(
    "MANGA_DETECTOR_MAX_BOX_AREA_RATIO", 0.35, minimum=0.01, maximum=1.0
)
DETECTOR_MAX_ASPECT_RATIO = _env_float(
    "MANGA_DETECTOR_MAX_ASPECT_RATIO", 25.0, minimum=1.0, maximum=100.0
)
DETECTOR_CONFIDENCE_MAX: Final[float] = 0.999998
DETECTOR_TTA_ENABLED = _env_bool("MANGA_DETECTOR_TTA", False)
DETECTOR_TTA_SMALL_SCALE = _env_float(
    "MANGA_DETECTOR_TTA_SMALL_SCALE", 0.85, minimum=0.25, maximum=1.0
)
DETECTOR_TTA_MIN_SIDE = _env_int(
    "MANGA_DETECTOR_TTA_MIN_SIDE", 10, minimum=1, maximum=256
)
DETECTOR_MIN_BOX_SIDE = _env_float(
    "MANGA_DETECTOR_MIN_BOX_SIDE", 4.0, minimum=1.0, maximum=128.0
)
DETECTOR_LETTERBOX_VALUE = _env_int(
    "MANGA_DETECTOR_LETTERBOX_VALUE", 114, minimum=0, maximum=255
)
DETECTOR_MASK_THRESHOLD = _env_float(
    "MANGA_DETECTOR_MASK_THRESHOLD", 0.50, minimum=0.0, maximum=1.0
)
DETECTOR_NMS_SCORE_FLOOR = _env_float(
    "MANGA_DETECTOR_NMS_SCORE_FLOOR", 0.05, minimum=0.0, maximum=1.0
)
DETECTION_CONTENT_STD_MIN = _env_float(
    "MANGA_DETECTION_CONTENT_STD_MIN", 24.0, minimum=0.0, maximum=128.0
)

# Bubble/text grouping and safety heuristics.
BUBBLE_GROUP_PAD_X = _env_int(
    "MANGA_BUBBLE_GROUP_PAD_X", 20, minimum=0, maximum=128
)
BUBBLE_GROUP_PAD_Y = _env_int(
    "MANGA_BUBBLE_GROUP_PAD_Y", 8, minimum=0, maximum=128
)
BUBBLE_TEXT_OVERLAP_MIN = _env_float(
    "MANGA_BUBBLE_TEXT_OVERLAP_MIN", 0.50, minimum=0.0, maximum=1.0
)
DETECTOR_TALL_SPLIT_HEIGHT_THRESHOLD = _env_int(
    "MANGA_DETECTOR_TALL_SPLIT_HEIGHT_THRESHOLD", 45, minimum=1, maximum=4096
)
DETECTOR_TALL_SPLIT_BACKGROUND_PERCENTILE = _env_float(
    "MANGA_DETECTOR_TALL_SPLIT_BACKGROUND_PERCENTILE",
    90.0,
    minimum=0.0,
    maximum=100.0,
)
DETECTOR_TALL_SPLIT_CONTRAST_DELTA = _env_float(
    "MANGA_DETECTOR_TALL_SPLIT_CONTRAST_DELTA",
    2.0,
    minimum=0.0,
    maximum=255.0,
)
DETECTOR_TALL_SPLIT_LINE_HEIGHT_MIN = _env_int(
    "MANGA_DETECTOR_TALL_SPLIT_LINE_HEIGHT_MIN", 4, minimum=1, maximum=256
)
DETECTOR_TALL_SPLIT_LINE_PADDING_MAX = _env_int(
    "MANGA_DETECTOR_TALL_SPLIT_LINE_PADDING_MAX", 3, minimum=0, maximum=128
)
DETECTOR_TALL_SPLIT_HORIZONTAL_PADDING = _env_int(
    "MANGA_DETECTOR_TALL_SPLIT_HORIZONTAL_PADDING", 10, minimum=0, maximum=256
)
FREE_TEXT_SINGLE_PAD_X = _env_int(
    "MANGA_FREE_TEXT_SINGLE_PAD_X", 20, minimum=0, maximum=128
)
FREE_TEXT_SINGLE_PAD_Y = _env_int(
    "MANGA_FREE_TEXT_SINGLE_PAD_Y", 6, minimum=0, maximum=128
)
FREE_TEXT_GROUP_PAD_X = _env_int(
    "MANGA_FREE_TEXT_GROUP_PAD_X", 20, minimum=0, maximum=128
)
FREE_TEXT_GROUP_PAD_Y = _env_int(
    "MANGA_FREE_TEXT_GROUP_PAD_Y", 8, minimum=0, maximum=128
)
FREE_TEXT_X_GAP = _env_int(
    "MANGA_FREE_TEXT_X_GAP", 35, minimum=0, maximum=256
)
FREE_TEXT_Y_GAP = _env_int(
    "MANGA_FREE_TEXT_Y_GAP", 40, minimum=0, maximum=256
)
FREE_TEXT_Y_ALIGNMENT = _env_int(
    "MANGA_FREE_TEXT_Y_ALIGNMENT", 45, minimum=0, maximum=256
)
FREE_TEXT_LINE_OVERLAP_MIN = _env_float(
    "MANGA_FREE_TEXT_LINE_OVERLAP_MIN", 0.50, minimum=0.0, maximum=1.0
)
FREE_TEXT_CLUSTER_SPLIT_COUNT = _env_int(
    "MANGA_FREE_TEXT_CLUSTER_SPLIT_COUNT", 3, minimum=1, maximum=50
)
FREE_TEXT_CLUSTER_HEIGHT_FACTOR = _env_float(
    "MANGA_FREE_TEXT_CLUSTER_HEIGHT_FACTOR", 4.0, minimum=1.0, maximum=20.0
)
FREE_TEXT_GROUP_HEIGHT_FACTOR = _env_float(
    "MANGA_FREE_TEXT_GROUP_HEIGHT_FACTOR", 3.0, minimum=1.0, maximum=20.0
)

FLAT_BUBBLE_BACKGROUND_RATIO_MIN = _env_float(
    "MANGA_FLAT_BUBBLE_BACKGROUND_RATIO_MIN", 0.70, minimum=0.0, maximum=1.0
)
FLAT_BUBBLE_MIN_SIDE = _env_int(
    "MANGA_FLAT_BUBBLE_MIN_SIDE", 9, minimum=1, maximum=512
)
FLAT_BUBBLE_TEXT_RATIO_MIN = _env_float(
    "MANGA_FLAT_BUBBLE_TEXT_RATIO_MIN", 0.005, minimum=0.0, maximum=1.0
)
FLAT_BUBBLE_TEXT_RATIO_MAX = _env_float(
    "MANGA_FLAT_BUBBLE_TEXT_RATIO_MAX", 0.28, minimum=0.0, maximum=1.0
)
FLAT_BUBBLE_PAGE_AREA_MAX = _env_float(
    "MANGA_FLAT_BUBBLE_PAGE_AREA_MAX", 0.18, minimum=0.001, maximum=1.0
)
FLAT_BUBBLE_WHITE_MIN = _env_int(
    "MANGA_FLAT_BUBBLE_WHITE_MIN", 205, minimum=0, maximum=255
)
FLAT_BUBBLE_WHITE_MEDIAN_MIN = _env_int(
    "MANGA_FLAT_BUBBLE_WHITE_MEDIAN_MIN", 215, minimum=0, maximum=255
)
FLAT_BUBBLE_DARK_TEXT_MAX = _env_int(
    "MANGA_FLAT_BUBBLE_DARK_TEXT_MAX", 180, minimum=0, maximum=255
)
FLAT_BUBBLE_BLACK_MAX = _env_int(
    "MANGA_FLAT_BUBBLE_BLACK_MAX", 50, minimum=0, maximum=255
)
FLAT_BUBBLE_BLACK_MEDIAN_MAX = _env_int(
    "MANGA_FLAT_BUBBLE_BLACK_MEDIAN_MAX", 45, minimum=0, maximum=255
)
FLAT_BUBBLE_LIGHT_TEXT_MIN = _env_int(
    "MANGA_FLAT_BUBBLE_LIGHT_TEXT_MIN", 78, minimum=0, maximum=255
)
FLAT_BUBBLE_INSET_RATIO = _env_float(
    "MANGA_FLAT_BUBBLE_INSET_RATIO", 0.04, minimum=0.0, maximum=0.25
)
FLAT_BUBBLE_INSET_MIN = _env_int(
    "MANGA_FLAT_BUBBLE_INSET_MIN", 2, minimum=0, maximum=64
)
FLAT_BUBBLE_INSET_MAX = _env_int(
    "MANGA_FLAT_BUBBLE_INSET_MAX", 10, minimum=1, maximum=128
)
FLAT_BUBBLE_INTERIOR_AREA_RATIO_MIN = _env_float(
    "MANGA_FLAT_BUBBLE_INTERIOR_AREA_RATIO_MIN", 0.12, minimum=0.0, maximum=1.0
)
FLAT_BUBBLE_INTERIOR_PIXELS_MIN = _env_int(
    "MANGA_FLAT_BUBBLE_INTERIOR_PIXELS_MIN", 64, minimum=1, maximum=100000
)
FLAT_BUBBLE_STROKE_CLOSE_KERNEL = _env_int(
    "MANGA_FLAT_BUBBLE_STROKE_CLOSE_KERNEL", 3, minimum=1, maximum=15, odd=True
)
FLAT_BUBBLE_TEXT_BBOX_PAD = _env_int(
    "MANGA_FLAT_BUBBLE_TEXT_BBOX_PAD", 4, minimum=0, maximum=64
)

# ---------------------------------------------------------------------------
# Secondary MSER recovery
# ---------------------------------------------------------------------------

MSER_DELTA = _env_int("MANGA_MSER_DELTA", 5, minimum=1, maximum=50)
MSER_MIN_AREA = _env_int("MANGA_MSER_MIN_AREA", 18, minimum=1, maximum=100000)
MSER_MAX_AREA = _env_int(
    "MANGA_MSER_MAX_AREA", 120000, minimum=100, maximum=20_000_000
)
MSER_REGION_MIN_SIDE = _env_int(
    "MANGA_MSER_REGION_MIN_SIDE", 4, minimum=1, maximum=128
)
MSER_REGION_MAX_WIDTH_RATIO = _env_float(
    "MANGA_MSER_REGION_MAX_WIDTH_RATIO", 0.90, minimum=0.01, maximum=1.0
)
MSER_REGION_MAX_HEIGHT_RATIO = _env_float(
    "MANGA_MSER_REGION_MAX_HEIGHT_RATIO", 0.65, minimum=0.01, maximum=1.0
)
MSER_REGION_AREA_MIN = _env_int(
    "MANGA_MSER_REGION_AREA_MIN", 24, minimum=1, maximum=1_000_000
)
MSER_REGION_AREA_RATIO_MAX = _env_float(
    "MANGA_MSER_REGION_AREA_RATIO_MAX", 0.20, minimum=0.001, maximum=1.0
)
MSER_SEED_CONTRAST_DELTA = _env_float(
    "MANGA_MSER_SEED_CONTRAST_DELTA", 22.0, minimum=0.0, maximum=255.0
)
MSER_SEED_CANNY_LOW = _env_int(
    "MANGA_MSER_SEED_CANNY_LOW", 45, minimum=0, maximum=255
)
MSER_SEED_CANNY_HIGH = _env_int(
    "MANGA_MSER_SEED_CANNY_HIGH", 120, minimum=0, maximum=255
)
MSER_SAFE_MASK_RATIO_MIN = _env_float(
    "MANGA_MSER_SAFE_MASK_RATIO_MIN", 0.015, minimum=0.0, maximum=1.0
)
MSER_SAFE_MASK_RATIO_MAX = _env_float(
    "MANGA_MSER_SAFE_MASK_RATIO_MAX", 0.42, minimum=0.0, maximum=1.0
)
MSER_SAFE_COMPONENT_SPAN_RATIO_MAX = _env_float(
    "MANGA_MSER_SAFE_COMPONENT_SPAN_RATIO_MAX",
    0.95,
    minimum=0.1,
    maximum=1.0,
)
MSER_SAFE_PAGE_AREA_RATIO_MAX = _env_float(
    "MANGA_MSER_SAFE_PAGE_AREA_RATIO_MAX", 0.035, minimum=0.0, maximum=1.0
)
MSER_SAFE_CLUSTER_MIN_REGIONS = _env_int(
    "MANGA_MSER_SAFE_CLUSTER_MIN_REGIONS", 3, minimum=1, maximum=100
)
MSER_EXISTING_IOU_SKIP = _env_float(
    "MANGA_MSER_EXISTING_IOU_SKIP", 0.55, minimum=0.0, maximum=1.0
)
MSER_CONTAINED_SAFE_SKIP_COUNT = _env_int(
    "MANGA_MSER_CONTAINED_SAFE_SKIP_COUNT", 2, minimum=1, maximum=20
)
MSER_PAGE_CLUSTER_SKIP_RATIO = _env_float(
    "MANGA_MSER_PAGE_CLUSTER_SKIP_RATIO", 0.45, minimum=0.0, maximum=1.0
)

# Residual-line verifier. These are separate from primary MSER mask authority:
# equal defaults must not imply shared semantics when one side is tuned later.
MSER_LINE_OVERLAP_MIN = _env_float(
    "MANGA_MSER_LINE_OVERLAP_MIN", 0.45, minimum=0.0, maximum=1.0
)
MSER_LINE_GAP_MIN = _env_float(
    "MANGA_MSER_LINE_GAP_MIN", 18.0, minimum=0.0, maximum=512.0
)
MSER_LINE_GAP_HEIGHT_FACTOR = _env_float(
    "MANGA_MSER_LINE_GAP_HEIGHT_FACTOR", 1.6, minimum=0.0, maximum=20.0
)
MSER_RESIDUAL_MAX_WIDTH_RATIO = _env_float(
    "MANGA_MSER_RESIDUAL_MAX_WIDTH_RATIO", 0.16, minimum=0.0, maximum=1.0
)
MSER_RESIDUAL_MAX_HEIGHT_RATIO = _env_float(
    "MANGA_MSER_RESIDUAL_MAX_HEIGHT_RATIO", 0.10, minimum=0.0, maximum=1.0
)
MSER_RESIDUAL_AREA_RATIO_MAX = _env_float(
    "MANGA_MSER_RESIDUAL_AREA_RATIO_MAX", 0.015, minimum=0.0, maximum=1.0
)
MSER_RESIDUAL_ASPECT_MIN = _env_float(
    "MANGA_MSER_RESIDUAL_ASPECT_MIN", 0.07, minimum=0.001, maximum=10.0
)
MSER_RESIDUAL_ASPECT_MAX = _env_float(
    "MANGA_MSER_RESIDUAL_ASPECT_MAX", 5.0, minimum=0.01, maximum=100.0
)
MSER_RESIDUAL_GRID_CELL = _env_int(
    "MANGA_MSER_RESIDUAL_GRID_CELL", 64, minimum=4, maximum=1024
)
MSER_RESIDUAL_GRID_Y_RADIUS = _env_int(
    "MANGA_MSER_RESIDUAL_GRID_Y_RADIUS", 2, minimum=0, maximum=32
)
MSER_RESIDUAL_DISTINCT_X_BUCKET = _env_int(
    "MANGA_MSER_RESIDUAL_DISTINCT_X_BUCKET", 8, minimum=1, maximum=256
)
MSER_RESIDUAL_DISTINCT_X_MIN = _env_int(
    "MANGA_MSER_RESIDUAL_DISTINCT_X_MIN", 5, minimum=1, maximum=100
)
MSER_RESIDUAL_MIN_WIDTH = _env_int(
    "MANGA_MSER_RESIDUAL_MIN_WIDTH", 70, minimum=1, maximum=4096
)
MSER_RESIDUAL_MIN_HEIGHT = _env_int(
    "MANGA_MSER_RESIDUAL_MIN_HEIGHT", 8, minimum=1, maximum=4096
)
MSER_RESIDUAL_MAX_LINE_HEIGHT_RATIO = _env_float(
    "MANGA_MSER_RESIDUAL_MAX_LINE_HEIGHT_RATIO", 0.12, minimum=0.0, maximum=1.0
)
MSER_RESIDUAL_MAX_BBOX_AREA_RATIO = _env_float(
    "MANGA_MSER_RESIDUAL_MAX_BBOX_AREA_RATIO", 0.05, minimum=0.0, maximum=1.0
)
MSER_RESIDUAL_GROUP_ASPECT_MIN = _env_float(
    "MANGA_MSER_RESIDUAL_GROUP_ASPECT_MIN", 1.8, minimum=0.01, maximum=100.0
)
MSER_RECOVERY_PAD = _env_int(
    "MANGA_MSER_RECOVERY_PAD", 6, minimum=0, maximum=256
)
MSER_RESIDUAL_REVIEW_CONFIDENCE = _env_float(
    "MANGA_MSER_RESIDUAL_REVIEW_CONFIDENCE", 0.18, minimum=0.0, maximum=1.0
)
MSER_RESIDUAL_EXISTING_IOU_SKIP = _env_float(
    "MANGA_MSER_RESIDUAL_EXISTING_IOU_SKIP", 0.25, minimum=0.0, maximum=1.0
)
MSER_RESIDUAL_SAFE_CENTER_PAD = _env_int(
    "MANGA_MSER_RESIDUAL_SAFE_CENTER_PAD", 8, minimum=0, maximum=256
)
MSER_CLUSTER_NEAR_X_FACTOR = _env_float(
    "MANGA_MSER_CLUSTER_NEAR_X_FACTOR", 1.8, minimum=0.0, maximum=20.0
)
MSER_CLUSTER_NEAR_Y_FACTOR = _env_float(
    "MANGA_MSER_CLUSTER_NEAR_Y_FACTOR", 1.3, minimum=0.0, maximum=20.0
)
MSER_CLUSTER_MIN_REGIONS = _env_int(
    "MANGA_MSER_CLUSTER_MIN_REGIONS", 2, minimum=1, maximum=100
)
MSER_CLUSTER_MIN_WIDTH = _env_int(
    "MANGA_MSER_CLUSTER_MIN_WIDTH", 12, minimum=1, maximum=4096
)
MSER_CLUSTER_MIN_HEIGHT = _env_int(
    "MANGA_MSER_CLUSTER_MIN_HEIGHT", 10, minimum=1, maximum=4096
)
MSER_REVIEW_CONFIDENCE = _env_float(
    "MANGA_MSER_REVIEW_CONFIDENCE", 0.20, minimum=0.0, maximum=1.0
)
MSER_SAFE_CONFIDENCE = _env_float(
    "MANGA_MSER_SAFE_CONFIDENCE", 0.35, minimum=0.0, maximum=1.0
)
MSER_RESIDUAL_FINAL_IOU_SKIP = _env_float(
    "MANGA_MSER_RESIDUAL_FINAL_IOU_SKIP", 0.35, minimum=0.0, maximum=1.0
)

# ---------------------------------------------------------------------------
# Mask building
# ---------------------------------------------------------------------------

MASK_DILATE_KERNEL_SIZE = _env_int(
    "MANGA_MASK_DILATE_KERNEL_SIZE", 7, minimum=1, maximum=31, odd=True
)
MASK_ADAPTIVE_DILATE_KERNEL_SIZE = _env_int(
    "MANGA_MASK_ADAPTIVE_DILATE_KERNEL_SIZE", 9, minimum=1, maximum=31, odd=True
)
MASK_ADAPTIVE_BORDER_STD_THRESHOLD = _env_float(
    "MANGA_MASK_ADAPTIVE_BORDER_STD_THRESHOLD", 18.0, minimum=0.0, maximum=128.0
)
MASK_EXPAND = _env_int("MANGA_MANUAL_MASK_EXPAND", 8, minimum=0, maximum=128)
MANUAL_MASK_THRESHOLD = _env_int(
    "MANGA_MANUAL_MASK_THRESHOLD", 10, minimum=0, maximum=254
)
MANUAL_CONFIDENCE_SENTINEL: Final[float] = 1.0

# ---------------------------------------------------------------------------
# Slicer / seam ownership
# ---------------------------------------------------------------------------

SLICE_TARGET_HEIGHT = _env_int(
    "MANGA_SLICE_TARGET_HEIGHT", 1400, minimum=256, maximum=8192
)
SLICE_SEARCH_WINDOW = _env_int(
    "MANGA_SLICE_SEARCH_WINDOW", 180, minimum=0, maximum=2048
)
SLICE_MIN_HEIGHT = _env_int(
    "MANGA_SLICE_MIN_HEIGHT", 500, minimum=128, maximum=8192
)
SLICE_MAX_HEIGHT = _env_int(
    "MANGA_SLICE_MAX_HEIGHT", 1536, minimum=256, maximum=8192
)
SLICE_SAFE_CUT_BAND = _env_int(
    "MANGA_SLICE_SAFE_CUT_BAND", 12, minimum=1, maximum=256
)
SLICE_MAX_SAFE_SEARCH_EXPANSION = _env_int(
    "MANGA_SLICE_MAX_SAFE_SEARCH_EXPANSION", 360, minimum=0, maximum=4096
)
SLICE_FALLBACK_BAND = _env_int(
    "MANGA_SLICE_FALLBACK_BAND", 18, minimum=1, maximum=512
)
SLICE_OVERLAP_CONTEXT = _env_int(
    "MANGA_SLICE_OVERLAP_CONTEXT", 384, minimum=0, maximum=2048
)
SLICE_CONTENT_CANNY_LOW = _env_int(
    "MANGA_SLICE_CONTENT_CANNY_LOW", 30, minimum=0, maximum=255
)
SLICE_CONTENT_CANNY_HIGH = _env_int(
    "MANGA_SLICE_CONTENT_CANNY_HIGH", 120, minimum=0, maximum=255
)
SLICE_BACKGROUND_DISTANCE = _env_int(
    "MANGA_SLICE_BACKGROUND_DISTANCE", 18, minimum=0, maximum=255
)
SLICE_CONTENT_SCORE_WEIGHT = _env_float(
    "MANGA_SLICE_CONTENT_SCORE_WEIGHT", 2.0, minimum=0.0, maximum=100.0
)
SLICE_CLOSE_KERNEL_SIZE = _env_int(
    "MANGA_SLICE_CLOSE_KERNEL_SIZE", 31, minimum=1, maximum=101, odd=True
)
SLICE_CONTOUR_MIN_WIDTH = _env_int(
    "MANGA_SLICE_CONTOUR_MIN_WIDTH", 15, minimum=1, maximum=512
)
SLICE_CONTOUR_MIN_HEIGHT = _env_int(
    "MANGA_SLICE_CONTOUR_MIN_HEIGHT", 15, minimum=1, maximum=512
)
SLICE_CONTOUR_PAD_Y = _env_int(
    "MANGA_SLICE_CONTOUR_PAD_Y", 40, minimum=0, maximum=512
)
SLICE_FALLBACK_TOLERANCE_RATIO = _env_float(
    "MANGA_SLICE_FALLBACK_TOLERANCE_RATIO", 0.08, minimum=0.0, maximum=1.0
)

# ---------------------------------------------------------------------------
# Inpaint / smart-fill / LaMa
# ---------------------------------------------------------------------------

USE_DYNAMIC_LAMA = _env_bool("MANGA_USE_DYNAMIC_LAMA", True)
INPAINT_SIZE = _env_int(
    "MANGA_INPAINT_SIZE", 512, minimum=128, maximum=4096
)
DYNAMIC_LAMA_MAX_SINGLE_CROP_DIM = _env_int(
    "MANGA_DYNAMIC_LAMA_MAX_SINGLE_CROP_DIM", 512, minimum=128, maximum=4096
)
INPAINT_CLUSTER_PADDING = _env_int(
    "MANGA_INPAINT_CLUSTER_PADDING", 35, minimum=0, maximum=256
)
INPAINT_CROP_PADDING = _env_int(
    "MANGA_INPAINT_CROP_PADDING", 35, minimum=0, maximum=256
)
INPAINT_CLUSTER_MAX_DIM = _env_int(
    "MANGA_INPAINT_CLUSTER_MAX_DIM", 600, minimum=64, maximum=4096
)
INPAINT_CLUSTER_SPLIT_COUNT = _env_int(
    "MANGA_INPAINT_CLUSTER_SPLIT_COUNT", 3, minimum=1, maximum=100
)
INPAINT_CLUSTER_SPLIT_HEIGHT_FACTOR = _env_float(
    "MANGA_INPAINT_CLUSTER_SPLIT_HEIGHT_FACTOR", 4.0, minimum=1.0, maximum=20.0
)
INPAINT_CLUSTER_LINE_OVERLAP_MIN = _env_float(
    "MANGA_INPAINT_CLUSTER_LINE_OVERLAP_MIN", 0.5, minimum=0.0, maximum=1.0
)
INPAINT_CLUSTER_GROUP_HEIGHT_FACTOR = _env_float(
    "MANGA_INPAINT_CLUSTER_GROUP_HEIGHT_FACTOR", 3.0, minimum=1.0, maximum=20.0
)
INPAINT_CROP_LONG_ASPECT_THRESHOLD = _env_float(
    "MANGA_INPAINT_CROP_LONG_ASPECT_THRESHOLD", 1.8, minimum=1.0, maximum=20.0
)
MANUAL_CROP_PADDING = _env_int(
    "MANGA_MANUAL_CROP_PADDING", 72, minimum=0, maximum=512
)
MANUAL_MIN_DILATION = _env_int(
    "MANGA_MANUAL_MIN_DILATION", 9, minimum=1, maximum=63, odd=True
)
MANUAL_MAX_DILATION = _env_int(
    "MANGA_MANUAL_MAX_DILATION", 15, minimum=1, maximum=127, odd=True
)
MANUAL_DILATION_SCALE = _env_float(
    "MANGA_MANUAL_DILATION_SCALE", 0.025, minimum=0.0, maximum=1.0
)
MANUAL_FEATHER_RADIUS = _env_int(
    "MANGA_MANUAL_FEATHER_RADIUS", 3, minimum=0, maximum=31
)
MANUAL_TILE_OVERLAP = _env_int(
    "MANGA_MANUAL_TILE_OVERLAP", 64, minimum=0, maximum=1024
)
FIXED_LAMA_TILE_ASPECT = _env_float(
    "MANGA_FIXED_LAMA_TILE_ASPECT", 1.6, minimum=1.0, maximum=20.0
)
FIXED_LAMA_CONCURRENT_INFERENCE = _env_bool(
    "MANGA_FIXED_LAMA_CONCURRENT_INFERENCE", False
)
SMART_FILL_CLEAN_RING_MARGIN = _env_int(
    "MANGA_SMART_FILL_CLEAN_RING_MARGIN", 6, minimum=1, maximum=64
)
SMART_FILL_CONTEXT_MARGIN_FACTOR = _env_int(
    "MANGA_SMART_FILL_CONTEXT_MARGIN_FACTOR", 3, minimum=1, maximum=16
)
SMART_FILL_RING_PIXELS_MIN = _env_int(
    "MANGA_SMART_FILL_RING_PIXELS_MIN", 32, minimum=1, maximum=100000
)
SMART_FILL_CANNY_LOW = _env_int(
    "MANGA_SMART_FILL_CANNY_LOW", 64, minimum=0, maximum=255
)
SMART_FILL_CANNY_HIGH = _env_int(
    "MANGA_SMART_FILL_CANNY_HIGH", 128, minimum=0, maximum=255
)
SMART_FILL_WHITE_LEVEL = _env_int(
    "MANGA_SMART_FILL_WHITE_LEVEL", 215, minimum=0, maximum=255
)
SMART_FILL_BLACK_LEVEL = _env_int(
    "MANGA_SMART_FILL_BLACK_LEVEL", 35, minimum=0, maximum=255
)
SMART_FILL_WHITE_RATIO_MIN = _env_float(
    "MANGA_SMART_FILL_WHITE_RATIO_MIN", 0.97, minimum=0.0, maximum=1.0
)
SMART_FILL_BLACK_RATIO_MIN = _env_float(
    "MANGA_SMART_FILL_BLACK_RATIO_MIN", 0.995, minimum=0.0, maximum=1.0
)
SMART_FILL_WHITE_STD_MAX = _env_float(
    "MANGA_SMART_FILL_WHITE_STD_MAX", 8.0, minimum=0.0, maximum=128.0
)
SMART_FILL_BLACK_STD_MAX = _env_float(
    "MANGA_SMART_FILL_BLACK_STD_MAX", 2.5, minimum=0.0, maximum=128.0
)
SMART_FILL_MIDTONE_STD_MAX = _env_float(
    "MANGA_SMART_FILL_MIDTONE_STD_MAX", 5.0, minimum=0.0, maximum=128.0
)
SMART_FILL_FULL_STD_MAX = _env_float(
    "MANGA_SMART_FILL_FULL_STD_MAX", 12.0, minimum=0.0, maximum=128.0
)
SMART_FILL_EDGE_DENSITY_MAX = _env_float(
    "MANGA_SMART_FILL_EDGE_DENSITY_MAX", 0.01, minimum=0.0, maximum=1.0
)
SMART_FILL_BLACK_EDGE_DENSITY_MAX = _env_float(
    "MANGA_SMART_FILL_BLACK_EDGE_DENSITY_MAX", 0.002, minimum=0.0, maximum=1.0
)
SMART_FILL_MIDTONE_MIN = _env_float(
    "MANGA_SMART_FILL_MIDTONE_MIN", 50.0, minimum=0.0, maximum=255.0
)
SMART_FILL_MIDTONE_MAX = _env_float(
    "MANGA_SMART_FILL_MIDTONE_MAX", 205.0, minimum=0.0, maximum=255.0
)
FIXED_LAMA_SESSION_MAX_RUNS = _env_int(
    "MANGA_FIXED_LAMA_SESSION_MAX_RUNS", 4, minimum=1, maximum=10000
)
FIXED_LAMA_RECYCLE_MEMORY_LIMIT_BYTES = _env_int(
    "MANGA_FIXED_LAMA_RECYCLE_MEMORY_LIMIT_BYTES",
    6 * 1024**3,
    minimum=256 * 1024**2,
    maximum=1024**4,
)

# ---------------------------------------------------------------------------
# Runtime concurrency / ONNX Runtime
# ---------------------------------------------------------------------------

PIPELINE_DEFAULT_WORKERS = _env_int(
    "MANGA_PIPELINE_DEFAULT_WORKERS", 2, minimum=1, maximum=32
)
PIPELINE_SLICE_WORKER_LIMIT = _env_int(
    "MANGA_PIPELINE_SLICE_WORKER_LIMIT", 8, minimum=1, maximum=64
)
PIPELINE_PROCESS_WORKER_LIMIT = _env_int(
    "MANGA_PIPELINE_PROCESS_WORKER_LIMIT", 2, minimum=1, maximum=16
)
MANIFEST_LOCK_TIMEOUT_SECONDS = _env_float(
    "MANGA_MANIFEST_LOCK_TIMEOUT_SECONDS", 30.0, minimum=1.0, maximum=600.0
)
PAGE_LOCK_TIMEOUT_SECONDS = _env_float(
    "MANGA_PAGE_LOCK_TIMEOUT_SECONDS", 60.0, minimum=1.0, maximum=600.0
)
STALE_TEMP_MAX_AGE_SECONDS = _env_float(
    "MANGA_STALE_TEMP_MAX_AGE_SECONDS", 3600.0, minimum=60.0, maximum=604800.0
)
ORT_HIGH_CPU_THRESHOLD = _env_int(
    "MANGA_ORT_HIGH_CPU_THRESHOLD", 8, minimum=1, maximum=256
)
ORT_MEDIUM_CPU_THRESHOLD = _env_int(
    "MANGA_ORT_MEDIUM_CPU_THRESHOLD", 4, minimum=1, maximum=256
)
ORT_HIGH_CPU_THREADS = _env_int(
    "MANGA_ORT_HIGH_CPU_THREADS", 4, minimum=1, maximum=64
)
ORT_MEDIUM_CPU_THREADS = _env_int(
    "MANGA_ORT_MEDIUM_CPU_THREADS", 2, minimum=1, maximum=64
)
ORT_INTER_OP_THREADS = _env_int(
    "MANGA_ORT_INTER_OP_THREADS", 1, minimum=1, maximum=64
)

# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------

OCR_IMAGE_CACHE_MB = _env_int(
    "MANGA_OCR_IMAGE_CACHE_MB", 128, minimum=0, maximum=1024
)
OCR_JOB_CONCURRENCY_LIMIT = _env_int(
    "MANGA_OCR_JOB_CONCURRENCY_LIMIT", 2, minimum=1, maximum=8
)
OCR_JOB_ACTIVE_LIMIT = _env_int(
    "MANGA_OCR_JOB_ACTIVE_LIMIT", 2, minimum=1, maximum=8
)
OCR_MASK_CROP_PADDING = _env_int(
    "MANGA_OCR_MASK_CROP_PADDING", 12, minimum=0, maximum=256
)
OCR_BOX_CROP_PADDING = _env_int(
    "MANGA_OCR_BOX_CROP_PADDING", 20, minimum=0, maximum=256
)
OCR_REJECT_CONFIDENCE = _env_float(
    "MANGA_OCR_REJECT_CONFIDENCE", 0.35, minimum=0.0, maximum=1.0
)
OCR_REVIEW_CONFIDENCE = _env_float(
    "MANGA_OCR_REVIEW_CONFIDENCE", 0.55, minimum=0.0, maximum=1.0
)
OCR_EN_SCRIPT_MISMATCH_RATIO = _env_float(
    "MANGA_OCR_EN_SCRIPT_MISMATCH_RATIO", 0.50, minimum=0.0, maximum=1.0
)
OCR_UNEXPECTED_LATIN_RATIO = _env_float(
    "MANGA_OCR_UNEXPECTED_LATIN_RATIO", 0.85, minimum=0.0, maximum=1.0
)
OCR_SYMBOL_NOISE_RATIO = _env_float(
    "MANGA_OCR_SYMBOL_NOISE_RATIO", 0.45, minimum=0.0, maximum=1.0
)
OCR_PADDLE_MIN_SIDE = _env_int(
    "MANGA_OCR_PADDLE_MIN_SIDE", 32, minimum=1, maximum=512
)
OCR_PADDLE_MAX_UPSCALE = _env_float(
    "MANGA_OCR_PADDLE_MAX_UPSCALE", 4.0, minimum=1.0, maximum=16.0
)
OCR_CENTER_ANCHOR_DISTANCE_MAX = _env_float(
    "MANGA_OCR_CENTER_ANCHOR_DISTANCE_MAX", 0.22, minimum=0.0, maximum=1.5
)
OCR_CENTER_BAND_OVERLAP_MIN = _env_float(
    "MANGA_OCR_CENTER_BAND_OVERLAP_MIN", 0.30, minimum=0.0, maximum=1.0
)
OCR_CENTER_AXIS_FACTOR = _env_float(
    "MANGA_OCR_CENTER_AXIS_FACTOR", 0.55, minimum=0.0, maximum=3.0
)
OCR_CENTER_DISTANCE_MARGIN = _env_float(
    "MANGA_OCR_CENTER_DISTANCE_MARGIN", 0.08, minimum=0.0, maximum=1.0
)
OCR_READING_HORIZONTAL_ASPECT = _env_float(
    "MANGA_OCR_READING_HORIZONTAL_ASPECT", 1.20, minimum=1.0, maximum=10.0
)
OCR_READING_VERTICAL_ASPECT = _env_float(
    "MANGA_OCR_READING_VERTICAL_ASPECT", 0.90, minimum=0.1, maximum=10.0
)
OCR_RUBY_SIZE_FACTOR = _env_float(
    "MANGA_OCR_RUBY_SIZE_FACTOR", 0.72, minimum=0.1, maximum=1.0
)
OCR_MAIN_SIZE_FACTOR = _env_float(
    "MANGA_OCR_MAIN_SIZE_FACTOR", 0.80, minimum=0.1, maximum=2.0
)
OCR_MAIN_SIZE_QUANTILE = _env_float(
    "MANGA_OCR_MAIN_SIZE_QUANTILE", 0.75, minimum=0.0, maximum=1.0
)
OCR_MAIN_LINE_ASPECT_MIN = _env_float(
    "MANGA_OCR_MAIN_LINE_ASPECT_MIN", 1.80, minimum=1.0, maximum=20.0
)
OCR_RUBY_OVERLAP_MIN = _env_float(
    "MANGA_OCR_RUBY_OVERLAP_MIN", 0.25, minimum=0.0, maximum=1.0
)
OCR_RUBY_MIN_SIDE = _env_float(
    "MANGA_OCR_RUBY_MIN_SIDE", 8.0, minimum=1.0, maximum=256.0
)
OCR_RUBY_DISTANCE_MIN_FACTOR = _env_float(
    "MANGA_OCR_RUBY_DISTANCE_MIN_FACTOR", 0.20, minimum=0.0, maximum=10.0
)
OCR_HORIZONTAL_RUBY_DISTANCE_MAX_FACTOR = _env_float(
    "MANGA_OCR_HORIZONTAL_RUBY_DISTANCE_MAX_FACTOR",
    1.20,
    minimum=0.0,
    maximum=10.0,
)
OCR_VERTICAL_RUBY_DISTANCE_MAX_FACTOR = _env_float(
    "MANGA_OCR_VERTICAL_RUBY_DISTANCE_MAX_FACTOR",
    1.35,
    minimum=0.0,
    maximum=10.0,
)
OCR_RUBY_FILTER_MIN_KEEP_DIVISOR = _env_int(
    "MANGA_OCR_RUBY_FILTER_MIN_KEEP_DIVISOR", 3, minimum=1, maximum=100
)
OCR_ROW_TOLERANCE_FACTOR = _env_float(
    "MANGA_OCR_ROW_TOLERANCE_FACTOR", 0.65, minimum=0.0, maximum=5.0
)
OCR_ROW_SIZE_QUANTILE = _env_float(
    "MANGA_OCR_ROW_SIZE_QUANTILE", 0.60, minimum=0.0, maximum=1.0
)
OCR_ROW_TOLERANCE_MIN = _env_float(
    "MANGA_OCR_ROW_TOLERANCE_MIN", 8.0, minimum=0.0, maximum=256.0
)
OCR_COLUMN_TOLERANCE_FACTOR = _env_float(
    "MANGA_OCR_COLUMN_TOLERANCE_FACTOR", 0.85, minimum=0.0, maximum=5.0
)
OCR_COLUMN_TOLERANCE_MIN = _env_float(
    "MANGA_OCR_COLUMN_TOLERANCE_MIN", 10.0, minimum=0.0, maximum=256.0
)

# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

MIN_FONT_SIZE = _env_int("MANGA_MIN_FONT_SIZE", 6, minimum=1, maximum=512)
MAX_FONT_SIZE = _env_int("MANGA_MAX_FONT_SIZE", 48, minimum=1, maximum=1024)
FONT_CACHE_SIZE = _env_int("MANGA_FONT_CACHE_SIZE", 256, minimum=1, maximum=4096)
RENDER_AUTO_TEXT_DARK_BG_THRESHOLD = _env_float(
    "MANGA_RENDER_AUTO_TEXT_DARK_BG_THRESHOLD", 135.0, minimum=0.0, maximum=255.0
)
RENDER_LINE_HEIGHT_FACTOR = _env_float(
    "MANGA_RENDER_LINE_HEIGHT_FACTOR", 1.18, minimum=0.5, maximum=3.0
)
RENDER_DEFAULT_PADDING = _env_int(
    "MANGA_RENDER_DEFAULT_PADDING", 6, minimum=0, maximum=128
)
RENDER_PADDING_RATIO_MAX = _env_float(
    "MANGA_RENDER_PADDING_RATIO_MAX", 0.08, minimum=0.0, maximum=0.5
)
RENDER_AUTO_STROKE_WIDTH = _env_int(
    "MANGA_RENDER_AUTO_STROKE_WIDTH", 2, minimum=0, maximum=12
)
RENDER_STROKE_WIDTH_MAX = _env_int(
    "MANGA_RENDER_STROKE_WIDTH_MAX", 12, minimum=0, maximum=64
)

# ---------------------------------------------------------------------------
# Visual QC
# ---------------------------------------------------------------------------

VISUAL_QC_REGION_MARGIN = _env_int(
    "MANGA_VISUAL_QC_REGION_MARGIN", 64, minimum=0, maximum=1024
)
VISUAL_QC_MERGE_GAP = _env_int(
    "MANGA_VISUAL_QC_MERGE_GAP", 32, minimum=0, maximum=1024
)
VISUAL_QC_DEEP_AREA_RATIO = _env_float(
    "MANGA_VISUAL_QC_DEEP_AREA_RATIO", 0.35, minimum=0.0, maximum=1.0
)
VISUAL_QC_MANUAL_COMPONENT_AREA_MIN = _env_int(
    "MANGA_VISUAL_QC_MANUAL_COMPONENT_AREA_MIN", 9, minimum=1, maximum=100000
)
VISUAL_QC_MANUAL_MASK_THRESHOLD = _env_int(
    "MANGA_VISUAL_QC_MANUAL_MASK_THRESHOLD", 10, minimum=0, maximum=255
)
VISUAL_QC_GLOBAL_BATCH_SIZE = _env_int(
    "MANGA_VISUAL_QC_GLOBAL_BATCH_SIZE", 2, minimum=1, maximum=8
)
VISUAL_QC_REGION_BATCH_SIZE = _env_int(
    "MANGA_VISUAL_QC_REGION_BATCH_SIZE", 4, minimum=1, maximum=8
)
VISUAL_QC_PAIR_BATCH_SIZE = _env_int(
    "MANGA_VISUAL_QC_PAIR_BATCH_SIZE", 2, minimum=1, maximum=8
)
VISUAL_QC_JOB_CONCURRENCY_LIMIT: Final[int] = 4
VISUAL_QC_JOB_CONCURRENCY = _env_int(
    "MANGA_VISUAL_QC_JOB_CONCURRENCY",
    2,
    minimum=1,
    maximum=VISUAL_QC_JOB_CONCURRENCY_LIMIT,
)

# ---------------------------------------------------------------------------
# Translation/network defaults
# ---------------------------------------------------------------------------

DEEPSEEK_API_URL = os.getenv(
    "MANGA_DEEPSEEK_API_URL", "https://api.deepseek.com/chat/completions"
).strip() or "https://api.deepseek.com/chat/completions"
DEEPSEEK_TRANSLATION_MODEL = os.getenv(
    "MANGA_DEEPSEEK_TRANSLATION_MODEL", "deepseek-v4-flash"
).strip() or "deepseek-v4-flash"
DEEPSEEK_PRICING_VERSION = os.getenv(
    "MANGA_DEEPSEEK_PRICING_VERSION", "2026-08"
).strip() or "2026-08"
DEEPSEEK_INPUT_CACHE_HIT_USD_PER_M = _env_float(
    "MANGA_DEEPSEEK_INPUT_CACHE_HIT_USD_PER_M", 0.0028, minimum=0.0
)
DEEPSEEK_INPUT_CACHE_MISS_USD_PER_M = _env_float(
    "MANGA_DEEPSEEK_INPUT_CACHE_MISS_USD_PER_M", 0.14, minimum=0.0
)
DEEPSEEK_OUTPUT_USD_PER_M = _env_float(
    "MANGA_DEEPSEEK_OUTPUT_USD_PER_M", 0.28, minimum=0.0
)

TRANSLATION_MAX_TOKENS = _env_int(
    "MANGA_TRANSLATION_MAX_TOKENS", 32768, minimum=256, maximum=131072
)
TRANSLATION_CONNECT_TIMEOUT_SECONDS = _env_float(
    "MANGA_TRANSLATION_CONNECT_TIMEOUT_SECONDS", 10.0, minimum=1.0, maximum=300.0
)
TRANSLATION_READ_TIMEOUT_SECONDS = _env_float(
    "MANGA_TRANSLATION_READ_TIMEOUT_SECONDS", 120.0, minimum=1.0, maximum=1800.0
)
TRANSLATION_PREFLIGHT_MIN_TOKENS = _env_int(
    "MANGA_TRANSLATION_PREFLIGHT_MIN_TOKENS", 1000, minimum=1, maximum=100000
)
TRANSLATION_PREFLIGHT_PROMPT_OVERHEAD = _env_int(
    "MANGA_TRANSLATION_PREFLIGHT_PROMPT_OVERHEAD", 2500, minimum=0, maximum=100000
)
TRANSLATION_PREFLIGHT_OUTPUT_MULTIPLIER = _env_float(
    "MANGA_TRANSLATION_PREFLIGHT_OUTPUT_MULTIPLIER", 2.0, minimum=0.1, maximum=20.0
)

REMOTE_CONNECT_TIMEOUT_SECONDS = _env_float(
    "MANGA_REMOTE_CONNECT_TIMEOUT_SECONDS", 30.0, minimum=1.0, maximum=300.0
)
REMOTE_CHUNK_BYTES = _env_int(
    "MANGA_REMOTE_CHUNK_BYTES", 64 * 1024, minimum=1024, maximum=4 * 1024 * 1024
)
DOWNLOAD_WORKER_LIMIT = _env_int(
    "MANGA_DOWNLOAD_WORKERS", 4, minimum=1, maximum=8
)
DOWNLOAD_STATIC_MIN_DECLARED_WIDTH = _env_int(
    "MANGA_DOWNLOAD_STATIC_MIN_DECLARED_WIDTH", 240, minimum=0, maximum=8192
)
DOWNLOAD_JS_MIN_IMAGE_WIDTH = _env_int(
    "MANGA_DOWNLOAD_JS_MIN_IMAGE_WIDTH", 300, minimum=0, maximum=8192
)
DOWNLOAD_JS_MAX_SCROLL_ROUNDS = _env_int(
    "MANGA_DOWNLOAD_JS_MAX_SCROLL_ROUNDS", 30, minimum=1, maximum=1000
)
DOWNLOAD_JS_STABLE_ROUNDS = _env_int(
    "MANGA_DOWNLOAD_JS_STABLE_ROUNDS", 3, minimum=1, maximum=50
)
DOWNLOAD_JS_SCROLL_WAIT_MS = _env_int(
    "MANGA_DOWNLOAD_JS_SCROLL_WAIT_MS", 400, minimum=0, maximum=10000
)
DOWNLOAD_JS_NAVIGATION_TIMEOUT_MS = _env_int(
    "MANGA_DOWNLOAD_JS_NAVIGATION_TIMEOUT_MS", 60000, minimum=1000, maximum=300000
)


# Cross-parameter constraints. Individual env parsers clamp each value, while
# these guards keep related values coherent.
DETECTOR_WINDOW_OVERLAP = min(DETECTOR_WINDOW_OVERLAP, DETECTOR_INPUT_SIZE - 1)
SLICE_MIN_HEIGHT = min(SLICE_MIN_HEIGHT, SLICE_MAX_HEIGHT)
SLICE_TARGET_HEIGHT = min(
    SLICE_MAX_HEIGHT, max(SLICE_MIN_HEIGHT, SLICE_TARGET_HEIGHT)
)
MANUAL_MIN_DILATION = min(MANUAL_MIN_DILATION, MANUAL_MAX_DILATION)
MIN_FONT_SIZE = min(MIN_FONT_SIZE, MAX_FONT_SIZE)
FLAT_BUBBLE_TEXT_RATIO_MIN = min(
    FLAT_BUBBLE_TEXT_RATIO_MIN, FLAT_BUBBLE_TEXT_RATIO_MAX
)
SMART_FILL_MIDTONE_MIN = min(SMART_FILL_MIDTONE_MIN, SMART_FILL_MIDTONE_MAX)
OCR_REJECT_CONFIDENCE = min(OCR_REJECT_CONFIDENCE, OCR_REVIEW_CONFIDENCE)
OCR_RUBY_DISTANCE_MIN_FACTOR = min(
    OCR_RUBY_DISTANCE_MIN_FACTOR,
    OCR_HORIZONTAL_RUBY_DISTANCE_MAX_FACTOR,
    OCR_VERTICAL_RUBY_DISTANCE_MAX_FACTOR,
)
MSER_RESIDUAL_ASPECT_MIN = min(MSER_RESIDUAL_ASPECT_MIN, MSER_RESIDUAL_ASPECT_MAX)


def parameter_snapshot() -> dict[str, Number | bool | str]:
    """Return current tuning values for diagnostics/reproducibility."""
    result: dict[str, Number | bool | str] = {}
    for name, value in globals().items():
        if not name.isupper() or name.startswith("_"):
            continue
        if isinstance(value, (bool, int, float, str)):
            result[name] = value
    return dict(sorted(result.items()))
