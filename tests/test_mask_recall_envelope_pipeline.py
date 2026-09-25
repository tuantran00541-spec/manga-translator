import numpy as np

from app.mask_recall_envelope_v2_pipeline import CompleteFlatEnvelopeMaskRecallPipeline
from app.parameters import (
    DETECTOR_INPUT_SIZE,
    DETECTOR_RESIDUE_VERIFY_MAX_SOURCE_SIDE,
    DETECTOR_TEXT_MASK_DECODE_PAD,
)


def test_residue_envelope_recovers_glyph_near_adaptive_outer_edge():
    image = np.full((420, 1200, 3), 245, dtype=np.uint8)
    source = {"x1": 150, "y1": 60, "x2": 950, "y2": 360}
    pad_x = int(CompleteFlatEnvelopeMaskRecallPipeline._FLAT_SEARCH_PAD_X)

    glyph_x1 = source["x2"] + pad_x - 26
    glyph_x2 = glyph_x1 + 10
    image[150:182, glyph_x1:glyph_x2] = 20

    frame_x1 = source["x2"] + pad_x - 8
    frame_x2 = frame_x1 + 2
    image[90:330, frame_x1:frame_x2] = 20

    found = CompleteFlatEnvelopeMaskRecallPipeline._flat_residual_ink_envelope(
        image,
        source,
    )

    assert found is not None
    (x1, y1, x2, y2), mask = found
    assert x2 == source["x2"] + pad_x
    assert x2 >= glyph_x2

    gx1, gy1 = glyph_x1 - x1, 150 - y1
    gx2, gy2 = glyph_x2 - x1, 182 - y1
    assert np.any(mask[gy1:gy2, gx1:gx2] > 127)

    fx1, fy1 = frame_x1 - x1, 90 - y1
    fx2, fy2 = frame_x2 - x1, 330 - y1
    assert not np.any(mask[fy1:fy2, fx1:fx2] > 127)


def test_residue_envelope_pad_is_derived_from_detector_geometry():
    side = min(
        int(DETECTOR_INPUT_SIZE),
        int(DETECTOR_RESIDUE_VERIFY_MAX_SOURCE_SIDE),
    )
    expected_x = min(
        max(1, int(DETECTOR_RESIDUE_VERIFY_MAX_SOURCE_SIDE) // 6),
        max(
            int(DETECTOR_TEXT_MASK_DECODE_PAD) * 4,
            int(round(side * 0.125)),
        ),
    )
    expected_y = min(
        max(1, int(DETECTOR_RESIDUE_VERIFY_MAX_SOURCE_SIDE) // 16),
        max(
            int(DETECTOR_TEXT_MASK_DECODE_PAD) * 2,
            int(round(side * 0.03125)),
        ),
    )

    assert CompleteFlatEnvelopeMaskRecallPipeline._FLAT_SEARCH_PAD_X == expected_x
    assert CompleteFlatEnvelopeMaskRecallPipeline._FLAT_SEARCH_PAD_Y == expected_y
