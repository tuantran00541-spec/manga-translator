import numpy as np

from app.mask_recall_envelope_v2_pipeline import CompleteFlatEnvelopeMaskRecallPipeline


def test_residue_envelope_recovers_glyph_beyond_old_96px_clip():
    image = np.full((420, 1200, 3), 245, dtype=np.uint8)
    source = {"x1": 150, "y1": 60, "x2": 950, "y2": 360}

    # This glyph begins 102 px beyond the detector edge. The old 96 px envelope
    # clipped it; the 128 px candidate must contain the full component.
    image[150:182, 1052:1062] = 20

    # A long frame-like stroke in the same envelope must still be rejected by
    # the inherited span/ring safety gates.
    image[90:330, 1070:1072] = 20

    found = CompleteFlatEnvelopeMaskRecallPipeline._flat_residual_ink_envelope(
        image,
        source,
    )

    assert found is not None
    (x1, y1, x2, y2), mask = found
    assert x2 >= 1062
    assert x2 == 1078  # source x2 + 128, clipped only by the page if necessary.

    gx1, gy1 = 1052 - x1, 150 - y1
    gx2, gy2 = 1062 - x1, 182 - y1
    assert np.any(mask[gy1:gy2, gx1:gx2] > 127)

    fx1, fy1 = 1070 - x1, 90 - y1
    fx2, fy2 = 1072 - x1, 330 - y1
    assert not np.any(mask[fy1:fy2, fx1:fx2] > 127)


def test_residue_envelope_keeps_vertical_pad_conservative():
    assert CompleteFlatEnvelopeMaskRecallPipeline._FLAT_SEARCH_PAD_X == 128
    assert CompleteFlatEnvelopeMaskRecallPipeline._FLAT_SEARCH_PAD_Y == 32
