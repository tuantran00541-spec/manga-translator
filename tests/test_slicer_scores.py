import cv2
import numpy as np
import pytest

from app.downloader.slicer import _find_cut_rows, _row_std, slice_image
from app.parameters import SLICE_MAX_HEIGHT


def test_row_spread_matches_numpy_on_a_tall_page():
    rng = np.random.default_rng(3)
    page = rng.integers(0, 256, (2500, 777), dtype=np.uint8)
    page[100:200] = 255
    assert np.array_equal(_row_std(page), page.std(axis=1).astype(np.float32))


def test_an_undecodable_image_yields_no_slices(tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"this is not an image")
    assert slice_image(bad, tmp_path, "p") == []
    assert slice_image(bad, tmp_path, "p", return_metadata=True) == []


def test_flat_colored_gutter_is_safe_and_cut_centers_inside_gap():
    from app.downloader.slicer import _find_safe_cut, _get_content_row_mask, _get_row_scores, _safe_cut_rows

    # Dark-slate flat gutter (gray=80) between two textured panels must be detected as safe,
    # and the cut must land near the center of the gutter rather than hugging the 12px edge.
    gray = np.full((3200, 600), 80, dtype=np.uint8)
    rng = np.random.default_rng(7)
    gray[100:1900, 40:560] = rng.integers(0, 255, (1800, 520), dtype=np.uint8)
    gray[2300:3100, 40:560] = rng.integers(0, 255, (800, 520), dtype=np.uint8)

    content_mask = _get_content_row_mask(gray, 3200, 600)
    assert not content_mask[1960:2240].any(), "flat colored gutter rows should be safe"

    scores = _get_row_scores(gray)
    cut = _find_safe_cut(_safe_cut_rows(content_mask), scores, lo=1600, hi=2600, target=1920)
    assert cut is not None
    assert 2020 <= cut <= 2180, f"expected centered safe cut in gutter, got {cut}"


def test_fallback_row_scores_penalize_text_stroke_rows():
    from app.downloader.slicer import _get_row_scores

    gray = np.full((400, 600), 220, dtype=np.uint8)
    # Row band with high-contrast text strokes
    cv2.putText(gray, "CHAPTER TITLE TEXT", (40, 120), cv2.FONT_HERSHEY_DUPLEX, 1.3, 10, 3)
    # Row band with smooth low-contrast vertical shading (same vertical intensity, no horizontal strokes)
    for x in range(600):
        gray[250:320, x] = int(120 + 60 * (x / 599.0))

    scores = _get_row_scores(gray)
    text_band_score = float(scores[90:130].max())
    smooth_band_score = float(scores[260:310].mean())
    assert text_band_score > smooth_band_score * 3.0


def test_slicer_waits_for_a_gutter_beyond_the_target_maximum():
    height = SLICE_MAX_HEIGHT + 1800
    gray = np.full((height, 40), 255, dtype=np.uint8)
    occupied = np.ones(height, dtype=bool)
    occupied[SLICE_MAX_HEIGHT + 300:SLICE_MAX_HEIGHT + 400] = False

    (cut,) = _find_cut_rows(gray, *gray.shape, unsafe_rows=occupied)

    assert SLICE_MAX_HEIGHT + 300 < cut < SLICE_MAX_HEIGHT + 400


@pytest.mark.parametrize("edge_color", [(0, 0, 0), (210, 255, 255)])
def test_slicer_preserves_a_source_with_no_safe_cut(tmp_path, edge_color):
    image = np.full((SLICE_MAX_HEIGHT + 1100, 120, 3), 255, dtype=np.uint8)
    image[:, 65] = edge_color
    source = tmp_path / "panel.png"
    assert cv2.imwrite(str(source), image)

    (part,) = slice_image(source, tmp_path, "p", return_metadata=True)

    assert part["source_y1"] == 0 and part["source_y2"] == image.shape[0]
    assert not part["unsafe_before"] and not part["unsafe_after"]
    assert np.array_equal(cv2.imread(str(part["path"])), image)


def test_very_tall_connected_page_has_overlapping_unsafe_seams(tmp_path):
    height = SLICE_MAX_HEIGHT * 3
    image = np.full((height, 120, 3), 255, dtype=np.uint8)
    image[:, 65] = (0, 0, 0)
    source = tmp_path / "long.png"
    assert cv2.imwrite(str(source), image)

    parts = slice_image(source, tmp_path, "long", return_metadata=True)

    assert len(parts) > 1
    assert parts[0]["unsafe_after"] and parts[1]["unsafe_before"]
    for first, second in zip(parts, parts[1:]):
        assert first["core_source_y2"] == second["core_source_y1"]
        assert first["source_y2"] > second["source_y1"]
        assert first["core_y2"] == first["core_source_y2"] - first["source_y1"]
        assert second["core_y1"] == second["core_source_y1"] - second["source_y1"]
        assert np.array_equal(
            cv2.imread(str(first["path"]))[second["source_y1"] - first["source_y1"] :],
            cv2.imread(str(second["path"]))[: first["source_y2"] - second["source_y1"]],
        )
