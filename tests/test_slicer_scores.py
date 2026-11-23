import numpy as np

from app.downloader.slicer import _row_std


def test_row_spread_matches_numpy_on_a_tall_page():
    rng = np.random.default_rng(3)
    page = rng.integers(0, 256, (2500, 777), dtype=np.uint8)
    page[100:200] = 255
    assert np.array_equal(_row_std(page), page.std(axis=1).astype(np.float32))


def test_an_undecodable_image_yields_no_slices(tmp_path):
    from app.downloader.slicer import slice_image

    bad = tmp_path / "bad.png"
    bad.write_bytes(b"this is not an image")
    assert slice_image(bad, tmp_path, "p") == []
    assert slice_image(bad, tmp_path, "p", return_metadata=True) == []


def test_flat_colored_gutter_is_safe_and_cut_centers_inside_gap():
    from app.downloader.slicer import _find_safe_cut, _get_content_row_mask, _get_row_scores

    # Dark-slate flat gutter (gray=80) between two textured panels must be detected as safe,
    # and the cut must land near the center of the gutter rather than hugging the 12px edge.
    gray = np.full((3200, 600), 80, dtype=np.uint8)
    rng = np.random.default_rng(7)
    gray[100:1900, 40:560] = rng.integers(0, 255, (1800, 520), dtype=np.uint8)
    gray[2300:3100, 40:560] = rng.integers(0, 255, (800, 520), dtype=np.uint8)

    content_mask = _get_content_row_mask(gray, 3200, 600)
    assert not content_mask[1960:2240].any(), "flat colored gutter rows should be safe"

    scores = _get_row_scores(gray)
    cut = _find_safe_cut(content_mask, scores, lo=1600, hi=2600, target=1920)
    assert cut is not None
    assert 2020 <= cut <= 2180, f"expected centered safe cut in gutter, got {cut}"


def test_fallback_row_scores_penalize_text_stroke_rows():
    import cv2
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
