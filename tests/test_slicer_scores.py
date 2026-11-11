import numpy as np

from app.downloader.slicer import _row_std


def test_row_spread_matches_numpy_on_a_tall_page():
    rng = np.random.default_rng(3)
    page = rng.integers(0, 256, (2500, 777), dtype=np.uint8)
    page[100:200] = 255
    assert np.array_equal(_row_std(page), page.std(axis=1).astype(np.float32))


def test_an_undecodable_image_yields_no_slices(tmp_path):
    """N13: a broken download contributes no slices, like the zip path's
    broken-image skip — returning the path itself would hand the
    pipeline an undecodable page to translate and render."""
    from app.downloader.slicer import slice_image

    bad = tmp_path / "bad.png"
    bad.write_bytes(b"this is not an image")
    assert slice_image(bad, tmp_path, "p") == []
    assert slice_image(bad, tmp_path, "p", return_metadata=True) == []
