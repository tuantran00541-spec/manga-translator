import numpy as np

from app.downloader.slicer import _row_std


def test_row_spread_matches_numpy_on_a_tall_page():
    rng = np.random.default_rng(3)
    page = rng.integers(0, 256, (2500, 777), dtype=np.uint8)
    page[100:200] = 255
    assert np.array_equal(_row_std(page), page.std(axis=1).astype(np.float32))
