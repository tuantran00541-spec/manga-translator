"""Export options: format, width, page height and names as a scanlation group asks for them."""
import io

import numpy as np
import pytest
from PIL import Image

from app.export_pages import ExportOptions, finish_page, split_rows


def test_options_are_checked_and_the_default_keeps_the_old_archive():
    assert ExportOptions.parse().default
    assert ExportOptions.parse("JPEG", 120, 800, 3000, "ch 01/../", "CBZ") == ExportOptions("jpg", 100, 800, 3000, "ch01..", "cbz")
    for bad in ({"fmt": "gif"}, {"width": 100}, {"split": 50}, {"archive": "rar"}):
        with pytest.raises(ValueError):
            ExportOptions.parse(**bad)


def test_a_tall_page_is_cut_at_a_quiet_gutter_below_the_limit():
    page = np.full((5000, 400, 3), 255, np.uint8)
    rng = np.random.default_rng(0)
    page[:2600] = rng.integers(0, 255, (2600, 400, 3))  # busy art down to row 2600
    page[2700:] = rng.integers(0, 255, (2300, 400, 3))  # and again from 2700: a white gutter between
    cuts = split_rows(page, 3000)
    assert len(cuts) == 1 and 2600 <= cuts[0] <= 2700


def test_a_page_is_resized_cut_and_encoded():
    page = Image.new("RGB", (1600, 9000), "white")
    files = finish_page(page, ExportOptions.parse("jpg", 85, 800, 2000))
    sizes = [Image.open(io.BytesIO(data)).size for data in files]
    assert all(width == 800 for width, _ in sizes) and sum(height for _, height in sizes) == 4500
    assert max(height for _, height in sizes) <= 2000


def test_webp_cuts_a_page_taller_than_it_can_hold():
    files = finish_page(Image.new("RGB", (100, 20000), "white"), ExportOptions.parse("webp"))
    assert len(files) == 2 and all(Image.open(io.BytesIO(d)).height <= 16000 for d in files)
