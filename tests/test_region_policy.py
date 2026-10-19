"""One overlap measure shared by the review, the editor and the export gate."""
from app.region_policy import overlap_share


def test_overlap_is_the_shared_area_over_the_smaller_box_for_dicts_and_tuples():
    big, small = {"x1": 0, "y1": 0, "x2": 100, "y2": 100}, (50, 50, 70, 70)
    assert overlap_share(big, small) == 1.0
    assert overlap_share((0, 0, 10, 10), (5, 0, 15, 10)) == 0.5
    assert overlap_share((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0


def test_a_malformed_box_overlaps_nothing():
    assert overlap_share({"x1": 0, "y1": 0}, (0, 0, 10, 10)) == 0.0
    assert overlap_share({"x1": "a", "y1": 0, "x2": 5, "y2": 5}, (0, 0, 10, 10)) == 0.0
