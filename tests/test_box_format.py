from app.box_format import scaled_box


def test_a_box_with_an_unreadable_corner_is_dropped():
    for bad in (float("nan"), float("inf"), "x", None):
        assert scaled_box({"x1": 0, "y1": 0, "x2": bad, "y2": 500}, 800, 600) is None
    assert scaled_box({"x1": -5, "y1": 0, "x2": 1200, "y2": 500}, 800, 600) == (0.0, 0.0, 800.0, 300.0)
