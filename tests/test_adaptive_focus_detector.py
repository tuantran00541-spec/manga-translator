from app.detector.adaptive_focus_detector import (
    plan_adaptive_windows,
    plan_focus_chips,
)
from app.detector.bubble_detector import BubbleBox


def test_adaptive_window_plan_matches_validated_v4_shapes():
    assert plan_adaptive_windows(1537) == [(0, 1024), (513, 1537)]
    assert plan_adaptive_windows(2400) == [(0, 1300), (1100, 2400)]
    assert plan_adaptive_windows(4096) == [
        (0, 1174),
        (974, 2148),
        (1948, 3122),
        (2922, 4096),
    ]


def test_focus_chips_are_bounded_and_surface_budget_deferred_regions():
    proposals = [
        BubbleBox(0, 0, 900, 4096, 0.9, semantic_type="free_text"),
        BubbleBox(30, 300, 180, 440, 0.9, semantic_type="speech_bubble"),
    ]

    chips, deferred = plan_focus_chips(4096, 900, proposals, max_chips=2)

    assert 1 <= len(chips) <= 2
    assert deferred
    assert all(0 <= x1 < x2 <= 900 and 0 <= y1 < y2 <= 4096 for x1, y1, x2, y2 in chips)
    assert all(max(x2 - x1, y2 - y1) <= 1344 for x1, y1, x2, y2 in chips)
