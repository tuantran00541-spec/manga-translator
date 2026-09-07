from app.detector.adaptive_focus_detector import (
    plan_adaptive_windows,
    plan_focus_bands,
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


def test_focus_bands_are_bounded_and_cover_proposals():
    proposals = [
        BubbleBox(
            10, 100, 200, 250, 0.9,
            semantic_type="speech_bubble",
        ),
        BubbleBox(
            10, 700, 200, 850, 0.9,
            semantic_type="free_text",
        ),
        BubbleBox(
            10, 2000, 200, 2150, 0.9,
            source_model="opencv_mser",
            semantic_type="text",
        ),
    ]

    bands = plan_focus_bands(2400, proposals, max_chips=2)

    assert 1 <= len(bands) <= 2
    assert all(0 <= start < end <= 2400 for start, end in bands)
    for proposal in proposals:
        assert any(
            start <= proposal.y1 and proposal.y2 <= end
            for start, end in bands
        )
