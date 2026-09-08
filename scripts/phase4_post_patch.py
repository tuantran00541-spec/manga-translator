from pathlib import Path
import re

adaptive_path = Path("app/detector/adaptive_focus_detector.py")
source = adaptive_path.read_text(encoding="utf-8")
source, count = re.subn(
    r"\n\ndef plan_focus_bands\(.*?\n\ndef _axis_tiles\(",
    "\n\ndef _axis_tiles(",
    source,
    count=1,
    flags=re.S,
)
if count != 1:
    raise RuntimeError(f"expected one legacy focus-band removal, got {count}")
adaptive_path.write_text(source, encoding="utf-8")

test_path = Path("tests/test_adaptive_focus_detector.py")
test_source = test_path.read_text(encoding="utf-8")
test_source = test_source.replace(
    "    plan_adaptive_windows,\n    plan_focus_bands,\n",
    "    plan_adaptive_windows,\n    plan_focus_chips,\n",
    1,
)
test_source, count = re.subn(
    r"def test_focus_bands_are_bounded_and_cover_proposals\(\):.*\Z",
    '''def test_focus_chips_are_bounded_and_surface_budget_deferred_regions():
    proposals = [
        BubbleBox(0, 0, 900, 4096, 0.9, semantic_type="free_text"),
        BubbleBox(30, 300, 180, 440, 0.9, semantic_type="speech_bubble"),
    ]

    chips, deferred = plan_focus_chips(4096, 900, proposals, max_chips=2)

    assert 1 <= len(chips) <= 2
    assert deferred
    assert all(0 <= x1 < x2 <= 900 and 0 <= y1 < y2 <= 4096 for x1, y1, x2, y2 in chips)
    assert all(max(x2 - x1, y2 - y1) <= 1344 for x1, y1, x2, y2 in chips)
''',
    test_source,
    count=1,
    flags=re.S,
)
if count != 1:
    raise RuntimeError(f"expected one adaptive focus test replacement, got {count}")
test_path.write_text(test_source, encoding="utf-8")
print("phase4 post patch applied")
