from app.visual_qc.batch_protocol import parse_region_batch_decisions
from app.visual_qc.gemini import _parse_issues
from app.visual_qc.regions import QCRegion


def test_a_page_issue_box_names_its_corners():
    # A model that reads [ymin, xmin, ...] as x first boxed a strip of art in A.I mode; QC asks for named corners too.
    issue = {"issue_type": "residual_text", "confidence": 0.9, "label": "left",
             "x1": 250, "y1": 800, "x2": 900, "y2": 950, "mask": [[0, 0], [1000, 0], [1000, 1000]]}
    unnamed = dict(issue, box_2d=[250, 800, 900, 950])  # x first, as such a model sends it
    for key in ("x1", "y1", "x2", "y2"):
        unnamed.pop(key)
    found = _parse_issues({"issues": [issue, unnamed]}, 1600, 4000)
    assert [(i.box_2d) for i in found] == [(3200, 400, 3800, 1440)], "(y1, x1, y2, x2) in pixels; unnamed lists ignored"


def test_a_region_issue_box_is_read_inside_its_crop():
    region = QCRegion(2, "r1", (100, 200, 500, 600), ("b",), ("text",), 0.1, False)
    decisions = parse_region_batch_decisions({"regions": [{"region_id": "r1", "status": "flagged", "issues": [
        {"issue_type": "smear", "confidence": 0.8, "x1": 0, "y1": 500, "x2": 500, "y2": 1000,
         "reason": "left half, lower half", "recommended_action": "repaint"}]}]}, {"r1": region})
    (issue,) = decisions[0].issues
    assert issue.bbox == (100, 400, 300, 600)
