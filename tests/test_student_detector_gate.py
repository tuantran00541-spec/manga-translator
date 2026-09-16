from __future__ import annotations

from scripts.benchmark_student_detector import box_iou


def test_student_detector_iou_is_half_open_and_symmetric():
    first = (0, 0, 10, 10)
    second = (5, 0, 15, 10)
    assert box_iou(first, second) == 1 / 3
    assert box_iou(second, first) == box_iou(first, second)
    assert box_iou(first, (20, 20, 30, 30)) == 0.0
