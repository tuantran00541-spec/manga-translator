import numpy as np

from app.detector.boxes import BubbleBox
from app.inpaint.clustering import cluster_boxes, split_oversized_cluster_area


def _mask(h=10, w=10):
    return np.ones((h, w), dtype=np.uint8) * 255


def test_cluster_boxes_merges_touching_boxes():

    boxes = [
        BubbleBox(0, 0, 10, 10, 0.9, _mask()),
        BubbleBox(12, 0, 22, 10, 0.8, _mask()),
    ]

    clusters = cluster_boxes(boxes)

    assert len(clusters) == 1
    assert len(clusters[0]) == 2


def test_split_oversized_cluster_area_splits_by_bounding_box_area():

    boxes = [
        BubbleBox(0, 0, 10, 10, 0.9, _mask()),
        BubbleBox(100, 100, 110, 110, 0.8, _mask()),
    ]

    result = split_oversized_cluster_area(boxes, img_w=120, img_h=120)

    assert len(result) == 2
    assert {min(b.x1 for b in group) for group in result} == {0, 100}
