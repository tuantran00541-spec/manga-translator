import numpy as np

from app.detector.boxes import BubbleBox
from app.inpaint.mask_geometry import remap_local_mask_page_space
from app.page_processing import _contained, _fold_nested
from app.pipeline import ChapterPipeline


def test_remap_local_mask_page_space_copies_intersection():
    source_box = {"x1": 0, "y1": 0, "x2": 4, "y2": 4}
    target_box = {"x1": 2, "y1": 2, "x2": 6, "y2": 6}
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[2, 2] = 255

    result = remap_local_mask_page_space(mask, source_box, target_box)

    assert result is not None
    assert result.shape == (4, 4)
    assert result[0, 0] == 255


def test_map_seam_detection_to_page_clips_mask_and_coordinates():
    mask = np.arange(1, 13, dtype=np.uint8).reshape(4, 3)
    box = BubbleBox(1, 0, 4, 4, 1.0, mask.copy())

    mapped = ChapterPipeline._map_seam_detection_to_page(
        box,
        seam_source_y1=10,
        page_source_y1=12,
        page_height=3,
    )

    assert mapped is not None
    assert (mapped.y1, mapped.y2) == (0, 2)
    assert mapped.mask is not None
    assert mapped.mask.shape == (2, 3)


def test_contained_uses_half_area_threshold():
    mask = np.ones((10, 10), dtype=np.uint8) * 255
    large = BubbleBox(0, 0, 100, 100, 0.9, mask)
    small = BubbleBox(10, 10, 20, 20, 0.8, mask[:10, :10].copy())

    assert _contained(small, large) is True


def test_fold_nested_merges_a_small_box_into_its_containing_box():
    mask = np.ones((10, 10), dtype=np.uint8) * 255
    large = BubbleBox(0, 0, 100, 100, 0.9, mask)
    small = BubbleBox(10, 10, 20, 20, 0.8, mask.copy())

    folded = _fold_nested([large, small])

    assert len(folded) == 1
