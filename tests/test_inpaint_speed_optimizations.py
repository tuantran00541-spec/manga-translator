from pathlib import Path

import cv2
import numpy as np

from app.inpaint.lama_inpainter import Inpainter, MANUAL_MAX_DILATION, MANUAL_MIN_DILATION
from app.pipeline import ChapterPipeline, write_image


def _legacy_manual_regions(mask: np.ndarray):
    binary_mask = (mask > 127).astype(np.uint8) * 255
    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    h, w = mask.shape[:2]
    regions = []
    for label in range(1, num_labels):
        component_mask = (labels == label).astype(np.uint8) * 255
        ys, xs = np.where(component_mask > 127)
        if len(ys) == 0:
            continue
        bbox_w = int(xs.max() - xs.min() + 1)
        bbox_h = int(ys.max() - ys.min() + 1)
        scale = max(1, min(bbox_w, bbox_h))
        kernel_size = int(np.clip(round(scale * 0.025) * 2 + 1, MANUAL_MIN_DILATION, MANUAL_MAX_DILATION))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        dilated = cv2.dilate(component_mask, kernel, iterations=1)
        dys, dxs = np.where(dilated > 127)
        crop_box = Inpainter._compute_manual_crop_region(
            int(dxs.min()), int(dys.min()), int(dxs.max()), int(dys.max()), w, h
        )
        x1, y1, x2, y2 = crop_box
        regions.append((crop_box, dilated[y1:y2, x1:x2]))
    return regions


def test_manual_component_roi_matches_legacy_geometry():
    rng = np.random.default_rng(123)
    for _ in range(12):
        h = int(rng.integers(80, 500))
        w = int(rng.integers(80, 500))
        mask = np.zeros((h, w), dtype=np.uint8)
        for _ in range(int(rng.integers(1, 12))):
            x = int(rng.integers(0, w))
            y = int(rng.integers(0, h))
            radius = int(rng.integers(1, 22))
            cv2.circle(mask, (x, y), radius, 255, -1)

        expected = _legacy_manual_regions(mask)
        captured = []
        inpainter = Inpainter.__new__(Inpainter)

        def capture(image, local_mask, crop_box, feather=False):
            captured.append((crop_box, local_mask.copy()))
            return image

        inpainter._smart_paint_region = capture
        inpainter.inpaint_mask(np.zeros((h, w, 3), dtype=np.uint8), mask)

        assert len(captured) == len(expected)
        for (expected_box, expected_mask), (actual_box, actual_mask) in zip(expected, captured):
            assert actual_box == expected_box
            assert np.array_equal(actual_mask, expected_mask)


class _DummyInpainter:
    def __init__(self):
        self.auto_calls = 0
        self.manual_calls = 0

    def inpaint(self, image, boxes):
        self.auto_calls += 1
        return np.clip(image.astype(np.int16) + 10, 0, 255).astype(np.uint8)

    def inpaint_mask(self, image, mask):
        self.manual_calls += 1
        output = image.copy()
        output[mask > 10] = 200
        return output


def test_repaint_reuses_auto_clean_cache(tmp_path: Path):
    image = np.full((80, 100, 3), 50, dtype=np.uint8)
    image_path = tmp_path / "page.png"
    write_image(image_path, image)

    mask = np.zeros((80, 100), dtype=np.uint8)
    mask[20:30, 30:45] = 255
    mask_path = tmp_path / "manual.png"
    write_image(mask_path, mask)

    pipeline = ChapterPipeline()
    dummy = _DummyInpainter()
    pipeline._inpainter = dummy

    first_path = pipeline._do_reinpaint(
        tmp_path, image_path, image, [], mask_path.as_posix(), reuse_auto_clean=False
    )
    first = cv2.imread(first_path)
    second_path = pipeline._do_reinpaint(
        tmp_path, image_path, image, [], mask_path.as_posix(), reuse_auto_clean=True
    )
    second = cv2.imread(second_path)

    assert dummy.auto_calls == 1
    assert dummy.manual_calls == 2
    assert np.array_equal(first, second)
    assert (tmp_path / "auto_clean_page.png").exists()
