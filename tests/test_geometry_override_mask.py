from pathlib import Path

import cv2
import numpy as np

from app.detector.bubble_detector import BubbleBox
from app.inpaint.mask_geometry import remap_local_mask_page_space
from app.manifest_utils import assign_stable_detector_box_ids
from app.pipeline import ChapterPipeline, _decode_mask, _encode_mask, write_image
from app.pipeline_artwork_safe import ArtworkSafeChapterPipeline


def _glyph_mask(h: int = 20, w: int = 20) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[5:15, 7:12] = 255
    return mask


def test_page_space_remap_expands_box_without_scaling_glyphs():
    source = {"x1": 20, "y1": 30, "x2": 40, "y2": 50}
    target = {"x1": 10, "y1": 20, "x2": 50, "y2": 60}
    mask = _glyph_mask()

    remapped = remap_local_mask_page_space(mask, source, target)

    assert remapped is not None
    assert remapped.shape == (40, 40)
    assert np.array_equal(remapped[15:25, 17:22], mask[5:15, 7:12])
    assert int(np.count_nonzero(remapped)) == int(np.count_nonzero(mask))


def test_page_space_remap_crops_box_without_stretching_mask():
    source = {"x1": 20, "y1": 30, "x2": 40, "y2": 50}
    target = {"x1": 28, "y1": 34, "x2": 38, "y2": 48}
    mask = _glyph_mask()

    remapped = remap_local_mask_page_space(mask, source, target)

    assert remapped is not None
    assert remapped.shape == (14, 10)
    expected = mask[4:18, 8:18]
    assert np.array_equal(remapped, expected)


def test_stable_id_reconciliation_keeps_fresh_detector_mask_on_override():
    fresh_mask = _glyph_mask()
    fresh = [{
        "x1": 20,
        "y1": 30,
        "x2": 40,
        "y2": 50,
        "confidence": 0.91,
        "_mask_array": fresh_mask.copy(),
    }]
    existing = [{
        "id": "box_keep",
        "origin": "detector",
        "x1": 12,
        "y1": 24,
        "x2": 48,
        "y2": 58,
        "confidence": 0.91,
        "mask": None,
        "geometry_overridden": True,
        "detector_anchor": {"x1": 20, "y1": 30, "x2": 40, "y2": 50},
    }]

    assign_stable_detector_box_ids(fresh, existing)

    box = fresh[0]
    assert box["id"] == "box_keep"
    assert box["geometry_overridden"] is True
    assert box["detector_anchor"] == {"x1": 20, "y1": 30, "x2": 40, "y2": 50}
    assert (box["x1"], box["y1"], box["x2"], box["y2"]) == (12, 24, 48, 58)
    assert box["_mask_array"] is not None
    assert box["_mask_array"].shape == (34, 36)
    assert int(np.count_nonzero(box["_mask_array"])) == int(np.count_nonzero(fresh_mask))
    # The pipeline's later legacy override branch sees this job-local copy and
    # therefore cannot discard the remapped mask again.
    assert existing[0]["geometry_overridden"] is False


def test_immediate_geometry_edit_preserves_detector_mask_and_anchor():
    old_geometry = {"x1": 20, "y1": 30, "x2": 40, "y2": 50}
    new_geometry = {"x1": 12, "y1": 24, "x2": 48, "y2": 58}
    box = {
        "id": "box_edit",
        "origin": "detector",
        **old_geometry,
        "confidence": 0.88,
        "mask": _encode_mask(_glyph_mask()),
    }

    ArtworkSafeChapterPipeline._apply_box_geometry(box, new_geometry)

    assert box["geometry_overridden"] is True
    assert box["detector_anchor"] == old_geometry
    assert (box["x1"], box["y1"], box["x2"], box["y2"]) == (12, 24, 48, 58)
    remapped = _decode_mask(box["mask"])
    assert remapped is not None
    assert remapped.shape == (34, 36)
    assert int(np.count_nonzero(remapped)) == int(np.count_nonzero(_glyph_mask()))


class _Detector:
    def detect(self, image, parallel=False):
        return [BubbleBox(20, 30, 40, 50, 0.91, _glyph_mask())]


class _CapturingInpainter:
    def __init__(self):
        self.boxes = None

    def inpaint(self, image, boxes):
        self.boxes = boxes
        return image.copy()

    def inpaint_mask(self, image, mask):
        return image.copy()


def test_process_page_passes_remapped_mask_to_inpainter(tmp_path: Path):
    image = np.full((90, 100, 3), 180, dtype=np.uint8)
    image_path = tmp_path / "page.png"
    write_image(image_path, image)

    pipeline = ChapterPipeline()
    pipeline._detector = _Detector()
    capture = _CapturingInpainter()
    pipeline._inpainter = capture

    existing = [{
        "id": "box_keep",
        "origin": "detector",
        "x1": 12,
        "y1": 24,
        "x2": 48,
        "y2": 58,
        "confidence": 0.91,
        "mask": None,
        "geometry_overridden": True,
        "detector_anchor": {"x1": 20, "y1": 30, "x2": 40, "y2": 50},
    }]

    result = pipeline._process_page(
        image_path,
        tmp_path,
        existing_boxes=existing,
        parallel_detectors=False,
    )

    assert capture.boxes is not None and len(capture.boxes) == 1
    passed = capture.boxes[0]
    assert (passed.x1, passed.y1, passed.x2, passed.y2) == (12, 24, 48, 58)
    assert passed.mask is not None
    assert passed.mask.shape == (34, 36)
    assert int(np.count_nonzero(passed.mask)) == int(np.count_nonzero(_glyph_mask()))

    persisted = result["boxes"][0]
    assert persisted["geometry_overridden"] is True
    assert persisted["detector_anchor"] == {"x1": 20, "y1": 30, "x2": 40, "y2": 50}
    assert _decode_mask(persisted["mask"]) is not None

    Path(result["tmp_clean"]).unlink(missing_ok=True)
