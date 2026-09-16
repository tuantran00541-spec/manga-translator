import numpy as np

from app.detector.bubble_detector import BubbleBox
from app.detector.mask_builder import build_mask
from app.image_io import encode_mask, read_image, write_image
from app.mask_recall_pipeline import MaskRecallOptimizedChapterPipeline


def _record(mask, *, overlap=False):
    return {
        "x1": 40,
        "y1": 35,
        "x2": 120,
        "y2": 85,
        "confidence": 0.95,
        "mask": encode_mask(mask),
        "source_model": "text_segmenter.onnx",
        "class_id": 0,
        "class_name": "text_comic",
        "semantic_type": "free_text",
        "mask_source": "text_segmenter",
        "safe_to_inpaint": True,
        "ocr_eligible": True,
        "needs_review": False,
        "source_role": "text_segmenter",
        "overlap_context_only": overlap,
    }


def test_residue_sources_skip_overlap_context_and_complete_segmenter_envelope():
    sparse = np.zeros((50, 80), dtype=np.uint8)
    sparse[20:25, 10:22] = 255
    records = [_record(sparse), _record(sparse, overlap=True)]

    boxes = MaskRecallOptimizedChapterPipeline._residue_repair_effective_boxes(
        records
    )

    assert len(boxes) == 1
    assert boxes[0].source_role == "text_segmenter"
    assert boxes[0].mask.shape == (50, 80)
    assert np.all(boxes[0].mask == 255)


def test_verified_residue_can_extend_beyond_original_mask_but_not_preserve(tmp_path):
    original = np.full((120, 160, 3), 230, dtype=np.uint8)
    clean = original.copy()
    sparse = np.zeros((50, 80), dtype=np.uint8)
    sparse[18:24, 8:20] = 255

    # Missed glyph evidence is well inside the detector box but deliberately far
    # from the original sparse segmentation mask.
    residue_mask = np.full((16, 18), 255, dtype=np.uint8)
    rx1, ry1, rx2, ry2 = 92, 55, 110, 71
    clean[ry1:ry2, rx1:rx2] = 80

    raw_path = tmp_path / "raw.png"
    clean_path = tmp_path / "clean.png"
    write_image(raw_path, original)
    write_image(clean_path, clean)

    record = _record(sparse)
    result = {
        "tmp_clean": clean_path.as_posix(),
        "boxes": [record],
        "residue_regions": [
            {
                "x1": rx1,
                "y1": ry1,
                "x2": rx2,
                "y2": ry2,
                "confidence": 0.9,
                "mask": encode_mask(residue_mask),
                "source_model": "text_segmenter.onnx",
                "source_role": "text_segmenter",
                "class_name": "text_comic",
                "semantic_type": "free_text",
                "deferred_reason": "post_inpaint_text_residue",
            }
        ],
        "detection_issues": ["post_inpaint_text_residue"],
        "processing_metrics": {
            "timing_ms": {"total": 10.0},
            "detector": {"post_inpaint_residue": 1},
        },
    }

    class FakeInpainter:
        def __init__(self):
            self.mask = None

        def inpaint_mask(self, image, mask, force_lama=False):
            assert force_lama is True
            self.mask = mask.copy()
            candidate = np.full_like(image, 17)
            return candidate

        def last_metrics(self):
            return {"lama_model_runs": 1, "lama_model_ms": 3}

    class FakeDetector:
        def verify_post_inpaint_residue(self, image, boxes):
            return []

        def last_residue_metrics(self):
            return {"residue_hits": 0, "model_calls": 1}

    source_box = BubbleBox(
        40,
        35,
        120,
        85,
        0.95,
        sparse,
        source_model="text_segmenter.onnx",
        semantic_type="free_text",
        mask_source="text_segmenter",
        safe_to_inpaint=True,
        ocr_eligible=True,
        needs_review=False,
        source_role="text_segmenter",
    )
    old_authority = build_mask(original.shape[:2], [source_box], original) > 127
    assert not np.any(old_authority[ry1:ry2, rx1:rx2])

    # Protect the right half of the verifier hit. Fresh verifier evidence may add
    # authority, but preserve geometry remains absolute.
    preserve = [{"x1": 101, "y1": 50, "x2": 114, "y2": 76}]

    pipeline = MaskRecallOptimizedChapterPipeline()
    fake_inpainter = FakeInpainter()
    pipeline._inpainter = fake_inpainter
    pipeline._detector = FakeDetector()
    updated = pipeline._repair_post_inpaint_result(raw_path, result, preserve)
    repaired = read_image(clean_path)

    repair_authority = fake_inpainter.mask > 127
    assert np.any(repair_authority[ry1:ry2, rx1:101])
    assert not np.any(repair_authority[ry1:ry2, 101:rx2])
    assert np.array_equal(repaired[~repair_authority], clean[~repair_authority])
    assert np.all(repaired[repair_authority] == 17)
    assert updated["residue_regions"] == []
    assert "post_inpaint_text_residue" not in updated["detection_issues"]
    assert (
        updated["processing_metrics"]["residue_repair"]
        ["outside_authority_changed_channel_values"]
        == 0
    )
