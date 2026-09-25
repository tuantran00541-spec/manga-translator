import numpy as np

from app.image_io import encode_mask, read_image, write_image
from app.inpaint.lama_inpainter import Inpainter
from app.optimized_pipeline import OptimizedChapterPipeline


def test_optimized_pipeline_uses_plain_lama_inpainter():
    pipeline = OptimizedChapterPipeline()
    assert isinstance(pipeline.inpainter, Inpainter)

def test_residue_second_pass_is_clipped_to_existing_authority(tmp_path):
    original = np.full((120, 160, 3), 230, dtype=np.uint8)
    clean = original.copy()
    local_mask = np.zeros((50, 80), dtype=np.uint8)
    local_mask[18:24, 14:66] = 255
    local_mask[31:37, 24:58] = 255
    clean[35:85, 40:120][local_mask > 127] = 120

    raw_path = tmp_path / "raw.png"
    clean_path = tmp_path / "clean.png"
    write_image(raw_path, original)
    write_image(clean_path, clean)

    record = {
        "x1": 40,
        "y1": 35,
        "x2": 120,
        "y2": 85,
        "confidence": 0.95,
        "mask": encode_mask(local_mask),
        "source_model": "text_segmenter.onnx",
        "class_id": 0,
        "class_name": "text_comic",
        "semantic_type": "free_text",
        "mask_source": "text_segmenter",
        "safe_to_inpaint": True,
        "ocr_eligible": True,
        "needs_review": False,
        "source_role": "text_segmenter",
    }
    result = {
        "tmp_clean": clean_path.as_posix(),
        "boxes": [record],
        "residue_regions": [
            {
                "x1": 52,
                "y1": 47,
                "x2": 108,
                "y2": 75,
                "confidence": 0.9,
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
            candidate = np.zeros_like(image)
            candidate[:] = 17
            return candidate

        def last_metrics(self):
            return {"lama_model_runs": 1, "lama_model_ms": 3}

    class FakeDetector:
        def verify_post_inpaint_residue(self, image, boxes):
            return []

        def last_residue_metrics(self):
            return {"residue_hits": 0, "model_calls": 1}

    pipeline = OptimizedChapterPipeline()
    fake_inpainter = FakeInpainter()
    pipeline._inpainter = fake_inpainter
    pipeline._detector = FakeDetector()
    updated = pipeline._repair_post_inpaint_result(
        raw_path,
        result,
        None,
    )
    repaired = read_image(clean_path)

    authority = fake_inpainter.mask > 10
    assert np.any(authority)
    assert np.array_equal(repaired[~authority], clean[~authority])
    assert np.all(repaired[authority] == 17)
    assert updated["residue_regions"] == []
    assert "post_inpaint_text_residue" not in updated["detection_issues"]
    assert updated["processing_metrics"]["residue_repair"]["attempted"] == 1
    assert (
        updated["processing_metrics"]["residue_repair"]
        ["outside_authority_changed_channel_values"]
        == 0
    )
