import numpy as np

from app.detector.bubble_detector import BubbleBox
from app.image_io import write_image
from app.inpaint.lama_inpainter import Inpainter
from app.inpaint.overlap_aware_adaptive_inpainter import (
    OverlapAwareAdaptiveFastInpainter,
)
from app.mask_recall_envelope_v2_pipeline import (
    CompleteFlatEnvelopeMaskRecallPipeline,
)
from app.optimized_pipeline import OptimizedChapterPipeline


def _box(x1, y1, x2, y2):
    mask = np.full((y2 - y1, x2 - x1), 255, dtype=np.uint8)
    return BubbleBox(
        x1,
        y1,
        x2,
        y2,
        0.95,
        mask,
        source_model="text_segmenter.onnx",
        semantic_type="free_text",
        mask_source="text_segmenter",
        safe_to_inpaint=True,
        ocr_eligible=True,
        needs_review=False,
        source_role="text_segmenter",
    )


def test_residue_overlap_authorities_merge_before_size_gate():
    first = _box(0, 100, 900, 500)
    second = _box(0, 0, 900, 520)

    assert len(Inpainter._cluster_boxes([first, second])) == 2

    clusters = OverlapAwareAdaptiveFastInpainter._cluster_boxes([first, second])
    assert len(clusters) == 1
    assert len(clusters[0]) == 2


def test_residue_textured_neural_hits_receive_bounded_followup_passes(
    monkeypatch,
    tmp_path,
):
    yy, xx = np.mgrid[:120, :160]
    clean = np.empty((120, 160, 3), dtype=np.uint8)
    clean[..., 0] = (40 + (xx * 7 + yy * 3) % 180).astype(np.uint8)
    clean[..., 1] = (50 + (xx * 5 + yy * 11) % 170).astype(np.uint8)
    clean[..., 2] = (60 + (xx * 13 + yy * 2) % 160).astype(np.uint8)

    raw_path = tmp_path / "raw.png"
    clean_path = tmp_path / "clean.png"
    write_image(raw_path, clean)
    write_image(clean_path, clean)

    record = {
        "x1": 10,
        "y1": 10,
        "x2": 150,
        "y2": 110,
        "confidence": 0.95,
        "source_model": "text_segmenter.onnx",
        "source_role": "text_segmenter",
        "class_name": "text_comic",
        "semantic_type": "free_text",
        "safe_to_inpaint": True,
        "ocr_eligible": True,
        "needs_review": False,
    }
    result = {
        "tmp_clean": clean_path.as_posix(),
        "boxes": [dict(record), dict(record)],
        "residue_regions": [],
        "detection_issues": [],
        "processing_metrics": {"timing_ms": {}, "detector": {}},
    }

    calls = {"count": 0}
    residues = [
        {"x1": 20, "y1": 20, "x2": 35, "y2": 45},
        {"x1": 80, "y1": 30, "x2": 95, "y2": 55},
        None,
    ]

    def fake_parent(self, img_path, current, preserve_regions):
        index = calls["count"]
        calls["count"] += 1
        residue = residues[min(index, len(residues) - 1)]
        if residue is None:
            current["residue_regions"] = []
            current["detection_issues"] = []
        else:
            current["residue_regions"] = [
                {
                    **residue,
                    "confidence": 0.8,
                    "source_model": "text_segmenter.onnx",
                    "source_role": "text_segmenter",
                    "class_name": "text_comic",
                    "semantic_type": "free_text",
                    "deferred_reason": "post_inpaint_text_residue",
                }
            ]
            current["detection_issues"] = ["post_inpaint_text_residue"]
        current.setdefault("processing_metrics", {})["residue_repair"] = {
            "attempted": 1,
            "repair_mask_pixels": 64,
            "outside_authority_changed_channel_values": 0,
        }
        return current

    monkeypatch.setattr(
        OptimizedChapterPipeline,
        "_repair_post_inpaint_result",
        fake_parent,
    )

    pipeline = CompleteFlatEnvelopeMaskRecallPipeline()
    updated = pipeline._repair_post_inpaint_result(raw_path, result, None)

    assert calls["count"] == 3
    assert updated["residue_regions"] == []
    metrics = updated["processing_metrics"]["mask_recall_repair"]
    assert metrics["repair_pass_budget"] == 3
    assert metrics["repair_passes"] == 3
    assert metrics["neural_followup_passes"] == 2
    assert metrics["neural_followup_outside_changed_channel_values"] == 0
