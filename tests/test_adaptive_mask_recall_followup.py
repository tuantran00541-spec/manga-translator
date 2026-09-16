import cv2
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

    # The production span gate (600px by default) separates these despite the
    # first authority being almost fully contained by the second.
    assert len(Inpainter._cluster_boxes([first, second])) == 2

    clusters = OverlapAwareAdaptiveFastInpainter._cluster_boxes([first, second])
    assert len(clusters) == 1
    assert len(clusters[0]) == 2


def test_residue_outlined_dense_overlap_becomes_compact_glyph_authority():
    height, width = 220, 420
    yy, xx = np.mgrid[:height, :width]
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[..., 0] = (105 + (xx * 3 + yy * 2) % 45).astype(np.uint8)
    image[..., 1] = (95 + (xx * 2 + yy * 5) % 50).astype(np.uint8)
    image[..., 2] = (110 + (xx * 5 + yy * 3) % 40).astype(np.uint8)

    # Two rows of black glyph cores with white outlines. Components share a
    # common scale and line structure while the surrounding artwork is textured.
    glyphs = []
    for top in (55, 125):
        for left in (75, 135, 205, 275):
            outer = (left, top, left + 30, top + 48)
            glyphs.append(outer)
            cv2.rectangle(
                image,
                (outer[0], outer[1]),
                (outer[2], outer[3]),
                (245, 245, 245),
                thickness=-1,
            )
            cv2.rectangle(
                image,
                (left + 6, top + 6),
                (left + 24, top + 42),
                (15, 15, 15),
                thickness=-1,
            )

    first = _box(40, 35, 360, 190)
    second = _box(30, 25, 370, 200)
    prepared, metrics = (
        OverlapAwareAdaptiveFastInpainter._prepare_outlined_dense_text_boxes(
            image,
            [first, second],
        )
    )

    assert len(prepared) == 1
    compact = prepared[0]
    assert (compact.x1, compact.y1, compact.x2, compact.y2) == (30, 25, 370, 200)
    assert compact.mask.shape == (175, 340)
    assert metrics["outlined_text_refined_groups"] == 1
    assert metrics["outlined_text_collapsed_boxes"] == 1
    assert metrics["outlined_text_saved_pixels"] > 0
    assert metrics["outlined_text_max_margin"] >= 2

    original_pixels = (compact.x2 - compact.x1) * (compact.y2 - compact.y1)
    compact_pixels = int(np.count_nonzero(compact.mask > 127))
    assert compact_pixels < original_pixels * 0.60

    # Every glyph remains destructive authority, including its black center.
    for left, top, right, bottom in glyphs:
        center_x = ((left + right) // 2) - compact.x1
        center_y = ((top + bottom) // 2) - compact.y1
        assert compact.mask[center_y, center_x] > 127

    # Text-free corners of the dense detector region are no longer repainted.
    assert not np.any(compact.mask[:15, :15] > 127)
    assert not np.any(compact.mask[-15:, -15:] > 127)


def test_residue_outlined_dense_gate_falls_back_without_paired_text():
    height, width = 220, 420
    yy, xx = np.mgrid[:height, :width]
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[..., 0] = (90 + (xx * 7 + yy * 3) % 80).astype(np.uint8)
    image[..., 1] = (85 + (xx * 5 + yy * 11) % 85).astype(np.uint8)
    image[..., 2] = (95 + (xx * 3 + yy * 13) % 75).astype(np.uint8)

    first = _box(40, 35, 360, 190)
    second = _box(30, 25, 370, 200)
    prepared, metrics = (
        OverlapAwareAdaptiveFastInpainter._prepare_outlined_dense_text_boxes(
            image,
            [first, second],
        )
    )

    assert len(prepared) == 2
    assert metrics["outlined_text_refined_groups"] == 0
    assert np.all(prepared[0].mask == 255)
    assert np.all(prepared[1].mask == 255)


def test_residue_textured_neural_hits_receive_bounded_followup_passes(
    monkeypatch,
    tmp_path,
):
    # Deliberately textured so flat-ink recovery cannot be the reason for a retry.
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

    # Two detector-owned sources yield a three-pass safety budget. The first
    # parent pass discovers hit A, follow-up discovers hit B, final pass clears it.
    assert calls["count"] == 3
    assert updated["residue_regions"] == []
    metrics = updated["processing_metrics"]["mask_recall_repair"]
    assert metrics["repair_pass_budget"] == 3
    assert metrics["repair_passes"] == 3
    assert metrics["neural_followup_passes"] == 2
    assert metrics["neural_followup_outside_changed_channel_values"] == 0
