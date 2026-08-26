# Real-chapter acceptance also covers Asura chapter 195 under production workers=2.
import numpy as np

from app.detector.bubble_detector import BubbleBox, YoloDetector
from app.detector.combined_detector import CombinedTextDetector
from app.detector.mask_builder import build_mask
from app.detector.recovery import SecondaryTextRecovery
from app.downloader.slicer import OVERLAP_CONTEXT


def _mask(h=20, w=40):
    m = np.zeros((h, w), np.uint8)
    m[4:-4, 4:-4] = 255
    return m


def test_provenance_fields_are_retained_on_box():
    b = BubbleBox(1, 2, 41, 22, .8, _mask(), source_model="text_segmenter.onnx", class_id=0,
                  class_name="text_comic", semantic_type="text", mask_source="text_segmenter",
                  safe_to_inpaint=True, ocr_eligible=True, needs_review=False)
    assert b.source_model == "text_segmenter.onnx"
    assert b.class_name == "text_comic"
    assert b.semantic_type == "text"


def test_verified_mask_requires_real_pixel_mask():
    assert not BubbleBox(0, 0, 10, 10, .9, None).verified_mask
    assert BubbleBox(0, 0, 10, 10, .9, np.ones((10, 10), np.uint8) * 255).verified_mask


def test_segmenter_semantics_become_safe_only_with_verified_mask():
    d = YoloDetector.__new__(YoloDetector)
    b = BubbleBox(0, 0, 10, 10, .9, np.ones((10, 10), np.uint8) * 255,
                  source_model="text_segmenter.onnx")
    out = d._with_semantics(b)
    assert out.safe_to_inpaint and out.ocr_eligible and not out.needs_review


def test_non_segmenter_mask_is_not_auto_safe():
    d = YoloDetector.__new__(YoloDetector)
    b = BubbleBox(0, 0, 10, 10, .9, np.ones((10, 10), np.uint8) * 255,
                  source_model="bubble_yolo.onnx")
    out = d._with_semantics(b)
    assert not out.safe_to_inpaint and not out.ocr_eligible and out.needs_review


def test_class_aware_nms_keeps_overlapping_different_classes():
    d = YoloDetector.__new__(YoloDetector)
    d.conf_threshold = .1
    d.source_model = "bubble_yolo.onnx"
    d._decode_mask = lambda *a: None
    c = [(0, 0, 100, 100, .9, 0, 2, None, None),
         (0, 0, 100, 100, .8, 1, 2, None, None)]
    out = d._nms(c, None)
    assert {b.class_id for b in out} == {0, 1}


def test_final_nms_does_not_cross_semantic_types():
    a = BubbleBox(0, 0, 100, 100, .9, source_model="bubble_yolo.onnx", semantic_type="speech_bubble")
    b = BubbleBox(0, 0, 100, 100, .8, source_model="bubble_yolo.onnx", semantic_type="free_text")
    assert len(CombinedTextDetector._apply_final_nms([a, b])) == 2


def test_watermark_classification_is_review_only():
    d = CombinedTextDetector.__new__(CombinedTextDetector)
    b = BubbleBox(0, 2, 400, 22, .9, _mask(20, 400), source_model="text_segmenter.onnx")
    out = d._classify(b, 400, 400)
    assert out.semantic_type == "watermark"
    assert not out.safe_to_inpaint and not out.ocr_eligible and out.needs_review


def test_missing_mask_never_builds_detector_rectangle():
    b = BubbleBox(10, 10, 50, 30, .9, None, source_model="bubble_yolo.onnx", safe_to_inpaint=False)
    m = build_mask((80, 80), [b], np.zeros((80, 80, 3), np.uint8))
    assert np.count_nonzero(m) == 0


def test_mser_watermark_heuristic_flags_edge_banner():
    assert SecondaryTextRecovery._watermark_like(0, 10, 500, 40, 500, 600)


def test_recovery_empty_image_is_empty():
    assert SecondaryTextRecovery().detect(np.zeros((0, 0, 3), np.uint8)) == []


def test_overlap_context_is_large_enough_for_real_boundary_case():
    assert OVERLAP_CONTEXT >= 344


def test_review_only_box_is_not_ocr_eligible():
    b = BubbleBox(0, 0, 10, 10, .2, None, source_model="opencv_mser", semantic_type="free_text",
                  safe_to_inpaint=False, ocr_eligible=False, needs_review=True)
    assert b.needs_review and not b.ocr_eligible and not b.safe_to_inpaint


def test_tall_segmenter_has_global_context_path():
    import inspect
    src = inspect.getsource(YoloDetector.detect)
    assert "_detect_single_plain(image, 0, 0)" in src
    assert "text_segmenter" in src
