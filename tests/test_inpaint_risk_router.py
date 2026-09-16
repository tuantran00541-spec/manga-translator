from __future__ import annotations

import cv2
import numpy as np

from app.detector.bubble_detector import BubbleBox
from app.inpaint.risk_aware_inpainter import (
    CoalescingRiskAwareInpainter,
    RiskAwareFastInpainter,
    measure_inpaint_risk,
)


def _box(x1: int, y1: int, x2: int, y2: int, *, semantic: str = "speech_bubble") -> BubbleBox:
    mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    cv2.rectangle(mask, (12, 10), (min(mask.shape[1] - 4, 70), 18), 255, -1)
    return BubbleBox(
        x1,
        y1,
        x2,
        y2,
        0.99,
        mask,
        source_model="text_segmenter.onnx",
        class_name="text_comic",
        semantic_type=semantic,
        mask_source="text_segmenter",
        safe_to_inpaint=True,
        ocr_eligible=True,
        needs_review=False,
        source_role="text_segmenter",
    )


def _gradient(height: int, width: int) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    return np.dstack(
        (
            35 + 120 * xx / max(1, width - 1),
            70 + 80 * yy / max(1, height - 1),
            180 - 90 * xx / max(1, width - 1),
        )
    ).astype(np.uint8)


def test_risk_measure_keeps_flat_context_low():
    crop = np.full((180, 260, 3), 248, dtype=np.uint8)
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    mask[70:82, 80:180] = 255

    risk = measure_inpaint_risk(crop, mask)

    assert risk.level == "low"
    assert risk.score == 0.0
    assert risk.context_pixels > 0


def test_risk_measure_marks_gradient_context_high():
    crop = _gradient(180, 260)
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    mask[70:82, 80:180] = 255

    risk = measure_inpaint_risk(crop, mask)

    assert risk.level == "high"
    assert risk.chroma_std > 10.0 or risk.gradient_mean > 2.0


def test_risk_router_falls_back_to_lama_for_gradient_bubble():
    image = _gradient(260, 360)
    box = _box(50, 50, 310, 210)
    calls = []
    inpainter = RiskAwareFastInpainter()

    def fake_lama(full, _crop, local_mask, crop_box, feather=False):
        calls.append((local_mask.shape, crop_box, feather))
        output = full.copy()
        x1, y1, x2, y2 = crop_box
        view = output[y1:y2, x1:x2]
        view[local_mask > 127] = (77, 77, 77)
        return output

    inpainter._lama_fill = fake_lama
    output = inpainter.inpaint(image, [box])

    metrics = inpainter.last_metrics()
    assert calls
    assert metrics["risk_fastpath_rejections"] == 1
    assert metrics["risk_forced_lama_regions"] == 1
    assert metrics["risk_medium_regions"] + metrics["risk_high_regions"] >= 2
    assert metrics["bubble_fast_fill_regions"] == 0
    assert np.array_equal(output[0, 0], image[0, 0])


def test_risk_router_keeps_flat_bubble_fast_path():
    image = np.full((220, 320, 3), 248, dtype=np.uint8)
    box = _box(50, 40, 270, 180)
    image[40:180, 50:270][box.mask > 127] = 10
    inpainter = RiskAwareFastInpainter()
    inpainter._lama_fill = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("flat bubble should not use LaMa")
    )

    output = inpainter.inpaint(image, [box])

    metrics = inpainter.last_metrics()
    assert metrics["risk_low_regions"] == 1
    assert metrics["bubble_fast_fill_regions"] == 1
    assert metrics["lama_model_runs"] == 0
    assert float(output[50:60, 62:122].mean()) > float(image[50:60, 62:122].mean())


def test_e6_coalesces_compatible_high_risk_clusters_without_outside_writes():
    image = _gradient(240, 420)
    first = _box(50, 70, 130, 120, semantic="free_text")
    second = _box(170, 70, 250, 120, semantic="free_text")
    batch_calls = []
    inpainter = CoalescingRiskAwareInpainter()
    inpainter._ensure_session = lambda: None
    inpainter.dynamic_lama = True

    def fake_batch(canvases, masks):
        batch_calls.append((canvases, masks))
        outputs = []
        for canvas, local_mask in zip(canvases, masks):
            output = canvas.copy()
            output[local_mask > 127] = (66, 66, 66)
            outputs.append(output)
        return outputs

    inpainter._run_lama_batch = fake_batch
    output = inpainter.inpaint(image, [first, second])

    metrics = inpainter.last_metrics()
    assert len(batch_calls) == 1
    assert len(batch_calls[0][0]) == 2
    assert metrics["coalesce_merges"] == 1
    assert metrics["coalesce_saved_clusters"] == 1
    changed = np.any(output != image, axis=2)
    authority = np.zeros(image.shape[:2], dtype=np.uint8)
    from scripts.benchmark_inpaint_accuracy import _box_mask

    authority = np.maximum(authority, _box_mask(image, first, inpainter))
    authority = np.maximum(authority, _box_mask(image, second, inpainter))
    assert np.count_nonzero(changed & ~authority) == 0
