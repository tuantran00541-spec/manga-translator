from __future__ import annotations

import cv2
import numpy as np

import app.detector.recovery as recovery_module
from app.detector.recovery import SecondaryTextRecovery
from app.inpaint.lama_inpainter import Inpainter


def _reference_cluster_rects(rects):
    remaining = list(rects)
    clusters = []
    while remaining:
        cluster = [remaining.pop()]
        changed = True
        while changed:
            changed = False
            keep = []
            cx1 = min(r[0] for r in cluster)
            cy1 = min(r[1] for r in cluster)
            cx2 = max(r[2] for r in cluster)
            cy2 = max(r[3] for r in cluster)
            ch = max(8, cy2 - cy1)
            for rect in remaining:
                rx1, ry1, rx2, ry2 = rect
                near_x = not (
                    rx1
                    > cx2 + ch * recovery_module.MSER_CLUSTER_NEAR_X_FACTOR
                    or rx2
                    < cx1 - ch * recovery_module.MSER_CLUSTER_NEAR_X_FACTOR
                )
                near_y = not (
                    ry1
                    > cy2 + ch * recovery_module.MSER_CLUSTER_NEAR_Y_FACTOR
                    or ry2
                    < cy1 - ch * recovery_module.MSER_CLUSTER_NEAR_Y_FACTOR
                )
                if near_x and near_y:
                    cluster.append(rect)
                    changed = True
                else:
                    keep.append(rect)
            remaining = keep
        clusters.append(cluster)
    return clusters


def test_vectorized_mser_cluster_growth_matches_historical_semantics():
    rng = np.random.default_rng(20260919)
    rects = []
    for _ in range(180):
        x1 = int(rng.integers(0, 1800))
        y1 = int(rng.integers(0, 2800))
        width = int(rng.integers(4, 90))
        height = int(rng.integers(4, 70))
        rects.append((x1, y1, x1 + width, y1 + height))

    expected = _reference_cluster_rects(rects)
    actual = SecondaryTextRecovery._cluster_rects(rects)

    assert actual == expected


def test_manual_mask_connected_components_scan_only_mask_bbox():
    image = np.full((600, 800, 3), 220, dtype=np.uint8)
    mask = np.zeros((600, 800), dtype=np.uint8)
    mask[210:225, 310:350] = 255
    mask[250:268, 390:430] = 255

    inpainter = Inpainter()
    calls = []

    def fake_paint(image_arg, local_mask, crop_box, **kwargs):
        calls.append(
            {
                "crop_box": tuple(int(value) for value in crop_box),
                "mask_pixels": int(np.count_nonzero(local_mask > 127)),
            }
        )
        return image_arg

    inpainter._smart_paint_region = fake_paint
    output = inpainter.inpaint_mask(image, mask)
    metrics = inpainter.last_metrics()

    assert np.array_equal(output, image)
    assert metrics["mask_components"] == 2
    assert len(calls) == 2
    assert all(call["mask_pixels"] > 0 for call in calls)
    assert 0 < metrics["mask_cc_input_pixels"] < mask.size
    assert metrics["mask_cc_saved_pixels"] == (
        mask.size - metrics["mask_cc_input_pixels"]
    )
    assert metrics["mask_cc_input_pixels"] < mask.size // 10


def test_fixed_lama_skips_same_size_resize(monkeypatch):
    crop = np.full((256, 512, 3), 127, dtype=np.uint8)
    mask = np.zeros((256, 512), dtype=np.uint8)
    mask[80:160, 120:390] = 255

    inpainter = Inpainter()
    inpainter._run_lama = lambda canvas, mask_canvas: canvas.copy()

    original_resize = cv2.resize
    resize_calls = []

    def tracked_resize(source, dsize, *args, **kwargs):
        resize_calls.append((tuple(source.shape[:2]), tuple(dsize)))
        return original_resize(source, dsize, *args, **kwargs)

    monkeypatch.setattr(cv2, "resize", tracked_resize)
    painted = inpainter._lama_fill_single_fixed(crop, mask)

    assert np.array_equal(painted, crop)
    assert resize_calls == []


def test_tile_tail_start_is_coverage_not_an_exact_duplicate():
    starts = Inpainter._tile_starts(1024, 512, 448)
    assert starts == [0, 448, 512]
    assert len(starts) == len(set(starts))


def test_recovery_reuses_caller_grayscale(monkeypatch):
    image = np.full((80, 120, 3), 180, dtype=np.uint8)
    gray = np.full((80, 120), 180, dtype=np.uint8)
    recovery = SecondaryTextRecovery.__new__(SecondaryTextRecovery)
    seen = {}

    def fake_extract(image_arg, gray_arg):
        seen["gray"] = gray_arg
        return np.empty((0, 4), dtype=np.int32)

    recovery._extract_primitives = fake_extract

    def forbidden_cvt_color(*args, **kwargs):
        raise AssertionError("provided grayscale page must be reused")

    monkeypatch.setattr(recovery_module.cv2, "cvtColor", forbidden_cvt_color)

    assert recovery.detect(image, gray=gray) == []
    assert seen["gray"] is gray


def test_tall_box_refinement_reuses_caller_grayscale(monkeypatch):
    import app.detector.combined_detector as combined_module
    from app.detector.bubble_detector import BubbleBox

    image = np.full((100, 140, 3), 200, dtype=np.uint8)
    gray = np.full((100, 140), 200, dtype=np.uint8)
    box = BubbleBox(10, 10, 50, 30, 0.9)

    def forbidden_cvt_color(*args, **kwargs):
        raise AssertionError("provided grayscale page must be reused")

    monkeypatch.setattr(combined_module.cv2, "cvtColor", forbidden_cvt_color)
    out = combined_module.CombinedTextDetector._refine_and_split_tall_boxes(
        [box],
        image,
        gray=gray,
    )

    assert len(out) == 1
    assert (out[0].x1, out[0].y1, out[0].x2, out[0].y2) == (10, 10, 50, 30)
