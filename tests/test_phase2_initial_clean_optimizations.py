import math

import cv2
import numpy as np

from app.config import SLICE_MAX_HEIGHT, SLICE_MIN_HEIGHT
from app.downloader.slicer import _find_cut_rows, slice_image_with_layout
from app.detector.bubble_detector import YoloDetector


def _legacy_postprocess(detector, outputs, scale, pad, orig_w, orig_h):
    pad_x, pad_y = pad
    out_arr = np.squeeze(outputs[0])
    if out_arr.ndim == 1:
        out_arr = out_arr[np.newaxis, :]
    if out_arr.ndim == 2 and out_arr.shape[0] < out_arr.shape[1]:
        out_arr = out_arr.T

    has_proto = len(outputs) > 1 and outputs[1].ndim == 4
    if has_proto:
        num_mask_coeffs = outputs[1].shape[1]
        num_classes = max(1, out_arr.shape[1] - 4 - num_mask_coeffs)
        prototypes = outputs[1][0]
    else:
        num_mask_coeffs = 0
        num_classes = max(1, out_arr.shape[1] - 4)
        prototypes = None

    candidates = []
    for pred in out_arr:
        if num_classes == 1:
            conf = float(pred[4])
        else:
            class_scores = pred[4 : 4 + num_classes]
            conf = float(class_scores[int(np.argmax(class_scores))])

        if conf < detector.conf_threshold:
            continue

        cx, cy, bw, bh = pred[:4]
        x1 = (cx - bw / 2 - pad_x) / scale
        y1 = (cy - bh / 2 - pad_y) / scale
        x2 = (cx + bw / 2 - pad_x) / scale
        y2 = (cy + bh / 2 - pad_y) / scale
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(orig_w, x2), min(orig_h, y2)

        if (x2 - x1) >= 4 and (y2 - y1) >= 4:
            mask_coeffs = None
            canvas_box = None
            if has_proto and num_mask_coeffs > 0:
                mask_coeffs = pred[4 + num_classes : 4 + num_classes + num_mask_coeffs].copy()
                canvas_box = (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)
            candidates.append((float(x1), float(y1), float(x2), float(y2), float(conf), canvas_box, mask_coeffs))

    return detector._nms(candidates, prototypes)


def _box_signature(boxes):
    result = []
    for b in boxes:
        mask = None if b.mask is None else (b.mask.shape, int(b.mask.sum()))
        result.append((b.x1, b.y1, b.x2, b.y2, round(float(b.confidence), 7), mask))
    return result


def test_dense_long_page_always_gets_bounded_slices():
    h, w = 16000, 900
    y = np.arange(h, dtype=np.uint16)[:, None]
    x = np.arange(w, dtype=np.uint16)[None, :]
    gray = ((x * 7 + y * 13) % 256).astype(np.uint8)

    cuts = _find_cut_rows(gray, h, w)
    segments = []
    start = 0
    for end in cuts + [h]:
        segments.append(end - start)
        start = end

    assert len(segments) == math.ceil(h / SLICE_MAX_HEIGHT)
    assert max(segments) <= SLICE_MAX_HEIGHT
    assert min(segments) >= SLICE_MIN_HEIGHT
    assert sum(segments) == h


def test_slice_layout_reconstructs_source_rows_without_gaps_or_duplicates(tmp_path):
    height, width = 1900, 64
    source = np.tile(np.arange(height, dtype=np.uint8)[:, None], (1, width))
    source_path = tmp_path / "source.png"
    assert cv2.imwrite(str(source_path), source)

    layouts = slice_image_with_layout(source_path, tmp_path, "slice")

    assert layouts[0].source_y_start == 0
    assert layouts[-1].source_y_end == height
    assert all(
        left.source_y_end == right.source_y_start
        for left, right in zip(layouts, layouts[1:])
    )
    reconstructed = np.vstack([
        cv2.imread(str(layout.path), cv2.IMREAD_GRAYSCALE) for layout in layouts
    ])
    assert np.array_equal(reconstructed, source)


def test_vectorized_postprocess_matches_legacy_detection_output():
    rng = np.random.default_rng(1234)
    det = object.__new__(YoloDetector)
    det.conf_threshold = 0.4

    preds = np.zeros((1, 6, 128), dtype=np.float32)
    rows = preds[0].T
    rows[:, 0] = rng.uniform(30, 980, size=128)
    rows[:, 1] = rng.uniform(30, 980, size=128)
    rows[:, 2] = rng.uniform(5, 180, size=128)
    rows[:, 3] = rng.uniform(5, 180, size=128)
    rows[:, 4:] = rng.uniform(0, 1, size=(128, 2))

    legacy = _legacy_postprocess(det, [preds], 1.25, (13, 17), 800, 700)
    current = det._postprocess([preds], 1.25, (13, 17), 800, 700)
    assert _box_signature(current) == _box_signature(legacy)


def test_vectorized_postprocess_matches_legacy_segmentation_output():
    rng = np.random.default_rng(4321)
    det = object.__new__(YoloDetector)
    det.conf_threshold = 0.2

    preds = np.zeros((1, 9, 96), dtype=np.float32)
    rows = preds[0].T
    rows[:, 0] = rng.uniform(20, 1000, size=96)
    rows[:, 1] = rng.uniform(20, 1000, size=96)
    rows[:, 2] = rng.uniform(8, 160, size=96)
    rows[:, 3] = rng.uniform(8, 160, size=96)
    rows[:, 4] = rng.uniform(0, 1, size=96)
    rows[:, 5:] = rng.normal(0, 0.5, size=(96, 4))
    proto = rng.normal(0, 0.5, size=(1, 4, 32, 32)).astype(np.float32)

    outputs = [preds, proto]
    legacy = _legacy_postprocess(det, outputs, 1.0, (0, 0), 1024, 1024)
    current = det._postprocess(outputs, 1.0, (0, 0), 1024, 1024)
    assert _box_signature(current) == _box_signature(legacy)
