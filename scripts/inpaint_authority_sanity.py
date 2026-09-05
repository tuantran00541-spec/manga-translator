from __future__ import annotations

import argparse
from pathlib import Path
import sys


def check_runtime_geometry() -> None:
    """Optional pixel-level checks; no model weights or LaMa inference needed."""
    import tempfile
    import threading
    from types import SimpleNamespace
    from unittest.mock import patch

    import cv2
    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from app.detector.bubble_detector import BubbleBox
    from app.detector.combined_detector import CombinedTextDetector
    from app.downloader import slicer
    from app.inpaint.lama_inpainter import Inpainter

    image = np.full((220, 340, 3), 100, np.uint8)
    mask = np.zeros((180, 300), np.uint8)
    for y, color in ((10, (5, 5, 5)), (75, (210, 245, 245)), (140, (5, 5, 5))):
        image[20 + y:36 + y, 50:250] = color
        mask[y:y + 16, 30:230] = 255
    box = BubbleBox(20, 20, 320, 200, .8, mask,
                    source_model="text_segmenter.onnx", safe_to_inpaint=True)
    refined = CombinedTextDetector._refine_and_split_tall_boxes([box], image)
    restored = np.zeros(image.shape[:2], np.uint8)
    for item in refined:
        restored[item.y1:item.y2, item.x1:item.x2] |= item.mask
    expected = np.zeros_like(restored)
    expected[20:200, 20:320] = mask
    assert np.array_equal(restored, expected), "Line refinement lost mask evidence"
    assert len(refined) == 1, "Mixed-polarity text must retain its original box"

    dark_mask = mask.copy()
    dark_mask[75:91] = 0
    dark_box = BubbleBox(20, 20, 320, 200, .8, dark_mask,
                         source_model="text_segmenter.onnx", safe_to_inpaint=True)
    dark_lines = CombinedTextDetector._refine_and_split_tall_boxes([dark_box], image)
    assert len(dark_lines) == 2, "Lossless dark-text splitting should remain available"
    unmasked = BubbleBox(20, 20, 320, 200, .3)
    assert all(item.mask is None for item in
               CombinedTextDetector._refine_and_split_tall_boxes([unmasked], image))

    proposal = BubbleBox(20, 20, 320, 200, .3,
                         source_model="bubble_yolo.onnx", semantic_type="free_text")
    detector = CombinedTextDetector.__new__(CombinedTextDetector)
    detector._metrics_local = threading.local()
    detector.bubble_detector = SimpleNamespace(detect=lambda _: [proposal])
    detector.recovery = SimpleNamespace(detect=lambda *a, **kw: [])
    for rgb_boxes, gray_boxes, calls, safe in (
        ([], [box], 2, True), ([box], [], 1, True), ([], [], 2, False),
    ):
        with patch("app.detector.combined_detector.DETECTOR_FREE_TEXT_GRAYSCALE_FALLBACK", True):
            with patch.object(detector, "text_detector", create=True) as text_detector:
                text_detector.detect.side_effect = [rgb_boxes, gray_boxes]
                result = detector.detect(image)
                assert text_detector.detect.call_count == calls
                assert any(item.safe_to_inpaint for item in result) == safe
                assert detector.last_metrics()["text_grayscale_fallback_runs"] == calls - 1
    print("Free-text fallback and lossless mask refinement OK")

    height, width = 6200, 120
    gray = np.tile(np.linspace(35, 220, width, dtype=np.uint8), (height, 1))
    dense = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    with patch.multiple(slicer, SLICE_TARGET_HEIGHT=2400, SLICE_MIN_HEIGHT=800,
                        SLICE_MAX_HEIGHT=4096, SLICE_SEARCH_WINDOW=360, OVERLAP_CONTEXT=768):
        with tempfile.TemporaryDirectory(prefix="slice-geometry-") as temporary:
            folder = Path(temporary)
            source = folder / "raw.png"
            assert cv2.imwrite(str(source), dense)
            parts = slicer.slice_image(source, folder, "slice", return_metadata=True)
            assert len(parts) == 2, "Dense 6200px page must not become five short slices"
            restored = []
            cursor = 0
            for part in parts:
                pixels = cv2.imread(str(part["path"]))
                assert part["core_source_y1"] == cursor
                owned = pixels[part["core_y1"]:part["core_y2"]]
                assert 800 <= len(owned) <= 4096
                assert np.array_equal(pixels, dense[part["source_y1"]:part["source_y2"]])
                restored.append(owned)
                cursor = part["core_source_y2"]
            assert cursor == height and np.array_equal(np.vstack(restored), dense)
            assert parts[0]["source_y2"] - parts[0]["core_source_y2"] == 768
            assert parts[1]["core_source_y1"] - parts[1]["source_y1"] == 768

        unsafe = np.ones(height, dtype=bool)
        unsafe[1820:1980] = False
        unsafe[4220:4380] = False
        cuts = slicer._find_cut_rows(gray, height, width, unsafe_rows=unsafe)
        assert len(cuts) == 2
        for cut in cuts:
            assert not unsafe[cut - slicer.SAFE_CUT_BAND:cut + slicer.SAFE_CUT_BAND + 1].any()
        with patch.multiple(slicer, SLICE_MIN_HEIGHT=4096):
            cuts = slicer._find_cut_rows(gray, height, width)
            assert max(np.diff([0, *cuts, height])) <= 4096
    print("Long-slice bounds, safe gutters, overlap and pixel-exact stitching OK")

    for dynamic in (False, True):
        painter = Inpainter()
        painter.session = object()  # Explicit geometry stub, not a model benchmark.
        painter.dynamic_lama = dynamic
        calls = []

        def paint(canvas, mask_canvas):
            calls.append(canvas.shape[:2])
            result = canvas.copy()
            result[mask_canvas > 127] = 201
            return result

        painter._run_lama = paint
        image = np.full((203, 716, 3), 17, np.uint8)
        mask = np.zeros(image.shape[:2], np.uint8)
        mask[60:140, 40:680] = 255
        painted = painter._lama_fill(image.copy(), image.copy(), mask, (0, 0, 716, 203))
        assert len(calls) == 2 and all(w == 512 for h, w in calls)
        assert np.array_equal(painted[mask == 0], image[mask == 0])
        assert np.all(painted[mask > 127] >= 200), "Tiled blend left an unpainted seam"
        calls.clear()
        with patch.object(painter, "_lama_fill_tiled", side_effect=AssertionError("Small crop tiled")):
            painter._lama_fill(image[:, :400].copy(), image[:, :400].copy(),
                               mask[:, :400], (0, 0, 400, 203))
        assert len(calls) == 1
    print("Fixed/dynamic LaMa tile routing and outside-mask pixel preservation OK")


def _require(path: str, *needles: str) -> None:
    source = Path(path).read_text(encoding="utf-8")
    missing = [needle for needle in needles if needle not in source]
    if missing:
        joined = "\n  ".join(missing)
        raise SystemExit(f"{path} is missing inpaint-authority contract markers:\n  {joined}")


def main() -> None:
    _require(
        "app/pipeline.py",
        "Persisted review-only detector masks are evidence, not erase",
        "if overlap_context_only and not geometry_overridden:",
        "if not (safe_to_inpaint or geometry_overridden or explicit_manual):",
        "safe_to_inpaint=safe_to_inpaint",
        "if geometry_overridden or explicit_manual:",
        "box_object.allow_rectangle_fallback = True",
        "manual_lama_mask_posix: str | None = None",
        "(manual_mask_path, False)",
        "(manual_lama_mask_path, True)",
    )
    _require(
        "app/detector/mask_builder.py",
        "AUTO_DESTRUCTIVE_MASK_SOURCES = frozenset(",
        '"text_segmenter"',
        '"bubble_flat_contrast"',
        '"opencv_mser"',
        "def is_destructive_box_authorized(box: BubbleBox) -> bool:",
        'getattr(box, "safe_to_inpaint", False)',
        "or _rectangle_fallback_allowed(box)",
        "if not is_destructive_box_authorized(box):",
        "Skipping non-authorized destructive mask",
    )
    _require(
        "app/detector/bubble_detector.py",
        "def _merge_text_mask_evidence(boxes: list[BubbleBox]) -> BubbleBox:",
        "Only verified",
        'if "text_segmenter" not in source_name:',
        "buckets[target].append(box)",
    )
    _require(
        "app/detector/combined_detector.py",
        'mask_source="bubble_flat_contrast"',
        "safe_to_inpaint=True",
        "YoloDetector._nms_box_group(",
        "np.any((box.mask > 0) & ~covered_mask)",
        "DETECTOR_FREE_TEXT_GRAYSCALE_FALLBACK and unmatched_free_text",
    )
    _require(
        "app/inpaint/lama_inpainter.py",
        "source_model=b.source_model",
        "mask_source=b.mask_source",
        "safe_to_inpaint=bool(b.safe_to_inpaint)",
        "ocr_eligible=bool(b.ocr_eligible)",
        "needs_review=bool(b.needs_review)",
    )
    _require(
        "scripts/model_e2e_gate.py",
        "from app.detector.mask_builder import AUTO_DESTRUCTIVE_MASK_SOURCES",
        "mask_source not in AUTO_DESTRUCTIVE_MASK_SOURCES",
        "source_model=box.source_model",
        "mask_source=box.mask_source",
        "safe_to_inpaint=bool(box.safe_to_inpaint)",
        "ocr_eligible=bool(box.ocr_eligible)",
        "needs_review=bool(box.needs_review)",
        "if not args.allow_empty_cleanup:",
        '"model E2E produced no authorized cleanup mask pixels; inpaint path was not exercised"',
        '"cleanup_evidence": cleanup_evidence',
        'box_counts["ocr_eligible"] += int(bool(record.get("ocr_eligible")))',
    )
    print("Inpaint destructive-authority source contract OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inpaint authority and optional pixel geometry checks")
    parser.add_argument("--runtime", action="store_true", help="Also check pixels with NumPy/OpenCV; no model weights required")
    args = parser.parse_args()
    main()
    if args.runtime:
        check_runtime_geometry()
