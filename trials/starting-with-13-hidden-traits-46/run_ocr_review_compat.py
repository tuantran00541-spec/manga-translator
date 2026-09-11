from __future__ import annotations

import importlib.util
from pathlib import Path

import app.ocr.service as ocr_service


def _review_crop_from_box(image, box: dict):
    """Proof-only crop helper; OCRService itself remains production-unmodified."""
    if image is None:
        raise ValueError("proof crop image is missing")
    h, w = image.shape[:2]
    pad = 24
    x1 = max(0, int(box.get("x1") or 0) - pad)
    y1 = max(0, int(box.get("y1") or 0) - pad)
    x2 = min(w, int(box.get("x2") or 0) + pad)
    y2 = min(h, int(box.get("y2") or 0) + pad)
    return image[y1:y2, x1:x2]


# The chapter-218 trial runner used a public proof-crop helper that production
# no longer exports. Inject it only for review-sheet generation before loading
# the chapter runner; no OCR inference or manifest behavior is changed.
ocr_service.ocr_crop_from_box = _review_crop_from_box

runner_path = Path(__file__).with_name("run_ocr_review.py")
spec = importlib.util.spec_from_file_location("chapter46_ocr_runner", runner_path)
if spec is None or spec.loader is None:
    raise RuntimeError("could not load chapter OCR runner")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.main()
