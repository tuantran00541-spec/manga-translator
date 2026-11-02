# Detection Safety Implementation Plan

> **For agentic workers:** Execute inline with test-driven development; do not delegate without user authorization.

**Goal:** Avoid duplicate dynamic-model inference and preserve detected regions whose segmentation fails.

**Architecture:** Keep one Kiuyha inference for the dynamic input path and retain an empty-mask detection as an OCR-eligible, review-only box without granting permission to erase its rectangle.

**Tech Stack:** Python, NumPy, OpenCV, pytest.

**Spec:** `docs/superpowers/specs/2026-11-01-core-audit-remediation-design.md`.

## Global Constraints

- Do not filter text as watermark, and do not erase an unverified rectangle.
- Keep existing chapter data and main unchanged.

### Task 1: Avoid repeated inference

**Files:** `app/detector/kiuyha_detector.py`; `tests/test_kiuyha_detector.py`.

**Interfaces:** `KiuyhaTextDetector.detect_slice(image)` must call `raw_boxes(image)` exactly once when `self.fixed is None` and preserve the padded boxes.

- [x] Add a test using `_Session([1, 3, 'height', 'width'])`, call `detect_slice(_image())`, and assert `len(session.blobs) == 1`.
- [x] Run the test and confirm that the current code calls the session twice.
- [x] Use `_coarse_boxes(image)` alone for the dynamic path and retain the existing combined passes for fixed inputs.
- [x] Run `tests/test_kiuyha_detector.py` and confirm the dynamic and static tests pass.

### Task 2: Retain uncertain detections safely

**Files:** `app/detector/kiuyha_detector.py`; `tests/test_kiuyha_detector.py`.

**Interfaces:** A detection for which `letter_mask` returns an empty mask produces an OCR-eligible `BubbleBox` with `safe_to_inpaint=False` and `needs_review=True`; it must not authorize a rectangle fallback.

- [x] Add a test that stubs `detect_slice` to return `(20, 20, 100, 60, 0.99)` and `letter_mask` to return an empty `(40, 80)` mask, then asserts one review-only candidate is returned.
- [x] Run the test and confirm the existing code discards the candidate.
- [x] Construct the review-only box without a mask and ensure it cannot be joined into destructive mask-bearing boxes, including seam NMS and nested folding.
- [x] Run the detector tests and the cleanup safety tests, including a regression that empty-mask boxes do not authorize a destructive mask.
