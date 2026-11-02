# Render Integrity Implementation Plan

> **For agentic workers:** Execute inline with test-driven development; do not delegate without user authorization.

**Goal:** Preserve every rendered object across slice edges and prevent OCR-derived sizing from bypassing fit checks.

**Architecture:** Keep the existing in-place image contract throughout page rendering. Treat OCR size as a hint rather than an explicit user-selected size; validate explicit numeric sizes before drawing.

**Tech Stack:** Python, Pillow, NumPy, pytest.

**Spec:** `docs/superpowers/specs/2026-11-01-core-audit-remediation-design.md`.

## Global Constraints

- Never filter detected text as a watermark before cleaning or translating.
- Preserve existing chapter data and avoid modifying main or unrelated files.
- Add regression tests before implementation and keep each change independently testable.

### Task 1: Preserve the caller's image across slice edges

**Files:** `app/render/page_renderer.py`; `tests/test_render_regions.py`.

**Interfaces:** `_render_in_region(image, text, box, **style) -> Image.Image` must return and mutate the original image even when a box crosses a slice boundary.

- [x] Add this regression to `tests/test_render_regions.py`:

```python
def test_edge_render_mutates_the_input_image():
    image = Image.new("RGB", (400, 400), "white")
    result = _render_in_region(image, "Edge text", (50, 350, 350, 430), **STYLE)
    assert result is image
    assert np.any(np.asarray(image) != 255)
```

- [x] Run `python -m pytest tests/test_render_regions.py::test_edge_render_mutates_the_input_image -q` and confirm failure.
- [x] Replace the final crop return with `image.paste(canvas.crop((0, top, width, top + height)))` followed by `return image`.
- [x] Run `python -m pytest tests/test_render_regions.py -q` and confirm all tests pass.

### Task 2: Guard numeric font sizes and OCR fallback

**Files:** `app/render/page_renderer.py`; `app/render/text_renderer.py`; `tests/test_render_regions.py`; `tests/test_ocr_render_metadata.py`.

**Interfaces:** Existing explicit `font_size` values remain available but must not silently overflow; OCR sizes remain hints and are reduced when necessary.

- [x] Add a regression that draws a long sentence with `font_size=80` inside a `120x70` region and asserts `ValueError` before drawing outside the region.
- [x] Add an object-level test with `style.fontSize='auto'`, `ocr_font_size=36`, and a long translation; assert the fitted result is drawn inside the region.
- [x] Run both tests and confirm they fail for the audited behavior.
- [x] Validate explicit sizes with `_fits(...)` before drawing, and reduce OCR hints only when they do not fit.
- [x] Run the render region and OCR metadata test files, then report any existing expectation that conflicts with the no-overflow contract.

### Follow-on plans

Plan bubble segmentation, detector and cleaning, AI-mode reliability, and gateway/security as separate deliverables against the approved spec after completing this plan. Validate destructive cleaning decisions on the recorded real-chapter evidence before changing defaults.
