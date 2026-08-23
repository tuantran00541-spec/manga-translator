# S22 — manual brush controls

Baseline: `manga-translator-patched-s21.zip`

Implemented from `brief_optimize_manual_brush_v2.md`:

- Keeps the S21 toolbar above each image.
- Adds an 8–80 px brush-radius slider beside the manual repaint controls.
- Preserves the previous auto-calculated brush size as the initial value.
- Magic-wand/flood-fill selections are additive: repeated double-clicks union their selected regions.

Not implemented yet:

- Edge-snap: the brief explicitly requires measuring the algorithm on the two reference images before coding it.
- Local inpaint preview endpoint: not included in this UI-only pass because it requires a separate backend preview contract and should remain isolated from submit/repaint.
- Backend inpaint algorithm remains unchanged.
- Slicer, OCR, detector, mask dilation/feather/crop behavior remain unchanged.
