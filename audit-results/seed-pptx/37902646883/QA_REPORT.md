# Build and QA record

## Files

- `skill_pptx.py` — reusable editable-shape PowerPoint toolkit.
- `SKILL.md` — research-informed design rules, layout selection and code patterns.
- `make_deck.py` — reproducible 8-slide technical example and basic slide-shape bounds audit.
- `quantum_utility_brief.pptx` — generated editable deck on the 127-qubit IBM Eagle / zero-noise extrapolation study.
- `render_final/` — LibreOffice PDF and slide PNG renders for visual inspection.

## Checks run

1. Built with installed python-pptx 1.0.2. All 8 slides generated successfully.
2. Exported through LibreOffice 7.3 using an isolated user profile; `pdftoppm` generated 1600 × 900 PNGs for all slides.
3. Inspected rendered slide dimensions and dark/colored pixel bounding extents via PIL. Slides 2–8 content is within expected page margins. Title slide has intentional full-bleed dark background and no off-canvas decorative rules.
4. Enumerated every text shape, its content and geometry using python-pptx. No shape exceeds the canvas after the title motif was corrected. Re-rendered after modifying title/subtitle geometry; footer and page numbers remain inside the 16:9 canvas.
5. Iteration changes: shortened the title motif rules to fit the 13.333-inch page; moved subtitle down and increased title box depth to reduce possible title/subtitle collision; increased content-slide lead box height and reduced its type from 17 to 15pt to avoid wrap clipping. Rebuilt and rerendered after these changes.

## Human review note

The programmatic pixel/geometry checks are warnings, not proof of legibility. This deck intentionally uses editable text and vector shapes. Before presenting, check the PNGs at full size on the intended display, especially title wraps, the comparison paragraphs and small source citations. Source notes are intentionally compact; expand them in a handout or spoken notes if the deck will be distributed without presenter context.

## Source discipline

Technical claims are based on Kim et al., Nature 618, 500–505 (2023), DOI 10.1038/s41586-023-06096-3. The manuscript's measured 127-qubit hardware, 60 CNOT-layer depth, zero-noise-extrapolation results and qualifications have been reflected with context. The deck explicitly does not promote the paper as a demonstrated practical quantum advantage. Design research and references are documented in `SKILL.md`.
