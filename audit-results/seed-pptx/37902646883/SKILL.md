# Evidence-led PowerPoint design skill

A reusable Python toolkit for making clean, editable, human-directed technical slides. The package is `skill_pptx.py`; the examples are `quantum_utility_brief.pptx` and its generation script `make_deck.py`.

## The design stance

A slide is a visual argument, not a decorated document. Start with a claim, give it evidence, and remove everything that does not help the audience understand or decide. Human-made work tends to show editorial intent: specific hierarchy, measured use of space, real sourced claims, selective emphasis, meaningful variation between slide structures, and care about the details of alignment and legibility. Avoid defaulting to a repeated row of three cards, ornamental icons, meaningless shapes, invented statistics, generic filler, gradient backgrounds, and the AI-template look.

These defaults are grounded in sources, not simply taste:

- Garr Reynolds / Presentation Zen recommends simplicity and audience-centered storytelling. This motivates a single editorial point per slide, generous whitespace and selective emphasis.
- Edward Tufte, *The Visual Display of Quantitative Information* (Graphics Press overview: https://www.edwardtufte.com/tufte/books_vdqi), emphasizes evidence-rich displays and resisting chartjunk. Use real values and state their provenance; don't decorate data.
- IBM Design Language, Color (https://www.ibm.com/design/language/color/), describes neutral-dominant palettes with color used purposefully, and warns against using color alone to convey meaning. The toolkit uses warm paper, dark ink, one deep green accent, and a restrained warm secondary accent.
- WCAG 2.2, 1.4.3 (https://www.w3.org/WAI/WCAG22/Understanding/contrast-minimum.html) specifies 4.5:1 contrast for ordinary text and 3:1 for large text. Treat those as useful checks for projection as well as accessibility; never use pale gray for essential small text.
- python-pptx's text documentation (https://python-pptx.readthedocs.io/en/latest/user/text.html) describes the text-frame/paragraph/run hierarchy and frame-level wrapping/autofit. Use explicit frame margins, wrapping and font styling rather than hoping defaults behave.
- python-pptx chart documentation (https://python-pptx.readthedocs.io/en/latest/user/charts.html) supports native editable charts, series, axes and labels. Prefer a simple appropriate visual over a chart type chosen for novelty.
- Nature (2023), Kim et al., “Evidence for the utility of quantum computing before fault tolerance,” https://www.nature.com/articles/s41586-023-06096-3, is the primary technical source for the sample deck. It carefully distinguishes an experiment beyond brute-force classical simulation in selected regimes from a demonstrated practical quantum advantage. That distinction is preserved in the sample.
- IEA, *Electricity 2024* executive summary, https://www.iea.org/reports/electricity-2024/executive-summary, is a second example of a primary-source report with dated, attributed forecasts; it illustrates why every statistic needs source and time context.

The toolkit deliberately avoids gradients by default, not because gradients are inherently forbidden, but because they rarely clarify these technical arguments and often become a substitute for hierarchy. If branding requires one, use it sparingly, verify contrast at the weakest point, and document why.

## Install and quick start

```python
from skill_pptx import new_deck, title_slide, content_slide, comparison_slide, data_slide, save

prs = new_deck()  # 16:9, 13.333 x 7.5 in

title_slide(prs, "Noise mitigation changes the useful depth",
            "What a 127-qubit experiment established—and what it did not.",
            kicker="QUANTUM COMPUTING / FIELD NOTE",
            detail="Source: Kim et al., Nature 618, 500–505 (2023)")

content_slide(prs, "What the experiment does establish", [
    ("127-qubit processor", "IBM Eagle device ran circuits with up to 60 layers of two-qubit gates."),
    ("Mitigation matters", "Zero-noise extrapolation improved observable estimates in the tested circuits."),
    ("Bounded claim", "The paper reports evidence for pre-fault-tolerant utility—not practical advantage."),
], page=2, source="Kim et al. (2023), Nature 618, 500–505. doi:10.1038/s41586-023-06096-3")

save(prs, "brief.pptx")
```

## Layout selection

### `title_slide(prs, title, subtitle, kicker, detail)`
Use once at the opening. It is intentionally typographic and quiet: a strong title, a clarifying subtitle, a small context line and one modest motif. Avoid logos or stock visuals unless they are supplied and meaningful. Keep title to two lines at most; do not force long titles into tiny font.

### `content_slide(prs, title, items, ...)`
Use for a sequence of 2–4 related explanations. Each item is a `(lead, body)` pair. The lead gives the scan target; the body explains it. Use parallel grammatical structure and comparable length. If the message has more than four items, split the story, not shrink the type. This differs from a generic bullet list by making the relationship between points explicit.

### `comparison_slide(prs, title, left_title, left_items, right_title, right_items, ...)`
Use when the audience must compare two approaches, conditions, or interpretations against matched criteria. Items are `(criterion, description)` pairs; keep the number and order aligned across columns. State the decisive difference in the title or subtitle. Do not use this as a two-column dumping ground for unrelated material.

### `data_slide(prs, title, metrics, ..., takeaway=None)`
Use for a small number of important numeric facts where values can be understood as metrics, not for dense time-series or distributions. Each metric is `(value, caption)`; label units, period and population in captions or the source line. Add a one-sentence takeaway if useful. For actual time-series / categories, create a native python-pptx chart (or a carefully built editable bar chart); never imply a trend from decorative geometry.

## Design and editing rules

- **Hierarchy:** a short, declarative title should carry the point. A title should say what changed or why it matters, not merely name a topic. Use one display size and one body size family; bold is emphasis, not default.
- **Type:** toolkit uses Lato for robust availability and neutral technical character. Use a familiar sans serif and no more than two type families. As starting points, use ~30 pt slide titles, 15–18 pt body text, 9 pt source labels, and larger numbers only when they deserve the attention. Large-room viewing, localization or long text requires larger sizes and fewer words. Check the actual rendered font substitution on the target system.
- **Whitespace:** keep consistent outer margins and align text to a small set of vertical guides. Whitespace is structure; don't fill every empty region with decoration. The toolkit's 0.72 in base margin and consistent footer establish a quiet grid.
- **Color:** warm off-white background, near-black/green ink, muted gray secondary text, deep green accent, and occasional rust for a contrasting signal. Color should encode emphasis or category, not merely make things lively. Pair hue with labels, shape or position; do not rely on hue alone.
- **Evidence:** cite source, publication/year, units and time period on the slide. Distinguish observed results, estimates and forecasts. Do not fabricate examples, data, quotes, or citations. Make uncertainty visible.
- **Density:** one job per slide. One meaningful diagram or data figure is better than several unrelated icons. Prefer editable native shapes/charts; images should be high-resolution, credited, and genuinely explanatory.
- **Variation:** maintain the same design system, but choose layout based on the argument. A title, explanation, comparison and result should not all be forced into the same three-card grid.
- **Accessibility:** ensure small text reaches at least 4.5:1 contrast where practical. Use large enough font, direct labels, non-color distinctions and plain language. Avoid thin/light type for body copy.

## API notes

`new_deck()` creates a blank 16:9 deck; all layouts use a blank slide to retain explicit placement. `text()` and `rect()` are low-level helpers for custom compositions. `base_slide()` applies the title, section label, source/footer and page number; pass a concise source citation. `metric()` builds the number-and-caption unit. `save(prs, path)` writes an editable `.pptx`.

All coordinates in low-level helper functions are inches. The toolkit imports python-pptx. In a fresh environment: `pip install python-pptx`.

## Quality assurance workflow

1. Render with LibreOffice: `libreoffice --headless --convert-to pdf --outdir render file.pptx`; then `pdftoppm -png -r 120 render/file.pdf render/slide`.
2. Inspect every slide image at presentation size; don't rely on source code or thumbnails alone. Check clipped text, awkward line breaks, clear reading order, alignment, source legibility and contrast.
3. Programmatically enumerate shape bounds against slide bounds and extract text/shape locations. Use pixel scans or OCR as a secondary warning, never as a replacement for human visual review.
4. Re-render after changes. Verify that the deck opens and every claim's source is present.

Sample deck content is a compact, carefully qualified retelling of Nature 2023's 127-qubit experiment: 60 two-qubit-gate layers, zero-noise extrapolation, selected test regimes, and the distinction between utility evidence and application advantage. It is not presented as a newly demonstrated commercial quantum advantage.

## Source URLs

- https://python-pptx.readthedocs.io/en/latest/user/quickstart.html
- https://python-pptx.readthedocs.io/en/latest/user/text.html
- https://python-pptx.readthedocs.io/en/latest/user/charts.html
- https://www.garrreynolds.com/preso-tips/design/
- https://www.presentationzen.com/
- https://www.edwardtufte.com/tufte/books_vdqi
- https://www.ibm.com/design/language/color/
- https://www.w3.org/WAI/WCAG22/Understanding/contrast-minimum.html
- https://www.nature.com/articles/s41586-023-06096-3
- https://www.iea.org/reports/electricity-2024/executive-summary

Some search targets render empty in the research browser. No factual claims above depend on those unavailable pages.