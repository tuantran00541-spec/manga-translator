# Human-first PowerPoint skill

A reusable editable `python-pptx` toolkit for slides that feel deliberately composed rather than auto-filled. Toolkit: `skill_pptx.py`. Substantive demonstrations: `earth_energy_budget.pptx` and `cloud_feedbacks_brief.pptx`.

## Design approach

**A slide is a designed argument, not a decorated document.** Start with the audience and the thought they should leave with. Write the conclusion as the title; let evidence earn its space. One main idea per slide is an editing heuristic, not an inflexible rule.

**Typography has hierarchy.** Defaults use Lato, a widely available sans serif: about 28–35 pt for the lead, 15–18 pt for key claims, 10–12 pt for explanations, and 8–10 pt for footnotes. Keep a small number of styles, align edges, and use line breaks intentionally. Do not shrink type to rescue excessive copy; edit or split the slide. Text frames, runs, font family, size, alignment, margins and vertical anchors are set explicitly rather than inherited from generic layouts.

**Whitespace is structure.** The 16:9 slide uses consistent ~0.7–0.9 inch margins, deliberate grouping and open space. Use fine dividers and quiet backgrounds before piling on cards, icons or decorative widgets. Space is not a defect to fill.

**Restraint and semantic color.** The paper/ink foundation uses a small palette: rust for emphasis, teal for a distinct data role, gold for secondary highlight. Saturated color has a job; maintain contrast and never make color the only signal. No default purple-blue gradients, ornamental blobs, generic icon rows or fake dashboards.

**Real evidence, honest charts.** Use attributed, substantive data, with units, time period, denominator and uncertainty visible. Avoid 3-D, perspective, unexplained dual axes, excessive gridlines, false precision and truncated bar baselines. Label the point that matters. Use native editable shapes when they communicate a small data series more clearly than a theme-dependent chart.

**Variety should follow the argument.** A title, explanation, genuine comparison and data display solve different problems. Do not repeat the same three-card layout regardless of content. Native PowerPoint text and shapes remain editable by recipients.

**Render, inspect, revise.** The `.pptx` is not finished when saved. Render through LibreOffice, view slides at presentation size and in sequence, inspect text fit, alignment and whitespace. `python-pptx` geometry does not know final rendered glyph wrapping. Autofit can shift typography across platforms and is a fallback, not a substitute for editing.

## Layouts

### `title_slide(prs, title, subtitle, eyebrow=..., byline=...)`
Opening slide with a specific subject, informative subtitle, small section label and restrained motif. Avoid generic promises. The orbital graphic is intentionally subtle and topic-related; remove it if irrelevant.

### `content_slide(prs, title, section, items, sub=None, num=1)`
For a short sequence of explanatory claims. `items` is `(claim, explanation)` pairs. Best for up to three points; if copy grows, regroup or split instead of packing.

### `comparison_slide(prs, title, section, left_title, left_items, right_title, right_items, takeaway, num=1, sub=None)`
For real contrasts: e.g., observations vs. radiative accounting, or amplifying vs. offsetting mechanisms. Item pairs are concise `(heading, detail)`. Two rows per side works well. The takeaway says why the contrast matters. Avoid implying a false binary where evidence is gradational.

### `data_slide(prs, title, section, headline, metric, values, labels, caption, num=1, source=None)`
For a single series of non-negative percentages that form one whole. The bars are native/editable, share a zero baseline and a common scale. It is not a general chart engine: use an appropriate line/scatter/table for time series, relationships or many series. Add uncertainty if material; preserve period, unit and source near the data.

## Code pattern

```python
from skill_pptx import make_deck, title_slide, content_slide

prs = make_deck()
title_slide(prs, "A finding, not a slogan", "What was measured, where and over what period.")
content_slide(prs, "State the slide's conclusion", "01 / EVIDENCE", [
    ("A short claim", "A sourced explanation in plain language."),
    ("A second claim", "Keep the period and denominator explicit."),
], num=2)
prs.save("briefing.pptx")
```

For bespoke needs use `text`, `shape`, `rule`, `header`, `footer` and `stat_callout` as building blocks. For an uncertainty interval or map, create a suitable native design rather than forcing it into a simple bar layout.

## Examples and content provenance

### `earth_energy_budget.pptx`
Nine-slide briefing grounded in IPCC AR6 WGI Chapter 7 (2021): global energy imbalance, heat inventory, budget closure, effective radiative forcing and uncertainty. Figures include:

- Mean imbalance **0.57 [0.43–0.72] W m⁻²** (1971–2018); **0.79 [0.52–1.06] W m⁻²** (2006–2018).
- Energy inventory change **434.9 ZJ** (1971–2018); shares rounded: ocean 91%, land 5%, cryosphere 3%, atmosphere 1%.
- Observation-based change **284 [96–471] ZJ** relative to 1850–1900; independent forcing/response estimates are consistent within assessed uncertainty.
- CO₂ doubling ERF **3.93 ± 0.47 W m⁻²**.

These are AR6-era findings, not live updates. The chart proportions are rounded; use the assessment table and likely ranges for quantitative analysis. Global averages are not local forecasts.

### `cloud_feedbacks_brief.pptx`
A four-slide technical demonstration of how evidence and uncertainty can share space: IPCC AR6 assesses net cloud feedback at **0.42 W m⁻² °C⁻¹**, likely range **0.12 to 0.72**, very likely range **−0.10 to +0.94**. The slide framing distinguishes low-cloud/high-cloud amplifying effects from offsetting effects and avoids presenting one mechanism as the whole result.

## Technical notes

Dependencies: `python-pptx` (tested at 1.0.2), LibreOffice and `pdftoppm` for rendering. Lato was available in the build environment; if a recipient system substitutes another font, rerender and check. PowerPoint text and shape graphics are editable.

```bash
python skill_pptx.py
libreoffice -env:UserInstallation=file:///tmp/lo-pptx-profile --headless \
  --convert-to pdf --outdir . earth_energy_budget.pptx
pdftoppm -png -r 110 earth_energy_budget.pdf preview
```

Review all rendered slides at presentation size: title and body legibility, collisions, chart labels, source text, margins, alignment and reading path. The two sample presentations were rendered to PNG (9 and 4 slides, 1467 × 825 px) and checked programmatically for object bounds and off-slide geometry; no shapes exceeded slide edges. Bounds checking supplements, not replaces, visual inspection.

## Sources informing this skill

- **python-pptx Quickstart** — textboxes, shapes, pictures, tables, slide creation and text extraction: https://python-pptx.readthedocs.io/en/latest/user/quickstart.html
- **python-pptx text API** — margins, wrapping, autofit behavior and run-level typography/alignment: https://python-pptx.readthedocs.io/en/latest/api/text.html
- **python-pptx charts** — native data structure, axes, labels and legends: https://python-pptx.readthedocs.io/en/latest/user/charts.html
- **Edward Tufte, The Visual Display of Quantitative Information** — graphical integrity, comparison, editing and data-ink: https://www.edwardtufte.com/book/the-visual-display-of-quantitative-information/
- **Presentation Zen** — preparation, design and delivery principles: https://www.presentationzen.com/ and https://www.garrreynolds.com/preso-tips
- **Canva Design School, Presentation design** — design resource/examples: https://www.canva.com/learn/presentation-design/
- **IPCC AR6 WGI Chapter 7** — energy budget, assessed values and forcing-response framework: https://www.ipcc.ch/report/ar6/wg1/chapter/chapter-7/
- **IPCC AR6 Figure 7.2** — energy-budget figure and caption: https://www.ipcc.ch/report/ar6/wg1/figures/chapter-7/figure-7-2/
- **IPCC AR6 Figure SPM.1 and SPM.2** — temperature history and attributed warming: https://www.ipcc.ch/report/ar6/wg1/figures/summary-for-policymakers/figure-spm-1/ and https://www.ipcc.ch/report/ar6/wg1/figures/summary-for-policymakers/figure-spm-2/
- **NASA Earth Indicator: Carbon Dioxide** — measurement origins, units, trend and seasonal variation: https://climate.nasa.gov/vital-signs/carbon-dioxide/

These sources ground practical defaults, not immutable laws. The audience, display, accessibility needs and evidence should shape each deck.
