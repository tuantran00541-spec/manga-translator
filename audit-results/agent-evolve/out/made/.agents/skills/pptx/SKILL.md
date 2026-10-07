---
name: pptx
description: Build professional PowerPoint decks with python-pptx - layout, color, typography, charts, self-check
---

# SKILL: building professional .pptx decks with python-pptx

Learned from expert slide-design sources (deckary, slidor, slide-deck.io, a1slides, gooddata data-viz guides) and verified by building 3 decks in `deck-lab/`.

## Core rules (what separates pro from amateur)

1. **One idea per slide.** If the title contains "and", split it.
2. **Action titles, not topic labels.** Title states the insight: "Revenue grew 34% YoY", not "Q3 Revenue".
3. **6x6 rule.** Max 6 bullets, 6 words each. If you must shrink text to fit, remove content instead.
4. **Two fonts max** (sans-serif for slides: Arial/Calibri/Helvetica family). Here use `DejaVu Sans` (installed) for both roles; differentiate with weight/size.
5. **Four color roles max**: background (white/near-white), primary text (near-black), accent (brand color), highlight. Charts reuse the same palette; one series colored, others muted.
6. **Contrast >= 4.5:1** for text under 18pt (WCAG AA; large text >= 3:1). Gray-on-white is the classic failure - use dark text (e.g. #2B2B2B), not #808080.
7. **Grid + margins.** 16:9 slide = 13.333 x 7.5 in. Left/right margin ~0.7in. Align every element to the margin or to other elements; no "almost aligned".
8. **White space is design.** 20 words/slide is minimal, 20-50 moderate, 50+ is a document - split the slide.
9. **Consistency = professionalism.** Same title position, same footer position, same card sizes, on every slide. Recurring elements (footer, kicker) always in the same place. Recurring elements repeat as-is: a card strip is ALWAYS the same color on every card - no alternating colors that imply a data coding that does not exist.
10. **Charts: clarity over decoration.** Bar = comparison, line = trend over time, scatter = correlation. Table = exact values. Avoid pie unless 2-3 slices. Bars start at 0, single color with one accent highlight, data labels on, minimal gridlines. Never 3D. Color is never the only cue: write the coding into the labels or add an inline legend ("supply (teal) / demand (slate)").

## Layout & grid (16:9)

- Slide: `prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)`; use blank layout `prs.slide_layouts[6]`.
- Margins: `ML = Inches(0.7)`, content width `CW = SW - 2*ML`.
- Standard skeleton for a content slide:
  - Kicker (11pt, accent, UPPERCASE) at y=0.45
  - Headline (26pt bold, dark navy) at y=0.75, height 0.75in
  - Accent underline bar at y~1.5 (1.2in wide, 3pt tall)
  - Content zone y=2.0 to 6.6
  - Footer (11pt, right) at y=7.08; gray on light bg, light-tint on dark bg
- Card rows: even spacing; card = light-fill rect + thin top strip + tag/head/body text inside with 0.25in padding.
- Two-column text zone: cap each column ~4.4-4.6in wide so body at 14-15pt wraps to 3-4 lines, not 6.

## Color palettes that work (pre-verified)

| Role | Hex | Use |
|---|---|---|
| Ink | #2B2B2B | body text (15+:1 on white) |
| Muted | #6E7B8A | kickers, captions - >= 11pt only, avoid for body |
| Deep navy | #1B3A5C | titles, dark slide bg |
| Accent blue | #4A90D9 | tags, secondary chart series |
| Accent orange | #E8792B | kicker, highlights, callout strip |
| Light fill | #F2F5F9 | card backgrounds |

On dark bg (#1B3A5C): white text, light blue #8FB4DD for secondary, orange for accents. Alternate palettes (fresh hue per deck, same roles): teal #0E7C7B / gold #D9A441 / slate #6B7A8F on white, ink #1F2933.

## Typography

- Title 26pt / big stat 64pt (box height >= 1.2in) / section head 20pt / body 15-16pt / caption 11pt.
- Body floor: 12pt minimum in practice, 15pt recommended. Never go under 11pt on a projected deck.
- Line spacing 1.2-1.3 for multi-line body; 1.0 for labels.
- `word_wrap=True`, set textbox margins to 0, estimate lines: `chars_per_line = width_in*72 / (pt*0.6)` (DejaVu Sans avg char width ~0.6em).
- 80pt numerals overflow visually even when the box math is fine - keep hero numbers at 64pt with a 1.2-1.4in box.
- Card body text: 13pt minimum (12pt in a 2.6in-tall card wraps to 4+ lines and reads small).

## Charts (python-pptx native charts are finicky - hand-draw bars)

Hand-drawn bars are more reliable: rect shapes for bars, textboxes for value + category labels, one thin baseline rect. Rules:
- Baseline rect at `base`; bar heights = `chart_h * value/max_value`; chart zone y=2.0 to 5.4 max so value labels clear the headline.
- Value labels ABOVE the bar (0.35in headroom; the tallest bar's label must not hit the title).
- Category labels BELOW the baseline; leave 0.5in for 2-line wraps.
- Chart label strategy, in order of preference:
1. Shorten labels so each fits under its bar in one line (best).
2. Number the bars (1, 2, 3...) above or inside, plus ONE compact key line below the axis: "1-2 dialog (clean/art), 3 caption, 4 sign, 5 SFX, 6 tiny".
3. A full key line is still needed for color coding: "blue = reliable, plum = needs a second pass" - state it in text, never in swatches next to the caption.
Never stack >2 key lines: merge into line 2. Never let a key line start within 0.05in of the previous - the checker will flag it.
- Pair every chart slide with a "HOW TO READ" column on the right: one bold insight (17-18pt) + one supporting paragraph (14pt), width ~4.4in.

## Self-check pipeline (MANDATORY after every build)

```bash
python deck-lab/check_pptx.py OUT.pptx 11          # overflow / overlap / small-font / contrast
soffice -env:UserInstallation=file:///tmp/lo_profile --headless --convert-to pdf --outdir OUT/ OUT.pptx
pdftoppm -png -r 50 OUT/OUT.pdf OUT/p               # then view_image each page
```

`check_pptx.py` (copy: .agents/skills/pptx/scripts/check_pptx.py) reloads the saved .pptx and reports:
- text overflow (estimated rendered height vs box height),
- shape off-slide,
- font below minimum,
- contrast ratio below 4.5:1 for text < 18pt,
- overlapping sibling text boxes.

Fix every finding, re-render, view the PNGs - eyeballing catches things the heuristic misses (wrapping labels, big-stat vertical position, label collision, color coding that implies data that is not there).

## Common mistakes seen while building

- Big-number textboxes are taller than they look: a 64pt line needs ~1.1in; anchor it so the baseline sits above the stat card.
- Multi-line category labels under bars wrap to 2-3 lines -> reserve 0.5in and check the tallest value label against the title.
- Two boxes that "look" side by side actually overlap horizontally: 0.1in gaps are fine, 0 overlap is required.
- Gray footer on a dark slide has terrible contrast: light-bg slides use gray footer, dark slides use light-tint footer.
- Chart caption under the x-axis labels is easy to forget - it's the "what is this number" context.
- Legend swatches + a centered axis caption in the same row always collide: pick the caption line only.
- 6 bars with 10+ char labels wrap to 2-3 lines under bars; prefer short labels + a numbered key line instead of 2-line category labels.
- A secondary subtitle on a dark title slide needs light tint (e.g. #F0DCE7 on plum), not mid-luminance - mid tints fall below 4.5:1 on mid-dark bgs.
- Highlighting one card in a row of identical cards (gold strip, others teal) reads as data coding - keep the strip color uniform across the row; reserve accent for a genuine emphasis.
- A two-column text zone with 8.5in-wide paragraphs makes 15pt wrap to 6 lines; cap column text width around 4.5in.

## File layout

- `deck-lab/check_pptx.py` - the self-check script (arg: pptx path, optional min font).
- `deck-lab/build_deck1_pptx.py` - reference build (Deep Work, 7 slides, navy/orange).
- `deck-lab/build_deck2_pptx.py` - water deck, built strictly from skill v1 (taught the skill the card-strip uniformity + label-collision rules).
- `deck-lab/build_deck3_pptx.py` - manga OCR deck, built strictly from skill v2 (taught it the key-line and card-body-size rules).
- Rendered PNGs: `deck-lab/outN/p-*.png` (50 dpi).
