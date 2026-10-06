# PPTX generation with python-pptx — deck-lab field notes

name: pptx
description: Build clean 16:9 PowerPoint decks with python-pptx from scratch (no template): layout grid, palette, fonts/sizes, transitions, charts, and a render-then-measure QA loop that catches text overflow, overlapping boxes, tiny fonts and low-contrast colors. Use when creating or fixing a .pptx deck.

## Workflow (the loop that actually worked)

1. Write the generator script in one pass with a **fixed canvas** and **helper functions** (below). Do not hand-place every shape with raw `Inches()` — centralize the grid.
2. Save the `.pptx`, then render and *measure*, don't just eyeball:
   ```bash
   soffice -env:UserInstallation=file:///tmp/lo_profile --headless --convert-to pdf --outdir out deck.pptx
   pdftoppm -png -r 50 out/deck.pdf out/slide      # 50 dpi is enough to spot overflow
   ```
   The `-env:UserInstallation` flag is required in this sandbox (otherwise LibreOffice's profile write fails).
3. Run `check_deck.py` on the pptx: it flags text that likely overflows its box, textboxes overlapping other text, fonts under 12pt, and contrast ratios under 3.0 (WCAG large-text minimum).
4. Open 2-3 rendered slides with view_image to confirm the checker's flags match what the eye sees (the checker is a heuristic — char-width estimate — so a false positive is possible; a true visual clip is not).
5. Fix in the generator, not the pptx; rerun 2-4 until `check_deck.py` prints "clean".

## Layout: 16:9 grid

- Slide is **13.333 × 7.5 in** (set `prs.slide_width`/`slide_height` — never trust the template).
- One **margin** of 0.6in on all sides → content width 12.13in. All x-coordinates come from `MARGIN` + column offsets; all y from a running cursor or named zones.
- Named zones (16:9, top→bottom): title band 0.4–1.5in (kicker + title + accent rule), content 1.8–6.8in, footer/page 6.9–7.3in. Never start content above 1.8in — the header zone owns 0.4–1.6in.
- For card grids: 3 cards = 3.8in wide × 4.17in pitch (0.37in gutter); 4 stats = 2.85in wide × 3.13in pitch. Row height ≤ 1.0in for list items, ≤ 2.4in for cards with a title + 3 lines of body.
- A slide with more than ~5 text blocks is too dense: split it. Six to eight slides total is the target for a short deck; each slide gets ONE idea.

## Colors

- Light theme: bg `#F7F9FC`, ink `#1F2430`, accent `#0F62FE`, muted `#5B6474`, panel `#E8EEF9`. Dark theme: invert — bg `#0F1420`, ink `#E8EDF5`, accent `#4D9FFF`, panel `#1B2436`.
- **Contrast is a hard rule, not taste**: body text ≥ 4.5:1, large text (≥18pt) ≥ 3:1 against the background it actually sits on. The checker computes this — let it. Common failure: white text on a panel that is "almost white" (contrast 1.0–1.5, invisible). If text sits on a filled panel, contrast is text-vs-panel, not text-vs-slide.
- Muted/secondary text: never go lighter than `#5B6474` on light bg (`#7a7a7a` is the practical floor; `#9aa` fails for body size).
- One accent per deck, max two colors of accent (accent + a success/warn if needed). Do not introduce a third hue per slide.

## Fonts & sizes (what fits on 13.33in at 16:9)

| Role | Size | Notes |
|---|---|---|
| Kicker/eyebrow | 13–14, bold, accent color, ALL CAPS | one line max |
| Slide title | 30–34 | one line max; keep under 34 or it wraps |
| Card/section heading | 18–20 | |
| Body | 15–16 | 15pt is the smallest comfortable body size at 50% zoom |
| Big stat number | 28–32 | |
| Caption under stat | 12–13 | |

- Hard floor 12pt for anything; floor 14pt for body. If you need more than one caption line, widen the box before shrinking the font.
- A line of 16pt Segoe UI holds roughly 100–110 chars across the full 12.1in content width. Bold is ~8% wider than regular — estimate with the checker, not by eye.
- Set `font.name` on every run; mixed fonts within a slide look broken. "Segoe UI" is safe cross-platform for generated decks; "Arial" is the conservative fallback.
- `word_wrap = True` and zero text-frame margins (inset) are non-negotiable for precise sizing; the default insets are 0.1in all around and silently steal width.

(see "Transitions (working recipe, verified)" section below for the concrete zip patch)

## Charts

- `from pptx.chart.data import CategoryChartData` + `slide.shapes.add_chart(chart_type, x, y, cx, cy, chart_data)`.
- 16:9: a column/bar chart needs at least 4.5in × 3.5in; a line chart 5in wide. Put a caption below it (13pt, muted), not inside it.
- Style the chart with `chart.font` / per-series `format.fill` to match the palette; default Office charts are blue-and-dirty-gray and clash with custom palettes.
- For a "stat row" (numbers + label), plain rounded-rectangle cards beat a real chart — a chart implies trend, a stat row implies magnitudes.

## Errors this deck hit, and the fix

1. **Text overflow that looks fine at 100%**: 54pt two-line title in a 1.6in box rendered OK but left no headroom; the checker flagged it. Fix: give title boxes the height of *max possible lines × size × 1.3*, not the current text. Budget per line: `size_pt × 1.3 / 72` inches.
2. **White checkmark on a light rounded-rect**: I drew the white ✓ in a box that had a light fill; contrast 1.05. Fix: the ✓ sits on the *accent* chip — make the chip the accent fill (0.4×0.4in rounded) with white text; the checker only trusts the panel it finds *under* the text, so the chip must actually be drawn before the text and fully contain it.
3. **Checker false negative — text on slide bg**: `bg_of()` returned the slide background instead of the card the text sat on, flagging every card text as low-contrast. Fix: pick the **smallest** filled shape that contains the text box; fall back to slide bg. (This is a checker bug, not a deck bug — but the fix matters because an always-red checker stops being trusted.)
4. **Save path**: `prs.save("deck1.pptx")` fails when run from the wrong cwd; save via `os.path.join(os.path.dirname(os.path.abspath(__file__)), name)`.

## Checklist before calling a deck done

- [ ] `check_deck.py` prints "clean" (no overflow flags, no overlaps, no <12pt, no contrast <3.0).
- [ ] `soffice` conversion produces a PDF with the same page count as the slide count.
- [ ] Every content slide has a header zone (kicker + title + rule) and nothing starts above 1.8in.
- [ ] Each slide states one idea; body text ≤ ~3 short lines per card.
- [ ] No shape exceeds slide bounds; text never within 0.15in of the slide edge.
- [ ] A second generator run reproduces the file (no nondeterminism).

## Helpers

- `make_deck_template.py` — working generator skeleton: helpers `rect()`, `text()`, `header()`, `blank_slide()`, `card_row()`, `build()` plus the light palette constants. Start new decks from it and swap the palette.
- `check_deck.py` — the QA checker described above.
- `add_fade_transitions.py` — post-save zip patch that adds a uniform fade transition to every slide; run it on the finished pptx.

## Transitions (working recipe, verified)

python-pptx has no public transition API. A deck-wide fade works by editing the **saved** file — copy every slide XML, insert `<p:transition><p:fade/></p:transition>` directly before `</p:sld>`, rezip. Use `add_fade_transitions.py deck.pptx` for it. Keep one transition for the whole deck.

## Theme learnings (decks 2 & 3)

- Invert the palette: bg `#0F1420`, ink `#E8EDF5`, accent `#4D9FFF`, panel `#1B2436`, muted `#A8B4C8`. The checker's contrast rule applies the same way — check text against the **panel**, not the slide bg.
- Dark bg bullets: draw the accent chip **before** the text, and draw the bullet glyph (●) in the *background* color on top of the chip — on dark slides "white on accent" reads better than "accent-on-panel".
- `card_row()` (template helper) takes `y, n, height, titles, bodies` — no extra kwargs; passing `bodies=` as a keyword raises `TypeError`.

**Warm light theme (deck 3)** — a third palette to prove the system is not tied to blue: bg `#FBF7EF` cream, ink `#2A231C` warm brown, accent `#C4571A` burnt orange, panel `#EFE6D8` sand, muted `#5C4F3E`. The accent must clear 4.5:1 on the panel, not just the bg — orange-family accents are the ones that usually fail there.

## Card titles: one line, no line breaks

- `card_row()` puts each title in a fixed 0.5in band, and the checker estimates lines from length: a 30-char title at 20pt bold wraps to 2 lines on a 3.8in card → overflow flag. Keep card titles **under ~22 characters** or drop `title_size` to 16. Code-like titles (e.g. `Decimal("0.1") + …`) count characters the same way — abbreviate rather than shrink.

## Checker behavior (read the flags carefully)

- "Textboxes overlap" fires for any two *non-empty* text shapes whose rectangles intersect by more than 0.15in × 0.15in — including a caption placed near a panel that has text on it. Stacking a caption *inside* a panel that carries its own text box is the usual cause: give each panel its own single text frame, or separate the boxes.
- "Low contrast" picks the **smallest filled shape containing the text** as the background; if it reports contrast 1.05 for white text, look for a light panel that actually sits under that box (it was not necessarily the one you intended).
