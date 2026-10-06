# pptx
name: pptx
description: Build PowerPoint decks from zero with python-pptx — layout grid, palette, typography, shapes, transitions, charts, QC. Use when asked to create or edit a .pptx file in this repo.

Create presentation decks programmatically with `python-pptx` (installed, v1.0.2).
The target of every deck is: it must survive the QC pipeline below with zero issues.

## Pipeline (always, in this order)

```bash
# 1. build
python3 make_deck.py          # writes OUT.pptx
# 2. render
soffice -env:UserInstallation=file:///tmp/lo_profile --headless \
  --convert-to pdf --outdir out/ OUT.pptx
pdftoppm -png -r 50 out/OUT.pdf out/pg
# 3. static + render QC (from deck-lab/check_deck.py, copied next to this skill)
python3 check_deck.py OUT.pptx out/pg
# 4. eyeball the pages with view_image; fix and re-run until CLEAN
```

`-env:UserInstallation=file:///tmp/lo_profile` is mandatory in the sandbox, otherwise
LibreOffice fails on the profile lock.

## The 10 rules

1. **Canvas**: 16:9 = 13.333in × 7.5in. Set it explicitly on the Presentation:
   `prs.slide_width = int(13.333 * 914400)` etc. All coordinates in inches.
2. **Backgrounds**: LibreOffice ignores slide-level `<p:bg>` in files produced by
   python-pptx. Use a **full-bleed rectangle** instead: `add_shape(MSO_SHAPE.RECTANGLE, 0, 0,
   int(13.333*EMU), int(7.5*EMU))`, solid fill, `line.fill.background()`.
3. **Fonts**: use `DejaVu Sans` (available; Liberation Sans also works). Set the font
   name on every run — never rely on theme defaults.
4. **Type scale**: titles 26–42pt bold, card titles 15–20pt bold, body 13–17pt,
   footnotes/sources 13pt bold. Never ship non-bold text under 13pt; nothing under 11pt.
   Body ≥ 13pt or it reads as a footnote and fails the QC floor.
5. **Line height**: python-pptx does not expose spacing well — budget `size * 1.2`
   EMU per line plus `space_after` when estimating.
6. **Grid**: 0.5in page margins; content starts at x=0.5 or 0.9 (card-inset);
   card pitch = card width + 0.2in gap. A 5-card row on 16:9 = 2.35in cards.
   Numbered-list rows: pitch = title box + 0.05 gap + desc box + 0.5in; for 4 items
   in a 6.5in card that's a pitch of ~1.25in (0.6in title + 0.6in desc + 0.05 gap).
7. **Contrast**: WCAG AA = 4.5:1 for text. On a dark deck (bg #14122B),
   near-white #F4F1FF and lavender #B8AEE6 pass; mid-grays and dark accents do not.
   Keep accent text ≥ 5.5:1 or make it large (≥ 24pt, then 3:1 suffices).
8. **No overflow**: `word_wrap = True` always. Estimate each paragraph's height
   (lines × 1.2 × size × EMU_PER_PT + space_after) and keep total ≤ box height.
   When a line must not wrap (a single-line label), set `word_wrap = False` and
   check width yourself.
9. **Overlap**: the checker only flags overlap between *fill* shapes. A card
   (fill rect) under text boxes is the intended pattern; card-to-card overlap is not.
10. **Transitions/animations**: python-pptx has no API. Inject
    `<p:transition>` via lxml **after the `p:cSld` child** (schema order:
    `cSld, clrMapOvr, transition`), or LibreOffice silently drops it:
    ```python
    el = etree.fromstring(f'<p:transition xmlns:p="{P}" spd="med"><p:fade/></p:transition>')
    slide._element.insert(2, el)
    ```
    Valid children: `push fade wipe blinds dissolve`. Animations: unsupported,
    skip — the render pipeline only shows static frames anyway.

## Building blocks

```python
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

EMU = 914400  # per inch

def slide(prs, bg):
    s = prs.slides.add_slide(prs.slide_layouts[6])  # Blank
    r = s.shapes.add_shape(1, 0, 0, 13_333_000, 6_858_000)  # full-bleed bg
    r.fill.solid(); r.fill.fore_color.rgb = bg
    r.line.fill.background()
    return s

def card(s, x, y, w, h, color):
    r = s.shapes.add_shape(1, int(x*EMU), int(y*EMU), int(w*EMU), int(h*EMU))
    r.fill.solid(); r.fill.fore_color.rgb = color
    r.line.fill.background()
    return r

def text(s, x, y, w, h, lines, space_after=6):
    """lines: [(text, pt, color, bold), ...] one paragraph each."""
    tb = s.shapes.add_textbox(int(x*EMU), int(y*EMU), int(w*EMU), int(h*EMU))
    tf = tb.text_frame; tf.word_wrap = True
    for i, (t, pt, col, bold) in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(space_after)
        r = p.add_run(); r.text = t
        r.font.size = Pt(pt); r.font.name = "DejaVu Sans"
        r.font.color.rgb = col; r.font.bold = bold
    return tb
```

Rounded rectangles: `add_shape(5, ...)` and set `shape.adjustments[0] = 0.05`.

## Palettes (proven to pass QC)

### Dark deck

| role | hex | notes |
|---|---|---|
| bg | `#14122B` | deep indigo |
| card | `#231F42` | one step lighter |
| ink | `#F4F1FF` | body, headings |
| muted | `#B8AEE6` | secondary text, captions |
| accent | `#9F7AF0` | titles, numerals |
| accent2 | `#5EE6C8` | highlights, labels |

### Light deck (ship30, measured)

| role | hex | measured on card/bg |
|---|---|---|
| bg | `#F6F4FB` | — |
| card | `#FFFFFF` | — |
| ink | `#241E3C` | 15.8:1 on card |
| muted | `#5A5378` | 6.5:1 on bg, 7.1:1 on card |
| accent | `#6E46E0` | 5.3:1 on bg |
| accent2 | `#0A6B55` | 6.0:1 on bg |

**Lesson**: green accents look great in dark decks and fail contrast on light
ones. Re-derive accent2 per theme — don't reuse the dark-deck hex verbatim.
Rounded cards on light: `add_shape(5, ...)`, `adjustments[0] = 0.04`, 1pt border
`#DDD8F0` — the border is what separates white cards from a near-white bg.

## Charts (native, not images)

`slide.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, x, y, w, h, category_data)`
where `category_data` is `CategoryChartData` with one series per bar group.
After adding: style each series' `fill.solid().fore_color.rgb`; keep
`has_legend = True` — the QC script can't measure chart text, so the legend is
your only QC-safe place to name the series. One accent + one gray is usually
legible; still, name them.
**Legend text is real text.** The QC script ignores it, but a wrapped label
like "Interruptions per day" wraps onto two lines in the legend and collides
with the bars on a 7.5in-wide chart. Keep each series name ≤ 12 characters
("Interruptions" + a sub-note, not "Interruptions per day").
Charts are XML — keep axis labels short (W1..W5, not full words) and cap
the category count at ~6. Pair every chart with a side-note box (4in wide,
start x=9.0) so the takeaway is prose, not bar-reading.

## Pitfalls collected from real failures

- **`RGBColor(0x333333)` does not work** — it takes three 8-bit args:
  `RGBColor(0x33, 0x33, 0x33)`.
- **Never trust "saved"** — `make_deck` once returned a success string with no
  `prs.save(path)` call. After any build, `os.path.exists(path)` before QC.
- **Glyph boxes overflow**: a `>` or `★` at 22pt inside a 0.5in box fails the
  `size*1.2` budget. Keep single-glyph runs ≤ 18pt in boxes ≤ 0.5in.
- **Pitch formula bug**: `min(1.25, 4.6/n)` is valid; `min(1.25, 4.6)/n` is not.
  Write `4.6/max(n,1)` explicitly; for n>4 rows use `4.6/n`, cap at 1.25.
- **Checker false positives on bg**: a 0,0,full-size rectangle is a background,
  not text — exempt by geometry, never by shape name.
- **`MSO_AUTO_SIZE`** — never set `TEXT_TO_FIT_SHAPE` on a box the checker measures;
  it changes geometry silently. Keep sizes static and predictable.
- **Placeholder layouts (0–5) carry a title box you didn't ask for.** Use layout 6
  (Blank) and place everything yourself.
- **`space_before`/`space_after` in `text()`**: the checker only counts `space_after`;
  keep it ≤ 6pt on dense slides or budget the extra.
- **One slide, one message.** If a slide needs two, split it.
- **Numbers**: use bold accent numerals (22pt+) for step lists — they pass size floors
  and read as structure.

## QC script

`check_deck.py` (copied into this skill folder alongside the deck source) flags:
1. Estimated text overflow vs box height (TTF-measured, DejaVu).
2. Non-bold text < 13pt anywhere; any text < 11pt.
3. Fill-shape overlap > 30% of the smaller shape.
4. Text/bg contrast < 4.5:1 (relative luminance).
5. Shapes extending past the slide bounds in the rendered PNG.
6. Uneven corner pixels on a rendered page (broken bg).

Run it; when it prints `=== CLEAN ===` and the `view_image` pages agree, ship it.

## Deck anatomy that works (deck 1: agents_2025)

S1 title (big type, 3 lines) · S2 context (one card, 4 bullets) · S3 shortlist
(5 cards in a row) · S4–S5 deep dives (two-panel) · S6 numbered checklist ·
S7 decision rows (row cards, 3 columns) · S8 closer + sources.

## Deck 3 (deepwork) — numbered-row audit variant

Same shape, but S6 is the daily-audit table: 4 numbered rows at the skill's
1.25in pitch (0.6in number + 0.6in title + 0.6in desc + 0.05in gap). Pitch
anything tighter and the desc box overflows ~0.03in — the number column is
the tell: if the numeral's box starts eating the title's space, pitch is too
small. The lesson: numbered-list rows are the most pitch-sensitive layout in
the skill; 1.25in is the proven number, don't tune it per-deck.

Same shape, plus: rounded card with hairline border, native clustered column
chart paired with a 4-column side-note box, numbered rows with a 1.25in pitch
(0.6in title + 0.6in desc + 0.05in gap — anything less and the desc box
overflows by ~0.03in, the most common off-skill failure).

S1 title (big type, 3 lines) · S2 context (one card, 4 bullets) · S3 shortlist
(5 cards in a row) · S4–S5 deep dives (two-panel) · S6 numbered checklist ·
S7 decision rows (row cards, 3 columns) · S8 closer + sources.

That's the default shape; deviate when the content needs it.
