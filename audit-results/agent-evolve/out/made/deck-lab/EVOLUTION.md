# pptx_lab plugin evolution log

Goal: evolve a plugin (make_deck + check_deck tools) that builds professional decks from JSON specs, applying the rules learned in `.agents/skills/pptx/SKILL.md`. After each round: build ≥3 decks across 3 different layout types (chart / process / compare), run check_deck, visually audit renders, then rewrite the plugin.

---

## Round 1 — initial plugin (make_deck + check_deck v1)

**Decks built:**
- `deck-lab/lab1_climate.pptx` — chart layout (emissions intensity, 4 bars)
- `deck-lab/lab2_onboarding.pptx` — process layout (two process slides + compare)
- `deck-lab/lab3_coffee.pptx` — compare layout (coffee vs tea, 3 rows)

**Layout types covered:** chart, process, compare (requirement met in round 1).

**Defects found by check_deck:** 1 (note column overflow on chart slide: 17pt insight in a 4.2in×1.3in box wrapped to more lines than fit).

**Defects found by visual audit (soffice → pdftoppm → explore agent):**
1. **Chart bar heights wrong** — `int(h * v)` truncated the float and the value labels (85/78/72/61%) did not track the bar heights; also bars appeared to "hang" without a shared baseline. Fix: keep float height `h*v`, draw a baseline rule at `base`.
2. **Note column overflow** — "How to read" block: insight at 17pt was too wide for 4.2in×1.3in. Fix: shorten note_body to 2 sentences for the insight box, move remainder to a 14pt supporting paragraph in a 1.6in-tall box.
3. **Process arrows overlapped card text** — the "→" glyph was placed at `x + cw - 0.05` (inside the card's right edge), colliding with the next card. Fix: move arrow into the gap between cards (`x + cw + (gap-0.3)/2`), widen gap to 0.4in.
4. **Inconsistent card header colors in process** — first card used navy, the rest blue. Fix: uniform blue `#4A90D9` header band on every step.
5. **Dark-slide footer/subtitle low contrast** — footer `#4A90D9` on `#1B3A5C` and subtitle `#4A90D9` on dark bg both fell below comfortable readability. Fix: light-blue `#8FB4DD` for secondary text on dark bg, footer `#A9C4E0`.
6. **Takeaway bullets (dark)** — `#8FB4DD` on navy read as dim. Fix: bump to `#A9C4E0`, 20pt.
7. **Unbalanced lower band on content slides** — cards/process left a big empty lower third. Fix: add a `statrow` layout (big number + caption cards, 44pt numerals) usable below a headline.

**Plugin v2 written:** float bar heights + baseline rule; note column resized (2-sentence insight + 2-sentence support); arrows in gaps; uniform header color; contrast bumps on dark slides; statrow layout added; footer/subtitle tints raised.

---

## Round 2 — plugin v2, decks rebuilt

**Decks rebuilt with v2:** same three files (climate / onboarding / coffee).

**check_deck:** all three clean (0 problems).

**Visual audit of v2 renders:**
1. **Dark bg tints were still deck-specific** — `#A9C4E0` was tuned for navy; on coffee's brown `#3A2B2B` it read slightly off. Also process card body 13pt on `#F2F5F9` was borderline small.
2. **Chart color semantics** — the 4th bar auto-colored orange (accent), which implied "this bar is special" with no key explaining why. Fix: bars default to a muted blue shade ramp (`#3D5A80 → #8FB4DD`); accent color only when the spec explicitly passes `color`.
3. **Chart category labels 11pt** — small next to 12pt value labels; raised to 12pt, value labels to 13pt bold dark ink.
4. **Compare rows** — 0.9in row pitch with 13pt text left ragged bottoms; raised row height to 0.85in at 14pt.
5. **Card/process body text 13pt** — raised to 14pt across card rows, process cards, compare rows (the 13pt floor from the skill was actually the *minimum*, and 14pt reads better in 2.4in cards).

**Plugin v3 written:**
- `_dark_tint(bg)` helper: picks light text color from the slide's actual dark bg luminance (`#DCE7F5` for blue-darks, `#E8E2D8` for warm-darks) instead of one hardcoded tint.
- Bars use a 4-step blue shade ramp by default; accent reserved for explicit `color` in data.
- Category labels 12pt, value labels 13pt bold `#1B2A3A`.
- Compare rows 14pt in 0.85in rows.
- Card/process body 14pt.
- Takeaway bullets get a "- " marker, 20pt, aligned to the left margin (was indented).
- statrow cards: highlight strip only on the *last* card (emphasis), others blue.

---

## Round 3 — plugin v3, decks rebuilt (final)

**Decks rebuilt with v3:**
- `deck-lab/lab1_climate.pptx` (6 slides: title, agenda, cards, chart, statrow, takeaway)
- `deck-lab/lab2_onboarding.pptx` (7 slides: title, section×2, process×2, compare, takeaway)
- `deck-lab/lab3_coffee.pptx` (5 slides: title, compare, chart, statrow, takeaway)

**check_deck results:**
- lab1_climate: clean
- lab2_onboarding: clean
- lab3_coffee: clean

**Final visual audit (hard defects only — overflow / overlap / off-slide / <11pt / low contrast):** all 18 slides, no defects found.

**Remaining known limitations (documented, not bugs):**
- Transitions are injected as `p:transition` XML (fade/push/wipe honored by LibreOffice render as PDF single-page, so not visible in PDF; visible in PowerPoint). No per-shape animations yet — a v4 could add appear/fade entrance XML.
- `_dark_tint` uses a luminance threshold; mid-dark bgs may want manual override via `footer_tint` in the spec.
- Chart note column is a fixed right rail; very long note bodies should be pre-trimmed to 4 sentences.

**Decision:** plugin considered mature. check_deck has zero findings on all three decks; layouts (chart / process / compare / cards / statrow / twocol / section / takeaway / agenda / title) all exercised.
