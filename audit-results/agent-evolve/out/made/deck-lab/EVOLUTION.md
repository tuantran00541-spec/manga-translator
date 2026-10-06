# pptx plugin evolution log

Plugin: `pptxdeck` (`.agents/plugins/pptxdeck_v*.py` → mounted at `~/.manga-agent/plugins/pptxdeck.py`).
Tools: `make_deck` (JSON spec → .pptx) and `check_deck_tool` (QC via `deck-lab/check_deck.py`).
Rule: after each round, find layout / color / type / effect weaknesses, rewrite the plugin, rebuild all 3 decks, re-QC. Only stop when `check_deck_tool` is CLEAN on every deck.

## Round 1 — v1 (baseline)

Spec: 8 layouts (title, bullets, cards, two_panel, chart, numbered_rows, compare, process), dark+light palettes, native clustered-column charts, full-bleed-rect backgrounds, DejaVu Sans, 16:9.

Decks: `pv1/chart_mkt.pptx` (dark, chart+compare), `pv1/process_onboarding.pptx` (light, 5-step process), `pv1/compare_solo.pptx` (dark, 4-step process).

QC: all 3 CLEAN. Visual inspection found:
- **B1 — bullets**: 3+ items in one full card stack vertically, last item collides with the card bottom; no two-column fallback.
- **B2 — compare**: no takeaway line; two columns read as a feature list, not a decision.
- **B3 — process**: the `>` arrow between step cards sits off the card midline and is tiny.
- **B4 — transitions**: absent entirely; spec had no field for them.
- **B5 — numbered_rows**: pitch formula `min(1.25, 4.6/n)` evaluates as `min(1.25, 4.6) / n` for n>4 → overflow on 5+ rows (caught by QC on a 5-row test before shipping).
- **B6 — deck-size guard**: no warning when a spec has fewer than 3 content slides.

## Round 2 — v2

Fixes: B1 (bullets auto-split into two 5.9in cards when n≥3), B2 (`takeaway_a` / `takeaway_b` under each compare card), B3 (arrow 22pt → then 18pt on its own 0.5in box, centered at card mid-height y=3.5), B5 (pitch = `4.6/n` for n>4, `min(1.25, 4.6/n)` for n≤4).

Decks rebuilt in `pv2/`. QC on first rebuild:
- **NEW — process arrows**: 22pt `>` in a 0.5in box = est. 0.37in text vs 0.34in usable → overflow flag on every arrow. Fixed to 18pt.
- All three pv2 decks CLEAN; bullet two-column and compare takeaways verified in rendered PNGs.

## Round 3 — v3

Additions: `transition` spec field (`push|fade|wipe|blinds|dissolve|none`) injected as `<p:transition>` after `p:cSld` (schema order matters, else LibreOffice drops it); deck-size <3 warning; card `takeaway` support; numbered_rows desc 14pt.

**Critical defect found**: `make_deck` returned "saved …" but the .pptx was not on disk. Traced round and round: layouts ran, `Presentation.save` was never reached because **`prs.save(path)` was missing from the function body** — the early draft assumed the file write happened. Every prior "saved" string was a lie. Fixed by adding `prs.save(path)` before the size check; re-verified with a direct `os.path.exists` check after each save (never trust a return string, trust the filesystem).

Decks in `pv3/` (chart_churn light+fade, process_bug dark+wipe, compare_sync dark+blinds): all three CLEAN, transitions present in saved XML.

## Round 4 — v4 + checker hardening

- Compare takeaway moved inside the card bottom (y=5.9) instead of hanging below it.
- Full-bleed bg rect (0,0,full size) excluded from the new left-margin check in `check_deck.py` (it previously false-positived on the background rectangle).
- `check_deck.py` gained: words-per-slide cap (>220 warns), left-margin <0.3in check for text shapes.
- All pv3 decks re-QCed against the new checker: CLEAN.

## Stopping condition

Last full pass: `chart_churn`, `process_bug`, `compare_sync` (+ all pv1/pv2 rebuilds) → `check_deck_tool` CLEAN on every deck, 6 slides or fewer dense each, transitions in XML, no overflow/overlap/contrast/margin/word-count flags. Plugin is mature for this repo's 16:9 dark/light decks.

## Recurring lessons (feed back into SKILL.md)

1. Never claim "saved" — verify the file exists on disk after `prs.save`.
2. Arrows/glyphs in small boxes are the top overflow cause: budget `pt*1.2` for even a one-character run.
3. Row-pitch formulas: write `4.6/max(n,1)` explicitly; `min(a, b/n)` parses as `min(a,b)/n`.
4. `<p:transition>` goes inside the slide element after `cSld`; schema order is enforced by LibreOffice.
5. Checker false positives come from treating the bg rectangle as a text shape — exempt by geometry, not by name.
