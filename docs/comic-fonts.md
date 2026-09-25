# Comic font library

The native renderer ships a catalog of 70 fonts under `app/static/fonts/`.
Each family is grouped by intended comic role:

- `dialogue/` — readable speech balloons and Vietnamese-friendly sans faces
- `emphasis/` — bold impact lettering
- `thought/` — casual handwritten narration
- `narration/` — serif caption styles
- `skill/` — fantasy, sci-fi, and game-like display faces
- `sfx/` — sound effects and display lettering
- `horror/` — distressed and horror lettering
- `romance/` — script and rounded display lettering

`font_catalog.json` is the source of truth. Clients should send a stable
catalog `font_id`, not a filesystem path. Legacy flat font IDs and `default`
remain supported by the resolver.

## Selection modes

Precedence, highest first:

- `font_selection_mode=user`: a valid editor/API choice is always kept. An
  unknown ID falls back to `default`.
- `font_selection_mode=ai`: a validated AI `font_id` (the translator may pick a
  catalog font by role: dialogue, SFX, horror, ...). An unknown ID falls back
  to `default`.
- `font_selection_mode=auto` (also used for newly detected text objects): use
  the AI choice when there is one, otherwise `default`.
- A legacy object with an unmarked non-default style keeps that font.

There is no visual font matcher. An earlier CPU matcher compared the original
lettering crop with catalog samples, but it could not work for Chinese,
Japanese or Korean sources (no bundled font has those glyphs, so every sample
rendered blank) and misranked Latin samples too, so it was removed.

The API surface is `GET /api/fonts` (grouped catalog metadata).

All bundled families are sourced from Google Fonts and redistributed under
their SIL Open Font License metadata in `app/static/fonts/licenses/`.
