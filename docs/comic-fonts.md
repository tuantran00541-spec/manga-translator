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

- `font_selection_mode=user`: an explicit editor/API choice wins.
- `font_selection_mode=ai`: a validated AI `font_id` is used when present.
- `font_selection_mode=auto` or a legacy `default`: the CPU matcher compares
  the original lettering crop with rendered samples from the catalog.
- If the crop is unavailable or confidence is low, rendering falls back to
  `default` without generating or downloading a new font.

The matcher is deterministic and CPU-only. It uses grayscale ink geometry,
projection profiles, edge profiles, and text metrics; it does not use a model
that can invent font files. The API surface is:

- `GET /api/fonts` — grouped catalog metadata
- `POST /api/fonts/match` — ranked suggestions for one page text object

All bundled families are sourced from Google Fonts and redistributed under
their SIL Open Font License metadata in `app/static/fonts/licenses/`.
