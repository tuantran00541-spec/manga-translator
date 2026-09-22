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

- The CPU matcher runs automatically for every render that has an original
  lettering crop. It compares that crop with rendered samples from the catalog
  and records the ranked candidates in `font_match`.
- `font_selection_mode=user`: a valid editor/API choice is kept only when it
  appears in the matcher candidates; otherwise the best visual match is used.
- `font_selection_mode=ai`: a validated AI `font_id` follows the same safety
  check, so an unsuitable AI choice falls back to the best visual match.
- `font_selection_mode=auto` (also used for newly detected text objects): the
  best visual match is applied. A legacy object with an unmarked `default`
  style remains on the historical default for compatibility, while its match
  candidates are still recorded.
- If the crop is unavailable or confidence is low, rendering falls back to
  `default` without generating or downloading a new font.

The matcher is deterministic and CPU-only. It uses grayscale ink geometry,
projection profiles, edge profiles, and text metrics; it does not use a model
that can invent font files. The API surface is:

- `GET /api/fonts` — grouped catalog metadata
- `POST /api/fonts/match` — ranked candidates for API/AI callers that want to
  inspect or reuse the same matcher outside the render pipeline

All bundled families are sourced from Google Fonts and redistributed under
their SIL Open Font License metadata in `app/static/fonts/licenses/`.
