# Comic Font Library and Automatic Font Matching

**Status:** design approved in conversation; spec awaiting user review  
**Target branch:** `main`  
**Repository:** `tuantran00541-spec/manga-translator`

## 1. Goal

Add a curated comic-font library of approximately 60–80 redistributable fonts,
organized by use case, while preserving the existing renderer and saved editor
data. A user may choose an exact font, an AI provider may return an exact font
ID, or the application may automatically retrieve the closest available font
from the original lettering when neither explicitly chooses one.

The automatic mode retrieves an existing bundled font. It does not generate a
new font, modify a font, or claim to reproduce a proprietary font exactly.

## 2. Existing constraints found on `main`

- `app/config.py` points `DEFAULT_FONT` to `app/static/fonts/default.ttf`.
- `app/render/text_renderer.py` currently resolves a flat font directory and
  `list_available_fonts()` only scans top-level `*.ttf` files.
- `/api/fonts` is served by `app/routers/render.py` and currently exposes only
  `{id, name}`.
- The editor creates a flat `<select>` from the `/api/fonts` response.
- Saved manifests already contain font names/IDs, so changing IDs without
  aliases would make existing drafts render with the wrong font.
- Rendering uses Pillow and already has a cached `ImageFont` loader, which can
  be reused by matching and rendering.

## 3. User-visible behavior

### 3.1 Explicit selection

The editor exposes a grouped font picker. Users can browse categories and pick
an exact stable `font_id`. The picker may show tags and Vietnamese coverage,
but it must remain usable as a normal font selector.

### 3.2 AI selection

Translation/vision providers may return an exact `font_id` in the style
payload. The backend validates the ID against the catalog before accepting it.
An invalid or unavailable ID is rejected or downgraded to automatic matching
according to the existing translation error policy; it must never become an
arbitrary filesystem path.

The provider-facing contract should support both:

```json
{
  "font_id": "emphasis/bangers",
  "font_category": "emphasis"
}
```

`font_id` has precedence. `font_category` is a semantic hint for automatic
matching and a fallback when no exact ID is returned.

### 3.3 Automatic matching

When the style requests automatic matching, the service uses the raw-lettering
crop retained before inpaint, OCR text when available, and the text region
geometry. It searches the full catalog, with category and language coverage as
priors, and returns a ranked list containing:

```json
{
  "font_id": "dialogue/be-vietnam-pro-semibold",
  "score": 0.87,
  "confidence": "high",
  "evidence": ["weight", "condensed-width", "glyph-shape"]
}
```

The renderer applies the best candidate only when confidence clears a
configured threshold. Otherwise it falls back to the category default or the
current default font and records that the result was uncertain.

## 4. Font storage and catalog

Fonts are stored once in a primary category directory:

```text
app/static/fonts/
├── dialogue/
├── emphasis/
├── thought/
├── narration/
├── skill/
├── sfx/
├── horror/
├── romance/
├── licenses/
├── font_catalog.json
└── default.ttf
```

Initial target allocation is approximately:

| Category | Target |
| --- | ---: |
| dialogue | 14 |
| emphasis | 9 |
| thought | 7 |
| narration | 9 |
| skill | 10 |
| sfx | 10 |
| horror | 5 |
| romance | 6 |
| **Total** | **70** |

Category is the primary folder only. A font may have multiple tags in the
catalog without being copied into multiple folders.

Each catalog entry must contain at least:

```json
{
  "id": "dialogue/be-vietnam-pro-semibold",
  "name": "Be Vietnam Pro SemiBold",
  "path": "dialogue/BeVietnamPro-SemiBold.ttf",
  "category": "dialogue",
  "tags": ["clean", "sans", "readable"],
  "vietnamese": true,
  "license": "OFL-1.1",
  "license_file": "licenses/be-vietnam-pro/OFL.txt",
  "source_url": "https://fonts.google.com/specimen/Be+Vietnam+Pro",
  "default_rank": 10
}
```

The catalog is the authority for resolving font IDs. Directory walking is only
used by a validation/import script, not by request-time path resolution.

## 5. Backward compatibility and security

- Keep `default.ttf` as the stable default path.
- Add an alias table for all existing flat IDs (`Mac-dinh-2`, `SFX-fonts-1`,
  and so on). Existing manifests must continue to render after migration.
- Only catalog paths under `app/static/fonts` may be loaded.
- Reject absolute paths, `..` traversal, symlinks escaping the font root, and
  unregistered files from API input.
- Font files must be validated as readable TrueType/OpenType files before they
  enter the catalog.

## 6. Matching algorithm

The first implementation is deterministic and CPU-friendly:

1. Extract the raw text crop and normalize it to a bounded grayscale mask.
2. Use OCR text and script coverage to reject candidates that cannot render the
   required characters.
3. Estimate inexpensive style features: ink density, stroke-width proxy,
   x-height/cap-height ratio, width/height ratio, slant, connected-component
   profile, rounded versus angular edge ratio, and outline/shadow presence.
4. Render the OCR string with each remaining candidate at normalized height,
   trying a small bounded set of size and horizontal-scale values.
5. Align the rendered mask to the source mask and score mask overlap, distance
   transform/chamfer error, edge agreement, width, and baseline geometry.
6. Combine the visual score with category/tag priors and return top-k results.

For noisy, rotated, decorative, or OCR-poor SFX, use the feature score and
geometry first. A future FontCLIP/DeepFont-style embedding backend may be
added behind the same interface, but it is not required for the first release
and must not add a heavy runtime dependency to the default CPU path.

The matcher must cache font descriptors and rendered candidate samples. It must
also expose timings so a chapter run can prove that matching is not becoming a
new bottleneck.

## 7. AI and API contract

Add a catalog response that includes stable IDs, category, tags, license,
Vietnamese support, and source metadata. Add a matching endpoint/service that
accepts a bounded crop reference or server-side text-object ID, never an
unvalidated filesystem path.

The translation pipeline should preserve the distinction between:

- explicit user style,
- explicit AI style,
- automatic style recommendation,
- unresolved/uncertain fallback.

The final manifest/render metadata should record the resolved `font_id` and
selection mode so a rerender is deterministic.

## 8. Font sourcing policy

The initial bundle uses fonts whose individual licenses permit redistribution
and embedding, preferably SIL Open Font License or an equivalent permissive
license. Sources to search first are Google Fonts, Font Library, and The League
of Moveable Type. Font Squirrel may be used only after checking each font's
individual license. Do not bulk-import fonts marked personal-use-only or with
unclear provenance.

Every bundled family gets its license/readme file and source URL. A script must
verify that every catalog entry points to an existing font and license file.

## 9. Testing and acceptance

Add tests for:

- recursive catalog loading and deterministic ordering;
- legacy flat-ID aliases;
- path traversal and unregistered-ID rejection;
- grouped `/api/fonts` response shape;
- Vietnamese glyph coverage using representative uppercase/lowercase strings;
- deterministic top-k matcher output on synthetic samples;
- low-confidence fallback behavior;
- explicit user choice taking precedence over AI and auto modes;
- AI `font_id` validation;
- render cache reuse and bounded matcher timing;
- existing render and manifest tests remaining green.

Acceptance requires a small specimen sheet covering every category, plus a
chapter-level smoke test proving that old manifests still render and that the
new catalog does not alter image pixels when the existing default font is
selected.

## 10. Deliberate non-goals

- No automatic generation of a brand-new font in the production path.
- No unlicensed “look-alike” font scraping or proprietary font redistribution.
- No mandatory CLIP/DeepFont/PyTorch dependency for ordinary rendering.
- No duplication of the same font file across category directories.
- No change to inpaint authority, OCR detection, or image-cleaning behavior.

