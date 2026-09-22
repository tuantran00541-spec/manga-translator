# Comic Font Library and Automatic Font Matching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a 60–80-font comic library with stable catalog IDs, grouped user/AI selection, and a CPU-first matcher that retrieves the closest bundled font from original lettering.

**Architecture:** Store each font once under a primary category and make `font_catalog.json` the only request-time authority for IDs and paths. Keep existing flat IDs as aliases, expose catalog and matching services through the existing render router, and resolve `font="auto"` during render from the raw page crop; explicit user or AI font IDs always override matching.

**Tech Stack:** Python 3, FastAPI, Pydantic 2, Pillow, NumPy/OpenCV already used by the repository, pytest, vanilla JavaScript editor, Google Fonts/Font Library metadata and SIL Open Font License files.

**Spec:** `docs/superpowers/specs/2026-09-22-font-library-and-auto-match-design.md`

## Global Constraints

- Work directly on `main`; preserve fast-forward history and never force-push.
- Bundle approximately 70 fonts in eight primary categories: dialogue, emphasis, thought, narration, skill, sfx, horror, romance.
- Prefer redistributable open licenses; every family must ship its license/readme and source URL.
- The catalog, not a request-provided path, is the authority for font resolution.
- Preserve aliases for every existing flat font ID and keep `default.ttf` stable.
- `font_id` or a user-selected exact font has precedence over automatic matching; `font="auto"` invokes matching.
- Do not add a mandatory PyTorch, CLIP, DeepFont, or diffusion dependency to the normal render path.
- Do not change detector, OCR, inpaint authority, slice ownership, or cleanup behavior.
- A low-confidence match must fall back deterministically and record the uncertainty.
- No bundled font may be personal-use-only or have unclear provenance.

## Review Focus

- A legacy manifest containing `SFX-fonts-1` must still render with the same file; test in Task 2.
- A malicious font ID such as `../../models/bubble_yolo.onnx` must never escape the font root; test in Task 2.
- A Vietnamese string containing uppercase and lowercase diacritics must not silently fall back to a missing-glyph font; test in Task 1 and Task 3.
- A noisy/empty OCR crop must not make the matcher choose a high-confidence arbitrary font; test in Task 3.
- When a user-selected font, AI-selected font, and auto mode conflict, the explicit user choice must win and the final manifest must record the resolved ID; test in Task 5 and Task 6.

### Task 1: Source, validate, and package the font library

**Files:**
- Create: `scripts/font_sources.json`
- Create: `scripts/build_font_catalog.py`
- Create: `app/static/fonts/font_catalog.json`
- Create: `app/static/fonts/dialogue/`, `emphasis/`, `thought/`, `narration/`, `skill/`, `sfx/`, `horror/`, `romance/`, and `licenses/`
- Modify: `requirements-test.txt` (add `fonttools==4.53.1`, used only by tests/scripts)
- Test: `tests/test_font_catalog_assets.py`

**Interfaces:**
- `scripts/font_sources.json` is a checked-in source manifest containing one entry per family: `id`, `name`, `category`, `google_family` or direct source URL, expected license, source URL, tags, and `default_rank`.
- `scripts/build_font_catalog.py --verify` exits non-zero for missing files, invalid font tables, missing license files, duplicate IDs, duplicate paths, unsupported categories, or a catalog count outside 60–80.
- `app/static/fonts/font_catalog.json` contains `fonts: list[dict]` and `aliases: dict[str, str]`; every `path` is relative to the font root.

- [ ] **Step 1: Write the failing asset and source-manifest tests.**

```python
def test_catalog_has_expected_shape_and_categories():
    catalog = load_checked_in_catalog()
    assert 60 <= len(catalog["fonts"]) <= 80
    assert {item["category"] for item in catalog["fonts"]} == {
        "dialogue", "emphasis", "thought", "narration",
        "skill", "sfx", "horror", "romance",
    }

def test_every_font_has_file_license_and_source():
    for item in load_checked_in_catalog()["fonts"]:
        assert (FONT_ROOT / item["path"]).is_file()
        assert (FONT_ROOT / item["license_file"]).is_file()
        assert item["source_url"].startswith(("https://fonts.google.com/", "https://fontlibrary.org/"))

def test_vietnamese_coverage_is_explicit():
    for item in load_checked_in_catalog()["fonts"]:
        assert isinstance(item["vietnamese"], bool)
    assert any(item["vietnamese"] for item in load_checked_in_catalog()["fonts"])
```

- [ ] **Step 2: Run the focused tests and verify they fail because the catalog/assets do not exist.**

Run: `pytest tests/test_font_catalog_assets.py -q`

Expected: FAIL with missing catalog/font-root errors.

- [ ] **Step 3: Populate the source manifest with the concrete initial families.**

Use these 70 Google Fonts families as the first curated set, assigning each family exactly one primary category while allowing cross-category tags:

```text
dialogue: Be Vietnam Pro, Noto Sans, Roboto, Roboto Condensed, Open Sans,
  Inter, Nunito Sans, Source Sans 3, Public Sans, Lato, Atkinson Hyperlegible,
  Lexend, IBM Plex Sans, Assistant
emphasis: Bangers, Anton, Bebas Neue, Oswald, Teko, League Spartan,
  Archivo Black, Bungee, Russo One
thought: Comic Neue, Patrick Hand, Kalam, Caveat, Itim, Mali, Indie Flower
narration: Merriweather, Lora, Libre Baskerville, Source Serif 4, Spectral,
  Alegreya, Vollkorn, Crimson Pro, Literata
skill: Phudu, Cinzel, Orbitron, Exo 2, Audiowide, Chakra Petch, Rajdhani,
  Michroma, Uncial Antiqua, Pirata One
sfx: Knewave, Luckiest Guy, Bowlby One SC, Titan One, Concert One, Fredoka,
  Modak, Rubik Mono One, Black Ops One, Special Elite
horror: Nosifer, Creepster, Eater, Metal Mania, Dokdo
romance: Dancing Script, Pacifico, Great Vibes, Comfortaa, Satisfy, Lobster
```

The manifest must mark Vietnamese coverage from actual cmap inspection; it must not assume that every Google Font supports Vietnamese. Download the font file and its OFL text from the pinned Google Fonts repository family directory, and preserve the upstream family name in the license/readme.

- [ ] **Step 4: Implement the builder/validator and generate the checked-in catalog.**

`build_font_catalog.py` must use `fontTools.ttLib.TTFont` to validate `name`, `cmap`, and outline tables, compute required Vietnamese coverage for `Ắ Ằ Ẳ Ẵ Ặ Â Ê Ô Ơ Ư Đ ă ằ ẳ ẵ ặ â ê ô ơ ư đ`, and write stable POSIX-relative paths. It must never overwrite a font or license file unless `--write` is explicitly supplied.

- [ ] **Step 5: Run the asset tests and validator.**

Run: `python scripts/build_font_catalog.py --verify && pytest tests/test_font_catalog_assets.py -q`

Expected: the validator exits 0, the count is 70, every path/license exists, and the focused tests pass.

- [ ] **Step 6: Commit the asset package.**

```bash
git add scripts/font_sources.json scripts/build_font_catalog.py app/static/fonts requirements-test.txt tests/test_font_catalog_assets.py
git commit -m "feat: add licensed comic font library"
```

### Task 2: Build the catalog service and legacy-safe resolver

**Files:**
- Create: `app/render/font_catalog.py`
- Modify: `app/config.py`
- Modify: `app/render/text_renderer.py`
- Test: `tests/test_font_catalog_service.py`
- Test: `tests/test_font_resolution_security.py`

**Interfaces:**
- `FontRecord` is a frozen dataclass with `id: str`, `name: str`, `path: Path`, `category: str`, `tags: tuple[str, ...]`, `vietnamese: bool`, `license: str`, `license_file: str`, `source_url: str`, and `default_rank: int`.
- `load_font_catalog() -> FontCatalog` loads and caches the JSON catalog after validating every record.
- `list_font_records() -> list[dict]` returns stable category/rank/name order for `/api/fonts`.
- `resolve_font_id(font_id: str) -> Path` accepts a catalog ID or legacy alias and returns a path below `FONT_ROOT`; unknown, absolute, traversal, symlink-escaping, and unregistered IDs raise `ValueError`.
- `resolve_font_name` in `text_renderer.py` delegates to `resolve_font_id`, preserving the existing `get_font_path(font_name="default")` call shape.

- [ ] **Step 1: Write resolver tests before implementation.**

```python
def test_legacy_flat_ids_resolve_to_catalog_paths():
    assert resolve_font_id("default").name == "default.ttf"
    assert resolve_font_id("SFX-fonts-1").is_file()

def test_catalog_ids_resolve_only_inside_font_root():
    path = resolve_font_id("emphasis/bangers")
    assert path.is_relative_to(FONT_ROOT)

@pytest.mark.parametrize("value", ["../../models/bubble_yolo.onnx", "/tmp/x.ttf", "unknown/font"])
def test_untrusted_font_id_is_rejected(value):
    with pytest.raises(ValueError):
        resolve_font_id(value)
```

- [ ] **Step 2: Run the resolver tests and verify they fail.**

Run: `pytest tests/test_font_catalog_service.py tests/test_font_resolution_security.py -q`

Expected: FAIL because `FontCatalog` and catalog-aware resolution are not defined.

- [ ] **Step 3: Implement catalog loading and path validation.**

Set `FONT_ROOT = BASE_DIR / "app" / "static" / "fonts"` and `FONT_CATALOG_PATH = FONT_ROOT / "font_catalog.json"` in `app/config.py`. Validate IDs with a strict `^[a-z0-9][a-z0-9/_-]{0,127}$` shape, resolve paths, require `path.is_file()`, and verify `resolved.is_relative_to(FONT_ROOT.resolve())`.

- [ ] **Step 4: Replace flat scanning in `get_font_path` and `list_available_fonts`.**

Keep the Windows system-font fallback only for the existing non-catalog compatibility behavior; catalog IDs must resolve first and must never accept a Windows path from a request. Return records with `id`, `name`, `category`, `tags`, `vietnamese`, `license`, and `source_url` while retaining the default record.

- [ ] **Step 5: Run service/security and existing renderer tests.**

Run: `pytest tests/test_font_catalog_service.py tests/test_font_resolution_security.py tests/test_ocr_render_metadata.py -q`

Expected: all focused tests pass and existing renderer metadata tests remain green.

- [ ] **Step 6: Commit the resolver.**

```bash
git add app/config.py app/render/font_catalog.py app/render/text_renderer.py tests/test_font_catalog_service.py tests/test_font_resolution_security.py
git commit -m "feat: add catalog-backed font resolver"
```

### Task 3: Implement the deterministic CPU font matcher

**Files:**
- Create: `app/render/font_matcher.py`
- Create: `scripts/font_match_benchmark.py`
- Test: `tests/test_font_matcher.py`
- Test: `tests/test_font_matcher_regressions.py`

**Interfaces:**
- `FontMatch` is a frozen dataclass with `font_id: str`, `score: float`, `confidence: Literal["high", "medium", "low"]`, and `evidence: tuple[str, ...]`.
- `match_fonts(image: Image.Image, region: tuple[int, int, int, int], source_text: str, *, category: str | None = None, top_k: int = 3) -> list[FontMatch]` ranks the catalog without mutating the image.
- `clear_match_caches() -> None` clears descriptor/render caches for tests and memory diagnostics.

- [ ] **Step 1: Write synthetic matching tests.**

```python
def test_matching_same_font_ranks_first():
    source = render_reference_crop("Hello Việt", "dialogue/be-vietnam-pro-semibold")
    result = match_fonts(source, (0, 0, *source.size), "Hello Việt", top_k=3)
    assert result[0].font_id == "dialogue/be-vietnam-pro-semibold"
    assert result[0].score > result[1].score

def test_category_is_a_prior_not_a_hard_filter():
    source = render_reference_crop("BOOM", "sfx/knewave")
    result = match_fonts(source, (0, 0, *source.size), "BOOM", category="dialogue", top_k=70)
    assert any(item.font_id == "sfx/knewave" for item in result)

def test_empty_or_unusable_crop_is_low_confidence():
    result = match_fonts(Image.new("RGB", (80, 30), "white"), (0, 0, 80, 30), "", top_k=3)
    assert result and all(item.confidence == "low" for item in result)
```

- [ ] **Step 2: Run matcher tests and verify they fail.**

Run: `pytest tests/test_font_matcher.py tests/test_font_matcher_regressions.py -q`

Expected: FAIL because `match_fonts` is not defined.

- [ ] **Step 3: Implement bounded crop normalization and script filtering.**

Clamp the region to image bounds; convert to grayscale; threshold with a stable foreground/background rule; reject a crop with no foreground pixels. Use the catalog's `vietnamese` flag and Pillow glyph measurements to exclude candidates that cannot render the source text.

- [ ] **Step 4: Implement cached feature extraction.**

Cache per-font measurements keyed by `(font_id, normalized_height)`. Compute ink density, connected-component count/profile, bounding width/height, approximate stroke width from distance transform, slant from centroid-vs-row drift, rounded/angular edge ratio, and baseline occupancy. Keep all arrays bounded to a maximum 256×128 working crop.

- [ ] **Step 5: Implement same-text candidate rendering and scoring.**

Render source text with each remaining font at normalized height and horizontal scales `(0.90, 1.00, 1.10)`, align by foreground bounding boxes, then combine IoU, distance-transform/chamfer error, edge overlap, width ratio, and feature distance. Apply a small category prior only after visual score. Return deterministic descending score with `font_id` as the final tie-breaker.

- [ ] **Step 6: Implement confidence thresholds and cache clearing.**

Use `high` only when the top score clears the high threshold and beats the runner-up by the configured margin; use `medium` for a useful but ambiguous result; mark empty/noisy results `low`. Keep thresholds as constants in `app/render/font_matcher.py` until benchmark evidence justifies centralizing them in `app/parameters.py`.

- [ ] **Step 7: Add the timing probe and run focused matcher tests.**

Implement `scripts/font_match_benchmark.py --samples N` using the same `match_fonts` interface and checked-in specimen strings. Run: `pytest tests/test_font_matcher.py tests/test_font_matcher_regressions.py -q` and `python scripts/font_match_benchmark.py --samples 20`.

Expected: deterministic top-k tests pass, low-confidence fallback is exercised, and the benchmark reports cold/warm timings without loading an additional neural model.

- [ ] **Step 8: Commit the matcher and benchmark.**

```bash
git add app/render/font_matcher.py scripts/font_match_benchmark.py tests/test_font_matcher.py tests/test_font_matcher_regressions.py
git commit -m "feat: add CPU comic font matcher"
```

### Task 4: Expose catalog and matching through the render API

**Files:**
- Modify: `app/schemas.py`
- Modify: `app/routers/render.py`
- Test: `tests/test_font_api.py`

**Interfaces:**
- Add `FontMatchRequest(BaseModel)` with `chapter_id: str`, `page_index: int >= 0`, `object_id: str`, `source_text: str | None = None`, and `top_k: int = Field(default=3, ge=1, le=5)`.
- `GET /api/fonts` returns the catalog records in deterministic order.
- `POST /api/fonts/match` returns `{object_id, matches, resolved_font_id, selection_mode}` and reads the managed original page path from the manifest; callers never submit an image filesystem path.

- [ ] **Step 1: Write API tests for response shape and managed-path behavior.**

```python
def test_fonts_endpoint_returns_group_metadata(client):
    response = client.get("/api/fonts")
    assert response.status_code == 200
    assert {item["category"] for item in response.json()} >= {"dialogue", "sfx"}
    assert {"id", "name", "category", "vietnamese"} <= response.json()[0].keys()

def test_match_endpoint_uses_manifest_object_and_rejects_unknown_object(client, chapter_fixture):
    response = client.post("/api/fonts/match", json={
        "chapter_id": chapter_fixture.id, "page_index": 0,
        "object_id": "missing", "top_k": 3,
    })
    assert response.status_code == 404
```

- [ ] **Step 2: Run the API tests and verify they fail.**

Run: `pytest tests/test_font_api.py -q`

Expected: FAIL because catalog metadata and `/api/fonts/match` do not exist.

- [ ] **Step 3: Implement catalog response and bounded match request.**

In `app/routers/render.py`, validate `chapter_id`, load the manifest, locate the exact page/object, validate the managed original path under `RAW_DIR / chapter_id`, load the image, choose `source_text` from the request only when non-empty otherwise the object's `ocr_text`, and call `match_fonts` in the existing threadpool pattern if the route is async.

- [ ] **Step 4: Implement deterministic fallback in the API.**

Set `resolved_font_id` to the top match only for `high` or `medium` confidence; for `low`, return the category default or `default` and set `selection_mode="auto-fallback"`. Never return a path to the client.

- [ ] **Step 5: Run API, security, and router tests.**

Run: `pytest tests/test_font_api.py tests/test_font_resolution_security.py tests/test_processing_pipeline_factory.py -q`

Expected: all pass, including missing object, missing page image, and invalid chapter cases.

- [ ] **Step 6: Commit the API.**

```bash
git add app/schemas.py app/routers/render.py tests/test_font_api.py
git commit -m "feat: expose font catalog and matching API"
```

### Task 5: Integrate explicit user/AI/auto selection into render and translation

**Files:**
- Modify: `app/text_objects.py`
- Modify: `app/schemas.py`
- Modify: `app/render/page_renderer.py`
- Modify: `app/routers/render_commit.py`
- Modify: `app/translation/deepseek.py`
- Modify: `app/translation/vision.py`
- Modify: `app/routers/translation.py`
- Test: `tests/test_font_selection_precedence.py`
- Test: `tests/test_vision_translation.py`
- Test: `tests/test_ai_provider_configuration.py`

**Interfaces:**
- Treat `style.font` values `default`, a catalog `font_id`, and `auto` as the only supported choices; keep legacy flat aliases valid. A legacy/default style is not considered an explicit user choice until `font_selection_mode="user"` is recorded.
- Add `font_id: str | None` and `font_mode: Literal["explicit", "auto"] | None` as optional fields in AI translation entries. Unknown IDs are discarded with a recorded validation reason, never passed to Pillow.
- Extend translation result objects with `styles: dict[str, dict[str, str]]` while preserving the existing `translations` mapping and constructor behavior for callers that do not request styles.
- Add `resolve_object_font(obj, source_image, region, source_text) -> tuple[str, str, dict]`, returning `(resolved_font_id, selection_mode, match_metadata)`; an explicit user choice (`font_selection_mode="user"` plus a catalog/legacy ID) wins, then a validated AI `font_id` when the current style is `default`/`auto`, then the `auto` matcher, then `default`.

- [ ] **Step 1: Write precedence and parser tests.**

```python
def test_user_font_wins_over_ai_and_auto():
    obj = {"style": {"font": "dialogue/roboto"}, "font_selection_mode": "user", "font_ai_id": "sfx/knewave"}
    resolved = resolve_object_font(obj, source_image, (0, 0, 80, 30), "Hello")
    assert resolved[0] == "dialogue/roboto"
    assert resolved[1] == "user"

def test_ai_font_wins_when_user_style_is_auto(monkeypatch):
    obj = {"style": {"font": "auto"}, "font_ai_id": "emphasis/bangers"}
    resolved = resolve_object_font(obj, source_image, (0, 0, 80, 30), "BOOM")
    assert resolved[:2] == ("emphasis/bangers", "ai")

def test_invalid_ai_font_degrades_to_auto(monkeypatch):
    obj = {"style": {"font": "auto"}, "font_ai_id": "../../bad.ttf"}
    monkeypatch.setattr("app.render.page_renderer.match_fonts", fake_low_confidence_match)
    resolved = resolve_object_font(obj, source_image, (0, 0, 80, 30), "text")
    assert resolved[1] == "auto-fallback"
```

- [ ] **Step 2: Run selection and vision tests and verify they fail.**

Run: `pytest tests/test_font_selection_precedence.py tests/test_vision_translation.py -q`

Expected: FAIL because style fields and precedence resolution are not implemented.

- [ ] **Step 3: Add auto/AI metadata to text-object state without breaking old styles.**

Keep `DEFAULT_TEXT_OBJECT_STYLE["font"] == "default"`; add object-level fields `font_selection_mode`, `font_match`, and `font_ai_id` only when a non-default selection is resolved. Existing `TextObjectStyle` validation must continue to accept legacy flat IDs through the catalog alias resolver.

- [ ] **Step 4: Implement AI payload parsing and validation.**

Update the OpenAI-compatible and vision JSON contracts to allow optional `font_id` and `font_mode` per known object ID. Preserve strict unknown/repeated ID checks for translations. Validate the font ID through `resolve_font_id` before storing it. A model may return `font_mode="auto"`, in which case it must not provide an arbitrary path.

- [ ] **Step 5: Resolve fonts once per render snapshot and persist metadata.**

Load the managed original page image in `render_commit.py` only when at least one active object has `style.font == "auto"` or a validated AI font ID. Resolve each object against its raw crop, pass the resolved ID into `render_text_objects`, and persist `font_selection_mode`, `font_match`, and resolved `style.font` in `_commit_render`. The input signature must include the raw image revision already used by the manifest so a changed page cannot commit a stale match.

- [ ] **Step 6: Keep explicit selections stable across rerenders.**

When the user sends an exact catalog ID through the existing render/style update path, skip matching and persist `selection_mode="user"`. When an AI ID is committed, persist `selection_mode="ai"`; when auto selects a candidate, persist the score/evidence and reuse that resolved ID until the source revision or text region changes.

- [ ] **Step 7: Run full focused integration tests.**

Run: `pytest tests/test_font_selection_precedence.py tests/test_vision_translation.py tests/test_ai_provider_configuration.py tests/test_ocr_render_metadata.py tests/test_repaint_original_context.py -q`

Expected: explicit user, AI, auto, invalid-AI, low-confidence, and legacy cases all pass without changing existing render metadata semantics.

- [ ] **Step 8: Commit the integration.**

```bash
git add app/text_objects.py app/schemas.py app/render/page_renderer.py app/routers/render_commit.py app/translation/deepseek.py app/translation/vision.py app/routers/translation.py tests/test_font_selection_precedence.py tests/test_vision_translation.py tests/test_ai_provider_configuration.py
git commit -m "feat: integrate explicit AI and automatic font selection"
```

### Task 6: Update the editor to show groups and auto-match controls

**Files:**
- Modify: `app/static/js/api.js`
- Modify: `app/static/js/editor-inspector.js`
- Create: `scripts/editor_font_picker_sanity.js`

**Interfaces:**
- `availableFonts` stores the catalog records returned by `GET /api/fonts`, not only `{id,name}`.
- The typography picker renders `<optgroup label="…">` from `category`, with a first option `Tự động tìm font gần giống` (`value="auto"`) and a stable default option.
- `loadFonts()` keeps its current fallback behavior when the API is unavailable.

- [ ] **Step 1: Write the browser sanity assertions.**

```javascript
const records = [
  { id: "auto", name: "Tự động tìm font gần giống", category: "Tự động" },
  { id: "dialogue/roboto", name: "Roboto", category: "Thoại" },
  { id: "sfx/knewave", name: "Knewave", category: "SFX" },
];
const select = buildFontSelect(records, "auto");
console.assert(select.querySelector('option[value="auto"]'));
console.assert(select.querySelector('optgroup[label="Thoại"] option[value="dialogue/roboto"]'));
console.assert(select.value === "auto");
```

- [ ] **Step 2: Run the sanity script and verify it fails because grouping is absent.**

Run: `node scripts/editor_font_picker_sanity.js`

Expected: FAIL on missing `buildFontSelect`/grouped option assertions.

- [ ] **Step 3: Update `loadFonts()` and the inspector picker.**

Keep the raw API records in `availableFonts`, map category IDs to Vietnamese labels in one constant, insert `auto` and `default`, group records by category, and preserve a saved exact ID even if it is an alias. Do not use `innerHTML` with font names or source metadata; create option/optgroup nodes and assign `textContent`.

- [ ] **Step 4: Preserve style during panel refresh and bulk render.**

Ensure `collectPanelState`, `_applyStateDiff`, `collectRenderPayload`, and pending persistence treat `font="auto"` and catalog IDs as ordinary style values. A picker change must persist immediately through the existing text-object update path and invalidate the page render.

- [ ] **Step 5: Run UI sanity and existing frontend contracts.**

Run: `node scripts/editor_font_picker_sanity.js && node scripts/theme_runtime_sanity.js && pytest tests/test_theme_picker_contract.py -q`

Expected: grouped picker, auto option, fallback, and existing theme contracts pass.

- [ ] **Step 6: Commit the editor update.**

```bash
git add app/static/js/api.js app/static/js/editor-inspector.js scripts/editor_font_picker_sanity.js
git commit -m "feat: add grouped comic font picker"
```

### Task 7: Add end-to-end validation, specimen output, and documentation

**Files:**
- Create: `scripts/font_specimen.py`
- Create: `tests/test_font_end_to_end.py`
- Modify: `README.md`
- Modify: `docs/UI_GUIDELINES.md` only where the typography picker contract is documented

**Interfaces:**
- `python scripts/font_match_benchmark.py --samples N` reports cold/warm matching time, cache hit count, and top-1/top-3 synthetic retrieval accuracy.
- `python scripts/font_specimen.py --output <path>` creates one PNG specimen sheet containing all categories and the Vietnamese coverage sample.
- The end-to-end test uses a temporary manifest/page and proves selection precedence, render commit, catalog metadata, and legacy default behavior.

- [ ] **Step 1: Write the end-to-end regression tests.**

```python
def test_default_render_pixels_are_unchanged_for_legacy_manifest(chapter_fixture):
    before = render_with_font(chapter_fixture, "default")
    after = render_with_font(chapter_fixture, "default")
    assert image_bytes(before) == image_bytes(after)

def test_auto_match_records_resolved_font_and_mode(chapter_fixture):
    result = render_with_font(chapter_fixture, "auto")
    obj = load_manifest(chapter_fixture.id)["pages"][0]["text_objects"][0]
    assert obj["font_selection_mode"] in {"auto", "auto-fallback"}
    assert isinstance(obj["style"]["font"], str)
    assert result["committed"] is True
```

- [ ] **Step 2: Implement benchmark/specimen tools and run the full suite.**

Run: `python scripts/font_specimen.py --output /tmp/font-specimen.png`, `python scripts/font_match_benchmark.py --samples 20`, and `pytest -q`.

Expected: specimen output exists, benchmark reports bounded timings, and all existing repository tests pass.

- [ ] **Step 3: Document user and AI selection.**

Update README with the picker modes, stable `font_id` example, `font="auto"` behavior, license policy, and the command used to verify catalog assets. Document that auto matching retrieves from the bundled catalog and can fall back when confidence is low.

- [ ] **Step 4: Run release-oriented gates.**

Run: `python scripts/build_font_catalog.py --verify && pytest -q && git diff --check`

Expected: catalog verification, all tests, and whitespace checks pass.

- [ ] **Step 5: Commit documentation and final validation.**

```bash
git add scripts/font_match_benchmark.py scripts/font_specimen.py tests/test_font_end_to_end.py README.md docs/UI_GUIDELINES.md
git commit -m "test: validate comic font library end to end"
```

## Self-review checklist

- Spec coverage: asset/license policy is Task 1; catalog/path security is Task 2; deterministic matching is Task 3; API is Task 4; explicit/AI/auto precedence and manifest persistence are Task 5; grouped user UI is Task 6; acceptance/specimen/release checks are Task 7.
- Placeholder scan: no `TODO`, `TBD`, `FIXME`, or unspecified “handle later” steps are used.
- Type consistency: `FontRecord`, `FontMatch`, `FontMatchRequest`, `match_fonts`, `resolve_font_id`, and `resolve_object_font` are defined before consumers and retain exact return shapes throughout the plan.
- Review focus coverage: every listed failure mode has an owning test in Tasks 1–6; Task 7 repeats the default-render and persisted-auto cases at chapter level.
- Scope: FontCLIP/DeepFont is intentionally an extension point, not a required implementation dependency; no unrelated OCR/inpaint refactor is included.
