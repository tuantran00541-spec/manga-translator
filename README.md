# Manga & Webtoon Translator Studio

Local-first, CPU-oriented translation and lettering workstation for manga, manhua, manhwa, and webtoon chapters.

Manga & Webtoon Translator Studio is built around a **human-in-the-loop editorial workflow**. Automation prepares a chapter, but the editor remains the source of truth for text, geometry, typography, cleanup decisions, and final export.

**Import → Slice → Detect → Clean → Review → OCR → Translate → Letter → Render → Export**

The project runs locally as a FastAPI application with a browser-based review workspace. Local ONNX models handle detection and inpainting; OCR and optional AI services are added around that local core rather than replacing it.

## What the project is

This is not a single “translate this image” script.

It is a chapter-oriented processing system with four persistent layers:

1. **Source artwork** — the original imported pages or downloaded chapter.
2. **Processing artifacts** — slices, detections, masks, cleaned pages, and OCR results.
3. **Editorial state** — text objects, translations, geometry, typography, skip/preserve decisions, and manual repairs.
4. **Published artifacts** — rendered pages and the final stitched ZIP export.

The architecture is deliberately revision-aware: expensive work is allowed to run concurrently, but stale results are rejected instead of silently replacing newer editor changes.

## End-to-end workflow

### 1. Import

Import a chapter from:

- PNG, JPEG, WEBP, or BMP files
- ZIP or CBZ archives
- A chapter URL

URL ingestion combines direct HTTP retrieval with Playwright-based discovery. The generic browser path understands relative image URLs, `srcset`, common lazy-load attributes, and pages whose images only appear after scrolling. The downloader layer also contains site-specific adapters where a stable fast path is useful.

Downloads and uploads are bounded and validated before they enter the processing pipeline.

### 2. Slice

Long webtoon pages are converted into processing-friendly slices.

The slicer does not treat the target slice height as a hard cut line. It searches for low-content and safe bands first, avoids known unsafe regions, and only falls back to a low-content or emergency boundary when a safe cut cannot be found.

Each slice keeps source-page identity, slice ordering, and stitch-core metadata so the review UI can present a continuous page while export can reconstruct the original long page.

### 3. Detect

Detection is local and ONNX-based.

The maintained detector stack can identify:

- speech bubbles
- free text
- text regions suitable for OCR
- outlined/SFX-like text recovered by secondary computer-vision logic

Detection records retain provenance and review metadata instead of collapsing everything into a single rectangle.

A key design rule is:

> **A detection proposal is not automatically an inpaint mask.**

Confidence, class, mask provenance, semantic type, OCR eligibility, and cleanup safety are kept separately so uncertain detections can be reviewed instead of being destructively painted over.

### 4. Clean / inpaint

Cleanup uses verified pixel masks and LaMa-based inpainting.

The system supports:

- the fixed `lama.onnx` backend
- the preferred `lama-manga-dynamic.onnx` backend when present
- tiled inference for long or narrow regions
- smart flat-fill shortcuts when the surrounding artwork is provably safe
- manual repaint and mask reset
- preserve/skip regions and other review-only boundaries

Artwork safety is enforced at the mask level. Automatic cleanup is not allowed to assume that a detector box is safe simply because the detector produced it.

User geometry is persisted in page coordinates so detector refreshes and mask remapping do not silently destroy editorial changes.

### 5. Review

The current UI is a unified canvas-oriented Review workspace rather than separate legacy screens.

The editor can:

- inspect the cleaned page and original artwork together
- navigate stitched long pages and slices
- review detections and OCR regions
- add, resize, move, merge, or remove text regions
- paint or reset cleanup masks
- preserve regions from destructive operations
- skip pages or regions intentionally
- edit translation and typography in-place
- re-render from the saved editorial state

The browser layer is designed to survive long-running chapter work: processing jobs belong to the backend, state is persisted, and refresh/navigation should reconnect to the current chapter rather than silently abandoning work.

### 6. OCR

OCR is a hybrid service rather than one universal recognizer.

The maintained production stack uses:

- **MangaOCR** for Japanese manga text
- **PP-OCRv6 / PaddleOCR** for Chinese, Korean, and English text

OCR is chapter-aware and revision-safe. Recognition results retain stable identity, reading-order information, and visual metadata where available.

OCR visual metadata can carry source text color, position, and size hints into the editor so the initial typeset result starts closer to the source artwork instead of rebuilding every property from scratch.

The service also uses bounded decoded-page caching and a fast recognition path with a full-page fallback when targeted recognition produces no usable result.

### 7. Translate

There are two translation modes.

#### Text translation

Chapter text objects can be translated in batches through configured OpenAI-compatible providers.

The translator:

- uses stable text-object IDs
- requires every returned ID to match an existing object
- rejects unknown or duplicated IDs
- keeps translation commits revision-safe
- refuses to overwrite newer editor changes
- supports a provider-specific budget guard where pricing is known

Direct DeepSeek translation keeps an explicit preflight cost estimate and configurable budget cap.

#### Vision translation

Vision translation works with the actual page context rather than OCR text alone.

For each slice, the service can send:

1. the **ORIGINAL** image containing the source lettering
2. the **CLEAN** image after inpainting

The model receives existing text-object IDs, source OCR as an optional hint, and stored pixel-space regions. It is instructed to translate only those existing objects.

Geometry, color, size, and placement are not regenerated by the model. They are already part of the editorial state.

The response is therefore an editorial patch, not a replacement page.

Vision translation can also return an optional installed-font choice for an existing text object.

### 8. Lettering and font matching

The renderer treats typography as editable data, not a one-shot post-processing step.

Text objects can retain and edit:

- translated text
- font
- font size
- weight
- stroke
- background
- alignment
- region geometry
- other renderer-supported style properties

The project includes a native comic-font catalog with **66 bundled OFL-1.1 fonts** across dialogue, emphasis, thought, narration, skill, SFX, horror, and romance categories.

Automatic font matching is the default path. It compares a source lettering crop against rendered candidates using deterministic CPU-side visual descriptors and returns ranked candidates with confidence/evidence. Explicit user or AI font choices are only accepted when they resolve to installed catalog fonts.

### 9. Render

Rendering is driven from persisted editorial state.

The renderer records the inputs used to produce the artifact and protects the publication step against stale edits. A render that finishes after the underlying page changed is not allowed to silently become the current render.

The chapter renderer can process all non-skipped pages, while the review workspace can display the current rendered result as a continuous stitched document.

### 10. Export

Final export is revision-safe.

Before creating the archive, the system captures the canonical page/render inputs. Encoding and stitching happen outside the long-held manifest lock. The inputs are then checked again before the ZIP is published.

For webtoon chapters, slices belonging to the same source page are vertically stitched back together using their persisted core ranges.

The result is a chapter ZIP containing the current published page artifacts rather than whatever files happened to exist on disk when export began.

## AI provider architecture

AI is intentionally a service layer around the local editor.

The current built-in provider registry includes:

- Google Gemini
- DeepSeek
- OpenAI
- OpenRouter
- Experiential Labs

The application can also register **custom OpenAI-compatible providers** with an HTTPS API base.

Provider configuration is capability-driven. A provider can advertise model discovery, visual QC, translation, cost tracking, and image transport separately.

The settings UI can:

- store provider credentials separately
- query a provider's `/models` catalog
- select an exact model ID
- configure custom OpenAI-compatible endpoints

Secrets are never embedded in provider URLs. Custom provider API bases must be public HTTPS URLs and are validated before outbound requests.

## Visual QC

Visual QC is separate from translation.

It is used to inspect processed pages and chapter regions for issues such as cleanup artifacts, lettering problems, or other visual regressions without letting a model directly become the editor of record.

The QC system supports chapter jobs, region-aware batching, revision-aware caching, bounded concurrency, cancellation/retry flows, provider isolation, and browser-tested result navigation.

The same provider registry used by the current settings surface allows built-in and custom visual-capable providers to participate without hard-wiring the UI to one vendor.

## CPU-first runtime

The supported baseline is CPU execution.

The production runtime has been repeatedly optimized around desktop-class CPU constraints:

- bounded page concurrency
- conservative ONNX Runtime thread usage
- OpenVINO execution on supported x86-64 Windows/Linux environments
- standard ONNX Runtime elsewhere
- cached decoded source pages for repeated OCR work
- bounded downloads and retry policy
- shorter manifest lock windows
- deterministic browser/runtime sanity checks

The current processing design intentionally favors a small bounded worker schedule over spawning many heavy inference jobs that compete for the same CPU and memory bandwidth.

GPU acceleration is not required for the core workflow.

## Models

The model binaries are intentionally excluded from Git.

Place these files in `models/`:

| Model | Role |
| --- | --- |
| `bubble_yolo.onnx` | Bubble/text detection |
| `text_segmenter.onnx` | Text segmentation and pixel-mask support |
| `lama-manga-dynamic.onnx` | Preferred dynamic-resolution inpainting backend |
| `lama.onnx` | Fixed-resolution LaMa fallback |

At runtime, the detector and segmenter are required. Inpainting requires either the dynamic model or the fixed LaMa model; when both exist, the dynamic model is preferred.

The repository therefore separates two kinds of validation:

- **Model-independent release gates** — source compilation, tests, browser/runtime checks, state and security invariants.
- **Model-artifact validation** — real ONNX detector/segmenter/inpaint behavior against locally supplied model files.

## Requirements

- Python 3.12
- CPU-capable machine
- Chromium for Playwright URL ingestion and browser regression checks
- The production baseline is tuned around two page workers
- Additional RAM is useful for OCR, large pages, and browser workloads

The image path accepts up to **100,000,000 decoded pixels** per image.

### Install locally

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
playwright install chromium
```

Copy the required ONNX files into `models/`, then start the application:

```bash
python run.py
```

Open:

```text
http://127.0.0.1:8000
```

### Docker

```bash
docker compose up --build
```

The container expects the model files through the mounted `./models` directory.

## Configuration

The runtime can be adjusted through environment variables such as:

- `HOST`
- `PORT`
- `RELOAD`
- `WORKERS`
- `MANGA_INPAINT_PRELOAD`

By default the server binds to localhost. A network-visible binding such as `0.0.0.0` should be protected by the deployment's firewall and authentication layer.

API keys are configured independently for AI providers:

```text
GEMINI_API_KEY
GOOGLE_API_KEY
DEEPSEEK_API_KEY
OPENAI_API_KEY
OPENROUTER_API_KEY
EXPLABS_API_KEY
```

## Safety and trust boundaries

The repository has treated security as part of the product rather than as a deployment afterthought.

Current protections include:

- remote URL scheme and hostname validation
- SSRF protection against local/private/link-local/reserved address ranges
- managed-path validation to prevent path escape
- upload size and file-count limits
- decoded image pixel limits
- ZIP/CBZ extraction limits and path-safe extraction
- bounded remote image/document downloads
- redirect controls on sensitive outbound requests
- safe browser URL checks for Playwright flows
- revision checks before publishing rendered and exported artifacts
- provider URL validation for custom AI endpoints

These controls protect the application boundaries, but a network-exposed deployment should still be treated as a service deployment rather than assuming localhost-grade trust.

## Main API surface

The REST API is organized around the same workflow exposed by the UI.

### Chapter and processing

- `POST /api/chapter` — import a chapter from URL
- `POST /api/chapter/upload` — import images, ZIP, or CBZ
- `POST /api/process_pages` — detect and clean selected pages
- `GET /api/chapter/{chapter_id}` — read the current chapter manifest
- `POST /api/workflow_checkpoint` — persist workflow state

### OCR and text objects

- `POST /api/ocr/chapter` — start chapter OCR
- `GET /api/ocr/chapter/{job_id}` — read OCR job state
- `POST /api/ocr/chapter/{job_id}/cancel` — cancel OCR
- `POST /api/ocr/chapter/{job_id}/retry` — retry stale/failed OCR work
- `POST /api/text_objects/ensure` — materialize stable editor text objects
- `POST /api/text_object/create|update|delete` — edit text objects

### Translation, render and export

- `POST /api/translate/chapter` — batch text translation
- `POST /api/render` — render one page
- `POST /api/render/chapter?chapter_id=...` — render the chapter
- `GET /api/download/{chapter_id}/{page_index}` — download the current page artifact
- `GET /api/export/{chapter_id}.zip` — export the chapter

### Visual QC and fonts

- `/api/visual_qc/...` — provider settings, model discovery, QC jobs, retry/cancel and results
- `/api/fonts` — bundled font catalog and stable `font_id` values

## Architecture

```text
                    ┌──────────────────────────────┐
                    │       Browser Workbench      │
                    │ Upload · Review · OCR · Edit │
                    │ Translate · Letter · Export  │
                    └──────────────┬───────────────┘
                                   │
                              FastAPI API
                                   │
          ┌────────────────────────┼────────────────────────┐
          │                        │                        │
          ▼                        ▼                        ▼
   Chapter Pipeline          Editorial State          AI Services
   import/slice/process      text/geometry/style      QC/translation
          │                        │                        │
    ┌─────┼─────┐                  │                 ┌─────┴─────┐
    │     │     │                  │                 │ providers │
    ▼     ▼     ▼                  │                 │ built-in  │
 detector OCR  inpaint             │                 │ + custom  │
    │     │     │                  │                 └───────────┘
    └─────┴─────┘                  │
          │                        │
          └──────────┬─────────────┘
                     ▼
              revision-safe render
                     │
                     ▼
              stitched chapter ZIP
```

## Project layout

```text
app/
  detector/           local bubble/text detection and recovery
  downloader/         HTTP, Playwright, site adapters, and webtoon slicing
  inpaint/            LaMa backends and mask geometry
  ocr/                MangaOCR + PP-OCRv6 services and jobs
  translation/        text and two-image vision translation
  render/             page/text rendering, font catalog and matching
  visual_qc/          visual inspection and provider orchestration
  routers/            FastAPI route layer
  static/             browser workbench
  pipeline*.py        chapter processing and runtime coordination
  manifest_utils.py   persistent chapter/editor state
  editorial_gate.py  artifact/editorial consistency rules
  security.py         request, URL, file and path boundaries
tests/                correctness, security and product regression tests
models/               local model artifacts; binaries are not committed
data/                 runtime chapter data (ignored)
docs/                  maintained architecture, UI, security and quality records
scripts/               release, browser, sanity and maintenance tooling
```

## Testing and release gates

The repository uses permanent release gates rather than relying on a single end-to-end happy path.

Install test dependencies and run:

```bash
python -m pip install -r requirements-test.txt
make release-check
```

Useful local commands:

```bash
make test
make test-browser
make health
make docker-up
```

The maintained checks cover:

- Python compilation and regression tests
- detector/inpaint mask authority rules
- geometry and revision safety
- OCR lifecycle and stale-result handling
- provider contracts and key redaction
- JavaScript runtime/syntax checks
- Chromium browser flows at desktop and mobile sizes
- long-image and stitched-page behavior
- end-to-end chapter closure

The central product-closure path is:

**processed page → OCR/text object → translation → revision-safe render → strict chapter export**

Provider network calls are stubbed where deterministic release tests require it; the editor state transitions, render publication, filesystem artifacts, and ZIP generation remain real.

## Development philosophy

The repository has gone through several major stages:

- early local script and security hardening
- completion of the first end-to-end translation-studio workflow
- permanent backend and frontend release gates
- consolidation into a unified Review/editor workspace
- CPU throughput and long-chapter hardening
- revision-safe OCR, translation, rendering, and export
- hybrid OCR and native font matching
- multi-provider Visual QC and configurable AI providers
- current custom-provider and two-image vision-translation work

The result is intentionally conservative about automatic mutation.

**Automation proposes. Editorial state decides. Published artifacts must match the current state.**

Experimental model research, benchmark-only code, temporary probes, and abandoned runtime approaches belong on experiment/archive branches until they become part of the maintained product.

## Current limitations

This project accelerates the production of translated chapters; it does not attempt to replace every part of professional lettering or redraw work.

Expect manual intervention for:

- difficult artwork reconstruction
- complex perspective or warped lettering
- curved/path text
- elaborate hand-drawn SFX recreation
- ambiguous OCR
- heavily protected or unusual reader sites
- typography choices that need artistic judgment

The system is strongest when used as a **reviewable editor with automation**, not as a black-box batch converter.

## License

See the repository's license files and bundled font metadata for software and asset licensing details.
