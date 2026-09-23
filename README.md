# Manga & Webtoon Translator Studio

Local-first, CPU-oriented manga / manhua / manhwa / webtoon translation studio for a human-in-the-loop workflow.

**Import → Slice → Detect/Clean → Review → OCR → Translate → Edit/Typeset → Render → ZIP**

The application keeps the editor in control. Detection, cleanup, OCR and translation create editable results; geometry, text, typography and cleanup can be corrected before export.

## Features

- Import PNG, JPEG, WEBP and BMP images, ZIP/CBZ archives, or chapter URLs.
- URL ingestion supports HTTP and Playwright discovery, relative URLs, `srcset`, common lazy-load attributes and scroll-until-stable discovery.
- Slice long webtoon pages into CPU-friendly segments with safe cut-band selection and non-overlapping stitch ownership.
- Detect speech bubbles and free text with local ONNX models while retaining model/class provenance.
- Recover outlined/SFX/free text with OpenCV/MSER without turning uncertain recovery into destructive cleanup.
- Clean text with verified pixel masks and LaMa. Proposal boxes are never treated as masks; uncertain detections and watermarks remain review-only.
- Remap detector masks and geometry edits in page coordinates so user edits survive re-detection.
- Review pages, repair cleanup manually and run optional visual QC through built-in or custom AI providers.
- OCR Japanese with MangaOCR and Chinese/Korean/English with PaddleOCR.
- Convert detector/OCR regions into stable editable text objects without overwriting user edits.
- Translate chapters through configured OpenAI-compatible providers with stale-write protection.
- Edit text, font, size, bold, stroke, background, alignment and text-region geometry.
- Native comic-font matcher runs automatically during rendering; the built-in catalog contains 70 OFL-licensed comic fonts.
- Render from persisted editor state and export a strict chapter ZIP. Webtoon slices from one source page are stitched back together for export.
- Long decoded images are supported up to 100 million pixels; browser coverage includes long-image and mixed-width stitched-page cases.

## Requirements

- Python 3.12 is the release-gate runtime.
- CPU execution is the supported baseline; GPU is not required.
- Chromium is required for Playwright URL ingestion and browser regression checks.
- The production baseline uses two page workers and is validated on a 4 GiB memory limit; additional RAM is useful for OCR and browser workloads.

### Install

```bash
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
playwright install chromium
```

### Local models

Put the following files in `models/`:

- `bubble_yolo.onnx`
- `text_segmenter.onnx`
- `lama.onnx`

Optional preferred inpaint model:

- `lama-manga-dynamic.onnx` — used automatically when present; `lama.onnx` remains the fallback.

Model binaries are intentionally not committed to Git. The model-dependent detector/inpaint path is therefore a local artifact gate.

The inpaint session is normally prepared during server startup. Set `MANGA_INPAINT_PRELOAD=0` to restore fully lazy loading when lower idle memory is preferred.

### Start

```bash
python run.py
```

Open `http://127.0.0.1:8000`.

Docker is also supported:

```bash
docker compose up --build
```

The Docker image expects model files through the mounted `./models` directory.

## AI providers

Visual QC supports the built-in Google Gemini, DeepSeek, OpenAI, OpenRouter and Experiential Labs providers. The application can also register custom AI providers using the OpenAI-compatible protocol.

Built-in credentials use:

| Provider | Environment variable |
| --- | --- |
| Google Gemini | `GEMINI_API_KEY` or `GOOGLE_API_KEY` |
| DeepSeek | `DEEPSEEK_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| OpenRouter | `OPENROUTER_API_KEY` |
| Experiential Labs | `EXPLABS_API_KEY` |

Custom providers use an HTTPS API base and OpenAI-compatible endpoints. Credentials are not accepted inside the API URL. Provider IDs are restricted to lowercase letters, numbers, `-` and `_`.

Use the settings UI to configure provider keys, discover `/models`, choose an exact model ID, or register a custom provider.

### Translation

Chapter translation is available through DeepSeek, OpenAI, OpenRouter and Experiential Labs, plus configured custom OpenAI-compatible providers. Direct DeepSeek translation uses the configured DeepSeek translation model with thinking disabled and retains the verified preflight cost check. The UI defaults to a `$0.02` DeepSeek cap.

Translation commits are revision-safe: object identity, OCR source text and existing translation are checked again after the network request. Concurrent editor changes win instead of being overwritten.

## Detection and inpainting safety

- Detection records retain source model, class, semantic type, mask provenance, cleanup safety, OCR eligibility and review state.
- NMS is class-aware, so bubble and free-text proposals do not suppress each other blindly.
- Proposal geometry is not an inpaint mask. Only verified pixel masks can trigger automatic cleanup.
- Watermarks, uncertain recovery and detector disagreement remain review-only.
- Content-heavy zero-box pages and unresolved regions are surfaced through detection state and review metadata instead of silently passing as clean.
- Flat-fill cleanup checks a clean ring and edge density before bypassing LaMa; nearby artwork edges force model inpainting.
- Long/narrow crops use tiled LaMa inference where required instead of squeezing the crop into a thin strip.
- Webtoon seams retain detector overlap context while stitch/export assigns each source pixel exactly once.
- Low-memory ONNX Runtime settings and CPU thread defaults respect the available CPU/memory environment.

## Main API surface

### Chapter and processing

- `POST /api/chapter` — import from URL.
- `POST /api/chapter/upload` — import images, ZIP or CBZ.
- `POST /api/process_pages` — detect and inpaint selected pages.
- `GET /api/chapter/{chapter_id}` — read the current chapter manifest.
- `POST /api/workflow_checkpoint` — persist editor stage/page state.

### OCR and editor

- `POST /api/ocr/chapter` — start chapter OCR.
- `GET /api/ocr/chapter/{job_id}` — read OCR job status.
- `POST /api/ocr/chapter/{job_id}/cancel` — cancel OCR.
- `POST /api/ocr/chapter/{job_id}/retry` — retry failed/stale OCR targets.
- `POST /api/text_objects/ensure` — create stable editor text objects from detected regions.
- `POST /api/text_object/create|update|delete` — manage manual text objects.

### Translation, render and export

- `POST /api/translate/chapter` — run budgeted chapter translation.
- `POST /api/render` — render one page from current revision state.
- `POST /api/render/chapter?chapter_id=...` — render all non-skipped pages.
- `GET /api/download/{chapter_id}/{page_index}` — download the current single page.
- `GET /api/export/{chapter_id}.zip` — export the current chapter.

### Visual QC and settings

- `/api/visual_qc/...` — visual-QC providers, model discovery, chapter jobs, retry/cancel and provider settings.
- `/api/fonts` — stable comic-font catalog and `font_id` values.

## Development and release checks

Install the test dependencies and run the maintained local gate:

```bash
python -m pip install -r requirements-test.txt
make release-check
```

Useful commands:

```bash
make test
make test-browser
make health
make docker-up
```

The release checks compile production code, run the maintained pytest suite, run browser/runtime sanity checks, and verify release, responsiveness and inpaint-authority invariants. The live UI workflow starts the real FastAPI application and exercises Chromium at desktop and mobile widths.

The chapter E2E path covers:

**processed OCR box → stable text object → translation commit → revision-safe render → strict ZIP export**

External provider calls are stubbed in deterministic release tests; filesystem rendering, persisted editor state and ZIP generation remain real.

For a manually reviewed processed bundle, `scripts/audit_processed_bundle.py` measures correction rate, review workload, residue state and persisted mask geometry. Completion metrics are not presented as detector recall.

## Project layout

```text
app/              production application
  detector/       bubble/text detection and recovery
  downloader/     URL/local ingestion and webtoon slicing
  inpaint/        LaMa cleanup and mask geometry safety
  ocr/            MangaOCR/PaddleOCR services and jobs
  render/         typography, font matching and render identity
  translation/    OpenAI-compatible chapter translation
  visual_qc/      visual quality inspection and provider orchestration
  routers/        FastAPI endpoints
  static/         browser UI
  pipeline.py     chapter workflow coordination
tests/            correctness, security and product-release regression tests
models/           local model files; binaries are not committed
data/             runtime chapter data (ignored)
docs/             maintained architecture, UI, security and quality records
```

## Repository hygiene

`main` contains maintained product code and release-critical checks, not exploratory benchmark generations or one-off debug artifacts. Historical benchmark/debug material is kept on dedicated archive/experiment branches.

New experiments should stay on feature or benchmark branches until they become part of the maintained product or release gate.

## Release rule

A change belongs on the release path when it helps a real chapter reach a correct, reviewable export without weakening cleanup, geometry, editor-state or security invariants. Experimental model work, benchmark-only tooling and advanced lettering features remain outside the release path until they are product-critical and validated.
