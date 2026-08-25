# Manga & Webtoon Translator Studio — v0.1

Local-first, CPU-oriented manga / manhua / manhwa / webtoon translation studio.

The v0.1 product flow is intentionally narrow and complete:

**Import → Slice → Detect/Clean → Review → OCR → Auto-translate → Edit/Typeset → Render chapter → ZIP export**

The application keeps a human editor in control. Automatic steps create a first pass; the editor can correct OCR, translation, geometry, typography and inpainting before export.

## What v0.1 does

- Import local PNG/JPEG/WEBP/BMP images, ZIP or CBZ archives.
- Import chapter URLs with HTTP + Playwright discovery, including relative image URLs, `srcset`, common lazy-load attributes, and scroll-until-stable discovery.
- Slice long webtoon images into CPU-friendly segments while preferring low-content cut bands.
- Detect speech/text regions with local ONNX models and clean source text using LaMa inpainting.
- Review cleaned pages, repaint mistakes manually, and optionally use Gemini or DeepSeek visual QC.
- OCR Japanese with MangaOCR and Chinese/Korean/English with PaddleOCR.
- Automatically turn detector boxes into editable text objects; re-processing keeps stable box links and does not overwrite user geometry/text edits.
- Translate OCR text in chapter batches with DeepSeek V4 Flash. Translation is opt-in, uses the DeepSeek key already stored by the app, has a per-run USD budget (UI default `$0.02`), and rejects stale writes if the editor changes text while the request is in flight.
- Edit translation, font, font size, bold, stroke, background, alignment, and text-region geometry.
- Render every non-skipped page with the revision-safe renderer.
- Export one ZIP for the chapter. Webtoon slices belonging to the same source page are vertically stitched back together before packaging.

## Current limits

v0.1 is designed to accelerate a human editor, not replace professional redraw/lettering work. Rotation, curved/path text, perspective/warp typography, advanced SFX recreation, and difficult art redraw still need external/editor intervention. URL scraping is generic rather than site-specific, so heavily protected readers may still require local upload.

Production ONNX binaries are intentionally not stored in Git. Model-dependent detector/inpaint validation therefore remains a local/model-artifact gate; the repository CI verifies the model-independent product closure path.

## Requirements

- Python 3.12 is the release-gate runtime.
- CPU execution is the supported baseline; GPU is not required.
- 8 GB RAM or more is recommended.
- Chromium is required for Playwright URL ingestion and browser regression tests.

Install:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
playwright install chromium
```

Required local models in `models/`:

- `bubble_yolo.onnx`
- `text_segmenter.onnx`
- `lama.onnx`

Optional preferred inpaint model:

- `lama-manga-dynamic.onnx` — used automatically when present; `lama.onnx` remains the fallback.

Start:

```bash
python run.py
```

Open `http://127.0.0.1:8000`.

Docker is also supported:

```bash
docker compose up --build
```

The Docker image expects model files to be supplied through the mounted `./models` directory.

## DeepSeek translation

Configure a DeepSeek API key through the app's AI settings (or `DEEPSEEK_API_KEY`). Translation uses the current `deepseek-v4-flash` chat-completion API in non-thinking JSON mode. The chapter translator performs a preflight cost check and the UI defaults to a `$0.02` cap per run; increase it only when a larger chapter needs it.

The provider response is never committed blindly: object identity, OCR source text, and existing translation are checked again after the network call. Concurrent editor changes win and are counted as stale instead of being overwritten.

## Main API surface

### Chapter / processing

- `POST /api/chapter` — import from URL.
- `POST /api/chapter/upload` — import images / ZIP / CBZ.
- `POST /api/process_pages` — detect and inpaint selected pages.
- `GET /api/chapter/{chapter_id}` — current manifest.
- `POST /api/workflow_checkpoint` — persist current editor stage/page.

### OCR / editor automation

- `POST /api/ocr/chapter` — start chapter OCR job.
- `GET /api/ocr/chapter/{job_id}` — OCR job status.
- `POST /api/ocr/chapter/{job_id}/cancel` — cancel OCR.
- `POST /api/ocr/chapter/{job_id}/retry` — retry failed/stale OCR targets.
- `POST /api/text_objects/ensure` — map detected boxes into stable editor text objects.
- `POST /api/text_object/create|update|delete` — manual text-object editing.

### Translation / render / export

- `POST /api/translate/chapter` — budgeted chapter translation.
- `POST /api/render` — revision-safe single-page render.
- `POST /api/render/chapter?chapter_id=...` — render all non-skipped pages from persisted editor state.
- `GET /api/download/{chapter_id}/{page_index}` — strict current single-page download.
- `GET /api/export/{chapter_id}.zip` — strict chapter ZIP export.

### Visual QC

- `/api/visual_qc/...` — Gemini / DeepSeek quality inspection, chapter jobs, retry/cancel and key settings.

## Release gate

The permanent GitHub Actions workflow is `.github/workflows/release-gate.yml`.

Local equivalent:

```bash
make release-check
```

The gate compiles source, runs the v0.1 model-independent integration/regression suite, then runs Chromium regressions. `tests/test_v01_product_closure.py` specifically validates the connected path:

**processed OCR box → auto text object → translation commit → revision-safe render → strict ZIP export**

The external DeepSeek network call is stubbed in that test; the product state transitions, filesystem render, render identity, and ZIP generation are real.

## Project layout

```text
app/
  detector/       local bubble/text detection
  downloader/     URL/local ingestion and webtoon slicing
  inpaint/        LaMa cleanup
  ocr/            MangaOCR/PaddleOCR service + jobs
  translation/    DeepSeek chapter translator
  render/         typography + render identity
  routers/        FastAPI endpoints
  static/         browser UI
  visual_qc/      Gemini/DeepSeek image QC
tests/            regression + product-closure tests
models/           local model binaries (not committed)
data/             runtime chapter data
```

## Release rule

For v0.1, new work must answer one question: **does it block a real chapter from reaching a correct export?** Non-blocking model experiments, benchmark refactors, extra providers, and advanced lettering features stay outside the release branch until the product-closure gate is green.
