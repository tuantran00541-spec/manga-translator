# Manga & Webtoon Translator Studio — v0.2

Local-first, CPU-oriented manga / manhua / manhwa / webtoon translation studio.

The v0.2 product flow is intentionally narrow and complete:

**Import → Slice → Detect/Clean → Review → OCR → Auto-translate → Edit/Typeset → Render chapter → ZIP export**

The application keeps a human editor in control. Automatic steps create a first pass; the editor can correct OCR, translation, geometry, typography and inpainting before export.

## What v0.2 does

- Import local PNG/JPEG/WEBP/BMP images, ZIP or CBZ archives.
- Import chapter URLs with HTTP + Playwright discovery, including relative image URLs, `srcset`, common lazy-load attributes, and scroll-until-stable discovery.
- Slice long webtoon images into CPU-friendly segments while preferring safe cut bands; unsafe boundaries keep detector-only overlap context while stitch ownership remains non-overlapping.
- Detect speech bubbles and free text with local ONNX models while preserving detector/class provenance; secondary OpenCV/MSER recovery surfaces outlined SFX/free text that the segmenter misses.
- Protect line art during cleanup with verified pixel masks only: proposal-only detections, uncertain recovery, and watermarks remain review-only instead of becoming destructive rectangle inpaint. Fixed-LaMa tiling and page-space mask remapping remain available for compatibility.
- Review cleaned pages, repaint mistakes manually, and optionally use Gemini or DeepSeek visual QC.
- OCR Japanese with MangaOCR and Chinese/Korean/English with PaddleOCR.
- Automatically turn detector boxes into editable text objects; re-processing keeps stable box links and does not overwrite user geometry/text edits.
- Translate OCR text in chapter batches with DeepSeek V4 Flash. Translation is opt-in, uses the DeepSeek key already stored by the app, has a per-run USD budget (UI default `$0.02`), and rejects stale writes if the editor changes text while the request is in flight.
- Edit translation, font, font size, bold, stroke, background, alignment, and text-region geometry.
- Render every non-skipped page with the revision-safe renderer.
- Export one ZIP for the chapter. Webtoon slices belonging to the same source page are vertically stitched back together before packaging.

## Current limits

v0.2 is designed to accelerate a human editor, not replace professional redraw/lettering work. Rotation, curved/path text, perspective/warp typography, advanced SFX recreation, and difficult art redraw still need external/editor intervention. URL scraping is generic rather than site-specific, so heavily protected readers may still require local upload.

Production ONNX binaries are intentionally not stored in Git. Model-dependent detector/inpaint validation therefore remains a local/model-artifact gate; the repository CI verifies the model-independent product closure path.

## Requirements

- Python 3.12 is the release-gate runtime.
- CPU execution is the supported baseline; GPU is not required.
- The v0.2 chapter acceptance gate is exercised under a 4 GiB memory limit with the production default two page workers; additional RAM is still useful for OCR/browser workloads.
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

## Inpainting safety

- Flat-fill shortcuts inspect a clean ring and edge density before bypassing LaMa; artwork edges near text force model inpainting.
- Missing detector segmentation masks no longer silently become destructive full-rectangle masks. Explicit manual boxes retain their rectangle fallback.
- Long, narrow crops on the fixed 512×512 LaMa model use tiled inference instead of squeezing the crop into a thin strip.
- Detector geometry edits remap masks in page coordinates instead of stretching or discarding them. Re-detection reconciles fresh detector masks to the persisted user geometry.
- Regression tests verify local mask behavior, geometry remapping, and that automatic compositing leaves pixels outside the effective mask unchanged.


## v0.2 detection and CPU-memory safety

- Detection records retain `source_model`, class id/name, semantic type, mask provenance, inpaint safety, OCR eligibility, and review state.
- NMS is class-aware, so `text_bubble` and `text_free` proposals do not suppress each other blindly.
- Proposal geometry is not an inpaint mask. Only verified pixel masks can trigger automatic cleanup; watermarks and uncertain outlined/SFX recovery are review-only.
- Content-heavy zero-box pages and detector disagreement are surfaced through `detection_state`, `detection_issues`, `unverified_regions`, and `needs_review` instead of silently passing as clean.
- Unsafe webtoon seams keep overlap context for detection while export/stitch owns each source core pixel exactly once. Intermediate slices use lossless PNG to avoid repeated WebP encoder retention/artifacts.
- ONNX Runtime CPU arena/memory-pattern retention is disabled by default on the low-memory path, thread defaults respect cgroup CPU quota, and dynamic LaMa can run across the production two-page worker schedule while the fixed fallback uses a serialized compatibility path.

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

The gate compiles source, runs the v0.1 + v0.2 model-independent integration/regression suite (including artwork-safety and geometry-mask tests), then runs Chromium regressions. `tests/test_v01_product_closure.py` specifically validates the connected path:

**processed OCR box → auto text object → translation commit → revision-safe render → strict ZIP export**

The external DeepSeek network call is stubbed in that test; the product state transitions, filesystem render, render identity, and ZIP generation are real.

## Project layout

```text
app/              production application
  detector/       local bubble/text detection
  downloader/     URL/local ingestion and webtoon slicing
  inpaint/        LaMa cleanup + mask geometry safety
  ocr/            MangaOCR/PaddleOCR service + jobs
  translation/    DeepSeek chapter translator
  render/         typography + render identity
  routers/        FastAPI endpoints
  static/         browser UI
  visual_qc/      Gemini/DeepSeek image QC
tests/            correctness, security and product-release regression tests
models/           local model files + setup note; binaries are not committed
data/             runtime chapter data (ignored)
docs/             maintained architecture/UI/security history
```

## Repository hygiene

The production branch intentionally does not carry exploratory benchmark generations, one-off debug scripts, frozen benchmark JSON, or stale model hash manifests. The pre-v0.1 benchmark/debug tree is preserved intact on `archive/pre-v0.1-benchmarks` for future archaeology or model experiments.

Release-critical tests use functional names rather than phase numbers. New experiments should live on a feature/benchmark branch and only enter `main` when they become part of the maintained product or release gate.

## Release rule

For v0.2, new work must answer one question: **does it block a real chapter from reaching a correct export?** Non-blocking model experiments, benchmark refactors, extra providers, and advanced lettering features stay outside the release branch until the product-closure gate is green.
