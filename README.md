# Manga & Webtoon Translator Studio

[![Release Gate](https://github.com/tuantran00541-spec/manga-translator/actions/workflows/release-gate.yml/badge.svg)](https://github.com/tuantran00541-spec/manga-translator/actions/workflows/release-gate.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/backend-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![CPU First](https://img.shields.io/badge/runtime-CPU--first-222222)](#cpu-first)

**A local-first manga, manhwa, manhua, and webtoon translation & lettering studio.**

Detect text, clean artwork, OCR, translate, typeset, review, render, and export complete chapters — while keeping the editor in control of the final result.

**Import → Slice → Detect → Clean → Review → OCR → Translate → Letter → Render → Export**

### Real production demo — Chapter 60

<p align="center">
  <strong>Slice 005_02</strong>
</p>
<p align="center">
  <img src="https://raw.githubusercontent.com/tuantran00541-spec/manga-translator/trial/chapter-render-60/trials/chapter-60/demo/01-raw.jpg" alt="Chapter 60 slice 005_02 raw" width="32%">
  <img src="https://raw.githubusercontent.com/tuantran00541-spec/manga-translator/trial/chapter-render-60/trials/chapter-60/demo/01-detector.jpg" alt="Chapter 60 slice 005_02 detector" width="32%">
  <img src="https://raw.githubusercontent.com/tuantran00541-spec/manga-translator/trial/chapter-render-60/trials/chapter-60/demo/01-clean.jpg" alt="Chapter 60 slice 005_02 clean" width="32%">
</p>

<p align="center">
  <strong>Slice 015_04</strong>
</p>
<p align="center">
  <img src="https://raw.githubusercontent.com/tuantran00541-spec/manga-translator/trial/chapter-render-60/trials/chapter-60/demo/02-raw.jpg" alt="Chapter 60 slice 000_00 raw" width="32%">
  <img src="https://raw.githubusercontent.com/tuantran00541-spec/manga-translator/trial/chapter-render-60/trials/chapter-60/demo/02-detector.jpg" alt="Chapter 60 slice 000_00 detector" width="32%">
  <img src="https://raw.githubusercontent.com/tuantran00541-spec/manga-translator/trial/chapter-render-60/trials/chapter-60/demo/02-clean.jpg" alt="Chapter 60 slice 000_00 clean" width="32%">
</p>

<p align="center"><sub>Two real Chapter 60 slices through the production pipeline: RAW → text detection → CLEAN.</sub></p>

> **Automation proposes. Editorial state decides. Published artifacts must match the current state.**

## Why this project exists

Most automated manga translators treat a page as something to transform once.

This project treats a chapter as an **editable production workspace**.

Detection, cleanup, OCR, translation, typography, and AI assistance are all allowed to make mistakes. The editor can inspect, correct, reject, repaint, resize, retranslate, or restyle those results before anything becomes the published chapter artifact.

The application is local-first and CPU-oriented. Local ONNX models provide the core detection and inpainting pipeline; OCR and external AI providers are optional layers around that local workflow.

## The workflow

| Stage | What happens |
| --- | --- |
| **1. Import** | Images, ZIP/CBZ archives, or chapter URLs |
| **2. Slice** | Long webtoon pages become processing-friendly slices |
| **3. Detect** | Speech bubbles, free text, OCR regions, outlined/SFX text |
| **4. Clean** | Verified-mask LaMa inpainting and manual repair |
| **5. Review** | Unified canvas workspace and editorial corrections |
| **6. OCR** | MangaOCR + PP-OCRv6 hybrid recognition |
| **7. Translate** | Batch text translation or two-image vision translation |
| **8. Letter** | Text objects, geometry, typography, font matching |
| **9. Render** | Revision-safe page publication |
| **10. Export** | Stitched chapter ZIP |

## Features

### 📥 Import

- PNG, JPEG, WEBP and BMP
- ZIP and CBZ chapters
- Chapter URLs through HTTP and Playwright
- Relative URLs, srcset, common lazy-load attributes, and scroll-based discovery
- Site-specific downloader adapters where a stable fast path is useful
- Bounded downloads and upload validation

### ✂️ Smart webtoon slicing

Long pages are split into CPU-friendly slices without blindly cutting at a fixed height.

The slicer prefers low-content and safe bands, preserves source-page and slice identity, and records stitch ownership so a webtoon page can be reconstructed without duplicated or missing pixels.

### 🔎 Detection

Local ONNX inference detects:

- speech bubbles
- free text
- OCR-ready regions
- outlined/SFX-like text recovered by secondary CV logic

Detection records retain source/model/class provenance and review state.

### 🧹 Artwork-safe cleanup

Cleanup is driven by **verified pixel masks**, not detector rectangles.

The pipeline supports:

- LaMa inpainting
- dynamic and fixed LaMa backends
- tiled inference for long/narrow regions
- safe flat-fill shortcuts
- manual repaint
- mask reset
- preserve/skip regions
- page-coordinate mask remapping

Uncertain detections, watermarks, and review-only regions are not silently promoted to destructive cleanup.

### 👁️ OCR

Hybrid chapter OCR uses:

- **MangaOCR** for Japanese
- **PP-OCRv6 / PaddleOCR** for Chinese, Korean, and English

OCR is chapter-aware, revision-safe, and can preserve source visual metadata such as lettering color, position, and size hints for downstream typesetting.

### 🌐 Translation

Two complementary modes are available.

**Text translation**

Batch existing text objects through configured OpenAI-compatible providers with stable IDs, stale-write protection, and provider-specific budget controls where pricing is known.

**Vision translation**

Send the **ORIGINAL** and **CLEAN** versions of the same slice together with existing text-object IDs and stored regions.

The model translates only the objects that already exist. Geometry, placement, color, and size remain editor-owned state rather than model-generated replacements.

### ✍️ Lettering

The browser workbench provides region-centric editing for:

- translated text
- font and size
- weight
- stroke
- background
- alignment
- text-region geometry
- manual text objects

The current UI is a unified Review workspace rather than a collection of legacy editor screens.

### 🔤 Automatic comic-font matching

The renderer includes a native catalog of **66 bundled OFL-1.1 comic fonts** across dialogue, emphasis, thought, narration, skill, SFX, horror, and romance categories.

Automatic font matching runs by default using deterministic CPU-side visual comparison of original lettering crops.

User and AI font choices are validated against the installed catalog instead of accepting arbitrary filesystem paths.

### 🧠 AI providers

Built-in providers currently include:

- Google Gemini
- DeepSeek
- OpenAI
- OpenRouter
- Experiential Labs

The settings layer can also register custom **OpenAI-compatible** providers with validated HTTPS API bases.

Providers advertise capabilities independently, allowing the same registry to power model discovery, translation, and Visual QC without hard-wiring the UI to one vendor.

### 🔍 Visual QC

Visual QC is an inspection layer, not the editor of record.

The system supports region-aware chapter inspection, revision-aware caching, bounded concurrency, cancel/retry flows, provider-isolated results, and browser-tested navigation of QC findings.

### 📦 Revision-safe render and export

Rendered pages and chapter exports are tied to canonical editorial state.

If an editor changes relevant content while expensive rendering or export work is running, stale artifacts are rejected instead of becoming the current published result.

Long webtoon slices are stitched back into their source-page structure during chapter export.

## Demo artifacts

The repository has kept several historical chapter trials for regression and visual review.

Examples include:

- source and slice contact sheets from **trial/chapter-render-60**
- rendered page sheets from **trial/chapter-render-60**
- OCR/translation/render visual proofs from **trial/chapter-render-251**
- long-page and mixed-width review coverage from later chapter runs

The demo above uses two real Chapter 60 slices selected from the imported production artifact for visual richness. Each set shows the same slice as **RAW → detector overlay → CLEAN**, with the detector boxes and inpaint result produced by the production pipeline.

The generated proof images live beside the historical trial evidence in `trials/chapter-60/demo/`. Full runtime chapter images remain artifacts rather than permanent Git fixtures.

## Quick start

### Requirements

- Python 3.12
- CPU-capable machine
- Chromium for Playwright URL ingestion and browser regression checks
- Additional RAM is useful for OCR and very large pages

The image path accepts up to **100,000,000 decoded pixels** per image.

### Install

~~~bash
git clone https://github.com/tuantran00541-spec/manga-translator.git
cd manga-translator

python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
playwright install chromium
~~~

### Models

Place these files in models/:

| File | Purpose |
| --- | --- |
| bubble_yolo.onnx | Bubble/text detection |
| text_segmenter.onnx | Text segmentation and pixel masks |
| lama-manga-dynamic.onnx | Preferred inpainting backend |
| lama.onnx | Fixed-resolution fallback |

Model binaries are intentionally not committed to Git.

### Run

~~~bash
python run.py
~~~

Open:

~~~text
http://127.0.0.1:8000
~~~

### Docker

~~~bash
docker compose up --build
~~~

The Docker image expects model files through the mounted ./models directory.

## AI setup

Provider keys are configured independently in the settings UI or through environment variables:

~~~text
GEMINI_API_KEY
GOOGLE_API_KEY
DEEPSEEK_API_KEY
OPENAI_API_KEY
OPENROUTER_API_KEY
EXPLABS_API_KEY
~~~

The settings UI can discover provider models through /models, select exact model IDs, and register custom OpenAI-compatible providers.

Custom provider endpoints must use public HTTPS URLs. Credentials are never accepted inside the API URL.

## Runtime settings

Detection, inpainting, OCR and rendering thresholds are fixed constants in `app/parameters.py`. Only operational settings read environment variables:

| Area | Variables |
| --- | --- |
| Processing | `MANGA_PIPELINE_DEFAULT_WORKERS`, `MANGA_PIPELINE_SLICE_WORKER_LIMIT`, `MANGA_USE_DYNAMIC_LAMA`, `MANGA_INPAINT_PRELOAD`, `MANGA_DETECTOR_RESIDUE_VERIFY_ENABLED`, `MANGA_FIXED_LAMA_*` |
| ONNX Runtime | `MANGA_ORT_PROVIDER`, `MANGA_ORT_REQUIRE_PROVIDER`, `MANGA_ORT_INTRA_OP_THREADS`, `MANGA_ORT_OPENVINO_*`, `MANGA_ORT_CPU_MEM_ARENA`, `MANGA_ORT_MEM_PATTERN`, `MANGA_ORT_SERIALIZE_INFERENCE` |
| OCR | `MANGA_PPOCRV6_TIER`, `MANGA_PPOCRV6_TEXTLINE_ORIENTATION`, `MANGA_OCR_TARGET_SELECTION`, `MANGA_OCR_IMAGE_CACHE_MB`, `MANGA_OCR_JOB_CONCURRENCY_LIMIT`, `MANGA_OCR_JOB_ACTIVE_LIMIT` |
| Network and AI | `MANGA_DOWNLOAD_WORKERS`, `MANGA_DOWNLOAD_JS_NAVIGATION_TIMEOUT_MS`, `MANGA_REMOTE_CONNECT_TIMEOUT_SECONDS`, `MANGA_TRANSLATION_CONNECT_TIMEOUT_SECONDS`, `MANGA_TRANSLATION_READ_TIMEOUT_SECONDS`, `MANGA_VISUAL_QC_*` |

`python scripts/parameter_report.py --env-only` prints the effective values of the ones defined in `app/parameters.py`.

## CPU-first design

The supported baseline is CPU execution.

The runtime has been optimized around bounded desktop-class workloads instead of assuming a GPU:

- bounded page concurrency
- conservative ONNX Runtime threading
- OpenVINO on supported x86-64 Windows/Linux systems
- standard ONNX Runtime elsewhere
- decoded-page caching for repeated OCR work
- bounded downloads and transient retries
- short manifest lock windows
- deterministic browser/runtime checks

The default production processing schedule is intentionally small enough to avoid multiplying heavy inference kernels until the machine becomes memory- or bandwidth-bound.

GPU acceleration is not required for the core workflow.

## Safety by design

Safety is part of the processing model, not just deployment hardening.

The application protects:

- remote URL imports against SSRF and unsafe address ranges
- managed files against path escape
- uploads and archives against oversized payloads
- decoded images against excessive pixel counts
- browser requests against unsafe remote targets
- custom AI endpoints against invalid/untrusted URLs
- rendered/exported artifacts against stale editor state

Network-exposed deployments still need normal firewall and authentication controls.

## Architecture

At a high level:

~~~text
                    Browser Workbench
                          │
                       FastAPI
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
   Processing         Editorial           AI
    Pipeline            State          Providers
        │                 │                 │
 Detect · OCR ·      Text · Geometry    Translate · QC
 Inpaint · Render    · Typography
        └─────────────────┼─────────────────┘
                          ▼
                  Revision-safe artifacts
                          │
                          ▼
                   Stitched ZIP export
~~~

For implementation details see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Documentation

The README is intentionally product-focused. Deeper engineering material lives in docs/.

- [Architecture](docs/ARCHITECTURE.md)
- [Security history](docs/SECURITY_HISTORY.md)
- [Model E2E gate](docs/MODEL_E2E_GATE.md)
- [Processed chapter quality baseline](docs/PROCESSED_CHAPTER_QUALITY_BASELINE_20260912.md)
- [Comic font library](docs/comic-fonts.md)
- [UI guidelines](docs/UI_GUIDELINES.md)

## Development

Install test dependencies:

~~~bash
python -m pip install -r requirements-test.txt
~~~

Useful commands:

~~~bash
make test
make test-browser
make health
make release-check
make docker-up
~~~

The maintained release path covers source compilation, regression tests, browser/runtime checks, OCR and provider contracts, mask authority, geometry/revision safety, long-image behavior, and the connected product flow:

**processed page → OCR/text object → translation → render → chapter ZIP**

Model-dependent validation remains a separate local artifact gate because production ONNX binaries are not stored in Git.

To screenshot every screen on a real chapter with the real models, run the **UI tour** workflow from the Actions tab on a feature branch. It imports the chapter, skips all but the chosen slices, repaints a region with LaMa, preserves a cleaned bubble and reprocesses, checks each result pixel by pixel, and commits the screenshots and `report.json` to `audit-results/ui-tour/` on that branch. Against a running local server: `python scripts/ui_tour.py --chapter-url <url> --keep 16,19,24,27`.

## Project layout

~~~text
app/
  detector/       bubble/text detection and recovery
  downloader/     HTTP, Playwright, adapters and slicing
  inpaint/        LaMa and mask geometry safety
  ocr/            MangaOCR + PP-OCRv6
  translation/    text and two-image vision translation
  render/         typography, font catalog and matching
  visual_qc/      visual inspection and provider orchestration
  routers/        FastAPI API surface
  static/         browser workbench
  pipeline*.py    chapter processing and runtime coordination
  manifest_utils.py
  editorial_gate.py
  security.py

tests/            correctness and release regressions
models/           local model artifacts
data/             runtime chapter data
docs/             maintained engineering documentation
scripts/          release, browser and sanity tooling
~~~

## Current limitations

This project is designed to accelerate chapter production, not eliminate every form of professional redraw and lettering work.

Manual intervention is still expected for:

- difficult artwork reconstruction
- perspective or heavily warped lettering
- curved/path text
- complex hand-drawn SFX recreation
- ambiguous OCR
- heavily protected reader sites
- typography requiring artistic judgment

The strongest use case is a **reviewable production workstation with automation**, not a black-box batch converter.

## Project philosophy

The repository has evolved through security hardening, end-to-end product closure, detector/inpaint safety work, hybrid OCR, CPU throughput work, a unified Review workspace, revision-safe rendering/export, Visual QC, native font matching, configurable AI providers, and vision translation.

Those features all reinforce the same rule:

> **The page belongs to the editor.**

Automation can detect it, clean it, read it, translate it, suggest a font, or inspect it — but the current editorial state remains authoritative until the final artifact is published.

## License

See the repository license and bundled asset metadata for software and font licensing details.
