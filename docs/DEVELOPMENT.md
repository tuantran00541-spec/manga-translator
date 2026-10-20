# Development

Tests, release checks, the code layout and the architecture.

## Development

~~~bash
python -m pip install -r requirements-test.txt
make test            # regression tests
make test-browser    # browser and runtime checks
make health
make release-check   # the full release gate
make docker-up
~~~

The release path covers source compilation, regression tests, browser and runtime checks, OCR and provider contracts, mask authority, geometry and revision safety, long-image behaviour, and the connected product flow: **processed page → OCR/text object → translation → render → chapter ZIP**. Production ONNX binaries are not in Git, so model-dependent checks run separately.

To clean a whole real chapter with the real models and see the speed and the text left behind, run the **Chapter run** workflow from the Actions tab. To run A.I mode on a chapter and get every page image and its cost, run **A.I cost**. Both save their results to the `audit-evidence` branch, with a side-by-side image per text block.

## Project layout

~~~text
app/
  detector/       text detection (Kiuyha) and letter masks
  downloader/     HTTP, Playwright, adapters and slicing
  inpaint/        LaMa, flat fill and mask geometry safety
  ocr/            MangaOCR + PP-OCRv6
  translation/    text and two-image vision translation
  ai_mode/        the A.I checkpoints, glossary and Jev pass
  render/         typography, font catalog and lettering fonts
  visual_qc/      Gemini and DeepSeek image helpers used by A.I mode
  routers/        FastAPI API surface
  static/         browser workbench
  pipeline*.py    chapter processing and runtime coordination
gateway/          Manga Cloud: accounts, wallet, metered A.I upstream
tests/            correctness and release regressions
models/           local model artifacts
data/             runtime chapter data
docs/             engineering documentation
scripts/          release, browser and benchmark tooling
~~~

## Architecture

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

For implementation details see [ARCHITECTURE.md](ARCHITECTURE.md).
