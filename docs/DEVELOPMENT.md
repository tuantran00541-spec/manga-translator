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

The release path covers source compilation, regression tests, browser and runtime checks (including `scripts/release_sanity.py`, `scripts/browser_sanity.py`, `node --check`, `scripts/studio_runtime_sanity.js`, `scripts/theme_runtime_sanity.js`, `scripts/process_responsiveness_sanity.py`, and `scripts/inpaint_authority_sanity.py`), ONNX/OCR/provider contracts, mask authority, geometry and revision safety, long-image behaviour, and the connected product flow: **processed page → OCR/text object → translation → render → chapter ZIP**.

The repository ships the Kiuyha ONNX detectors under `models/`; LaMa and `ctd_seg.onnx` are downloaded/built outside Git and model-dependent checks run separately. To dispatch raw Kiuyha detection runs over RawKuma chapters with evidence pushed to `audit-evidence`, use **Kiuyha RawKuma detect** (`workflow_dispatch` on `.github/workflows/kiuyha-rawkuma-detect.yml` when batch helper scripts are present on the target branch).

## Project layout

~~~text
app/
  agent/          Chaos local coding agent (sessions, sandbox, MCP, skills)
  detector/       text detection (Kiuyha) and letter masks
  downloader/     HTTP (IP-pinned redirects), Playwright, adapters and slicing
  inpaint/        LaMa, flat fill and mask geometry safety
  ocr/            MangaOCR + PP-OCRv6
  translation/    text and two-image vision translation
  ai_mode/        A.I checkpoints, chapter queue, glossary and translation polish
  render/         typography, font catalog and lettering fonts
  visual_qc/      Gemini and DeepSeek image helpers used by A.I mode
  routers/        FastAPI API surface (15 routers incl. api_proxy, agent, extension)
  static/         browser workbench (4 CSS + 34 JS incl. i18n switcher & proxy monitor)
  api_proxy.py    API error-rescue proxy and learned auto-wrap rules
  model_contracts.py  ONNX input/output contract validators (LaMa & Kiuyha)
  pipeline*.py    chapter processing and runtime coordination
extension/        Chrome/Edge unpacked extension for browser-side chapter capture
gateway/          Manga Cloud: accounts, wallet, metered A.I upstream
tests/            correctness and release regressions
models/           local model artifacts (Kiuyha ONNX weights + LaMa/CTD downloads)
data/             runtime chapter data
docs/             engineering documentation
scripts/          release, browser and benchmark tooling
~~~

## Architecture

~~~text
                    Browser Workbench (VI / EN)
                          │
                       FastAPI
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
   Processing         Editorial      AI · Proxy · Agent
    Pipeline            State          Providers
        │                 │                 │
 Detect · OCR ·      Text · Geometry   Translate · Rescue Proxy
 Inpaint · Render    · Typography      · Queue · Chaos Sandbox
        └─────────────────┼─────────────────┘
                          ▼
                  Revision-safe artifacts
                          │
                          ▼
                   Stitched ZIP export
~~~

For implementation details see [ARCHITECTURE.md](ARCHITECTURE.md).
