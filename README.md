# Manga & Webtoon Translator Studio

[![Release Gate](https://github.com/tuantran00541-spec/manga-translator/actions/workflows/release-gate.yml/badge.svg)](https://github.com/tuantran00541-spec/manga-translator/actions/workflows/release-gate.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/backend-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![CPU First](https://img.shields.io/badge/runtime-CPU--first-222222)](#cpu-first-design)

**Translate a manga, manhwa or webtoon chapter on your own computer, with no GPU.**

Give it a chapter URL or a folder of images. It erases the original lettering, translates it and letters the translation back, and leaves every step open to the editor: nothing becomes the final chapter until you say so.

**Import → Slice → Detect → Clean → Review → OCR → Translate → Letter → Render → Export**

<p align="center">
  <img src="docs/images/demo-raw.jpg" alt="Original English page with two speech balloons" width="32%">
  <img src="docs/images/demo-clean.jpg" alt="The same page after cleanup, balloons empty" width="32%">
  <img src="docs/images/demo-translated.jpg" alt="The page translated to Vietnamese by A.I mode" width="32%">
</p>

<p align="center"><sub>RAW → CLEAN → translated to Vietnamese by A.I mode, cropped from page 3 of <a href="https://www.peppercarrot.com/en/webcomic/ep38_The-Healer.html">Pepper&amp;Carrot episode 38, “The Healer”</a> by David Revoy, <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>. The lettering was erased and the translation lettered by this app.</sub></p>

> **Automation proposes. Editorial state decides. Published artifacts must match the current state.**

## What it does well

### It erases text without wrecking the art

Cleanup runs on a mask of the letters themselves, not on detector rectangles. A comic-text-detector mask head reads the letters; the app adds a thin outline round every letter, then grows into the glow and shadow round it. Art that touches the letters stays.

Two whole webtoon chapters on a 4-core GitHub runner, real models, 2 page workers (evidence on the `audit-evidence` branch):

| Chapter | Slices | Text blocks | Time | Text left |
| --- | --- | --- | --- | --- |
| The World After the End, ch. 254 | 68 | 167 | 418 s | **0** |
| Youngest Scion of the Mages, ch. 137 | 52 | 145 | 332 s | **0** |

Times move with the runner's CPU: the first chapter took 418 s and 446 s on two different CPU models.

On labelled synthetic text, the mask erases 73.7 % of what has to go at 73.1 % precision; the hand-tuned rules it replaced managed 65.4 % at 43.1 %.

### The same image always gives the same result

A page cleaned twice gives the same pixels, on any CPU: the sampled slices of a whole chapter came out pixel-identical on two different CPU models. The one source of run-to-run drift, a random seed in the background-colour clustering, is fixed.

### It is fast where it matters

- A 2.4 M-parameter detector finds the text of a slice in a few packed passes (halves side by side, near-native bands, page-scale columns for big captions), on the OpenVINO runtime when available.
- The letter-mask model runs about 40× faster than the released one with identical masks.
- LaMa fills each region in one downscaled pass (512 px long side), and only where a fill is needed: a hole ringed by one colour, of any shade, is filled flat with no model run.

### The whole chapter runs on a laptop

Everything on the cleanup path is a small ONNX model on the CPU. Nothing needs a GPU or a cloud account; OCR and AI translation are optional layers around that.

## Two ways to work

**By hand.** Import a chapter and clean it, then correct, OCR, translate and letter it in the browser workbench. Every step can be redone.

**A.I mode.** Paste a chapter URL with an AI key (or use Manga Cloud) and pick the target language: Vietnamese, English, Indonesian, Spanish, Portuguese, French or German. Four checkpoints run the chapter, and the lettered result opens in the same editor:

1. **Scan** the raw slices: skip credit and textless slices, keep series logos untouched.
2. **Clean** the text (Kiuyha + LaMa). Meanwhile the chapter's names and terms are read into a glossary.
3. **Review** each raw and clean slice pair: erase missed text, repaint leftovers, restore art erased by mistake.
4. **Translate** each slice from its raw image with the glossary, naming each text's role and container; the font follows them, so one kind of text always looks the same.

An optional **Jev** pass grades every translated line and rewrites the weak ones.

Across five chapters of five series, A.I mode cost about **$0.07** of model calls per chapter.

## The workflow

| Stage | What happens |
| --- | --- |
| **1. Import** | Images, ZIP/CBZ archives, or chapter URLs |
| **2. Slice** | Long webtoon pages become processing-friendly slices |
| **3. Detect** | Text blocks and a letter mask for each (letters, outlines, glow) |
| **4. Clean** | LaMa inpainting on letter masks, leftover passes, manual repair |
| **5. Review** | Unified canvas workspace and editorial corrections |
| **6. OCR** | MangaOCR + PP-OCRv6 hybrid recognition |
| **7. Translate** | Batch text translation or two-image vision translation |
| **8. Letter** | Text objects, geometry, typography, fonts by text role |
| **9. Render** | Revision-safe page publication |
| **10. Export** | Stitched chapter ZIP |

## Features

### Import and slicing

- PNG, JPEG, WEBP and BMP; ZIP and CBZ chapters
- The browser extension: open the chapter in your own Chrome or Edge, click once, and its images go into the app. Sites behind Cloudflare that block Playwright work, because your browser has already loaded the pages.
- Chapter URLs through HTTP and Playwright: relative URLs, srcset, lazy-load attributes and scroll-based discovery, with site adapters where a fast path is stable
- Bounded downloads and upload validation
- Long pages are cut in low-content bands, never at a fixed height. Source-page identity and stitch ownership are recorded, so the chapter is rebuilt with no duplicated or missing pixels.

### Detection

The Kiuyha ONNX text detector (`models/kiuyha_text_1280.onnx`) finds the text blocks of a slice. Each block gets an erase mask:

- letters read by the mask head of comic-text-detector (`models/ctd_seg.onnx`), at full and half size
- a thin outline round every letter, even one the same colour as the background
- glow and shadow: smooth pixels unlike the colours round the block, grown out from the letters
- art touching the letters stays: growth that runs off the block is dropped
- a box holding two separate captions becomes two texts
- text across a slice seam is detected once on a strip over the cut and joins the block it belongs to

Detection records keep their source model and review state. [docs/detection.md](docs/detection.md) explains how and why.

### Artwork-safe cleanup

- LaMa inpainting with a zeroed hole, and nearby text not erased yet hidden from its context
- leftover passes only where the letter model still reads text after cleanup
- flat fill instead of LaMa where a hole is ringed by one colour
- dynamic and fixed LaMa backends, with tiling for the fixed 512 model
- manual repaint, mask reset, and preserve/skip regions
- a restore brush that puts the original page back where the cleanup went too far; the page keeps it through later repaints

Uncertain detections, watermarks and review-only regions are not silently promoted to destructive cleanup.

### OCR

- **PP-OCRv6 / PaddleOCR** for Japanese, Chinese, Korean and English. On 110 Black Jack text boxes it missed 3.2% of the dialogue characters, against 13% for MangaOCR, which the app used before.

OCR is chapter-aware and revision-safe, and keeps the source lettering's colour, position and size as hints for typesetting.

### Translation

**Text translation.** Batch existing text objects through configured OpenAI-compatible providers, with stable IDs, stale-write protection and budget controls where pricing is known.

**Vision translation.** Send the **ORIGINAL** and **CLEAN** versions of a slice together with the existing text-object IDs and their regions. The model translates only objects that already exist; geometry, placement, colour and size stay editor-owned.

### Lettering and fonts

The browser workbench edits translated text, font, size, weight, stroke, background, alignment, region geometry and manual text objects in one Review workspace.

- **Undo and redo** (Ctrl+Z, Ctrl+Y) for text edits, moves, resizes, styles, new and deleted text objects, and whole find-and-replace or preset runs.
- **Keyboard proofreading**: A/D move between slices, Alt+Up/Down between text objects, Enter edits the translation, Ctrl+Enter saves and goes to the next one, Esc returns to the page; `?` lists every shortcut.
- **Proofreading panel** (Ctrl+F, Ctrl+H): every source line and translation of the chapter in reading order, editable in place, with find, replace-all and an untranslated-only filter.
- **Style presets**: save a text object's look as "Dialogue", "Narration" or "SFX", apply it with keys 1-9 or to many checked rows at once. Presets are kept in `data/style_presets.json`.

The renderer bundles **71 comic fonts** (69 under OFL-1.1, 40 covering Vietnamese) across dialogue, emphasis, thought, narration, skill, SFX, horror and romance. A.I mode letters each text in the font for its role and container (speech, narration, shouts, screens, SFX); every font choice is validated against the installed catalog rather than accepted as a file path. Wrapped lines are balanced so a line never leaves one word alone. See [docs/comic-fonts.md](docs/comic-fonts.md).

### AI providers

Built in: Google Gemini, DeepSeek, OpenAI and OpenRouter. The settings layer also registers custom **OpenAI-compatible** providers on validated HTTPS API bases. Each provider advertises its capabilities separately, so one registry serves model discovery and A.I mode.

### Agent mode

The **Agent** screen is a coding agent like Claude Code or Codex on any provider from Settings, including a custom OpenAI-compatible base URL. Models that refuse the native tools field fall back to tool calls written as text, and rate-limited calls wait and retry.

- **Tools:** list, read, search and glob files; write, edit, or change many files at once with Codex's `apply_patch` format; run commands; read web pages; keep a plan (`todo_write`); hand a job to a helper agent (`task`).
- **Sandbox:** commands run read-only, workspace-write (the default) or with full access, network off unless allowed. Linux enforces it with Landlock and macOS with Seatbelt; Windows has no sandbox, so there commands always wait for approval unless you choose to allow everything. A command may ask to run outside the sandbox, which always needs your approval.
- **Approvals:** ask before every change, edit and run sandboxed commands freely, or do everything.
- **Project context:** reads `AGENTS.md` and `CLAUDE.md` from the workspace and your home folder.
- **Skills:** [Agent Skills](https://agentskills.io/specification) folders in `.agents/skills`, `.claude/skills` or `.codex/skills` of the workspace or your home. Only their names and descriptions are sent until the agent loads one.
- **Bundled skills:** 18 ready ones ship with the app and load on demand: `systematic-debugging`, `verification-before-completion`, `test-driven-development`, `writing-plans`, `receiving-code-review`, `requesting-code-review`, `dispatching-parallel-agents`, `grilling`, `diagnosing-bugs`, `research`, `codebase-design`, `prototype`, `pr`, `frontend-design`, `webapp-testing`, `mcp-builder`, plus the manual `/handoff` and `/improve-codebase-architecture`. They come from obra/superpowers, mattpocock/skills and anthropics/skills (sources and licenses in `app/agent/builtin_skills/NOTICE.md`). `/skills add owner/repo[@ref][/folder]` installs more from any public GitHub repository into `~/.manga-agent/skills`; read what you install, since a skill is instructions the agent follows.
- **MCP:** stdio and streamable HTTP servers from the workspace `.mcp.json`, `~/.manga-agent/mcp.json`, `~/.claude.json` and `~/.codex/config.toml`. A server the workspace declares starts only after you allow it, and the trust is pinned to its exact config.
- **Hooks and commands:** `PreToolUse` and `PostToolUse` hooks from `.agents/settings.json` or Claude Code's `.claude/settings.json` (workspace hooks need your trust; a hook exiting 2 blocks the call). Custom slash commands come from `.agents/commands`, `.claude/commands` and `~/.codex/prompts`. Built-ins are `/help`, `/compact`, `/init`, `/skills`, `/mcp`, `/model`, `/mode`, `/sandbox`, `/plan`, `/goal`, `/undo`, `/memory`, `/agents`, `/rules` and `/clear`.
- **Precise edits:** `read_file` with `anchors=true` shows lines as `N#hh|text` and `edit_lines` changes them by anchor, so weak models never retype code; a stale anchor is refused. Edits also report Python and JSON syntax errors at once (idea from oh-my-pi).
- **Permission rules:** `allow`, `ask` or `deny` by glob for commands, paths, URLs, agents and MCP tools, in `.agents/settings.json` or `~/.manga-agent/settings.json` (`permission`) or Claude Code's `permissions` lists; the last matching rule wins and `.env` files are denied by default. A project can only tighten rules, never allow. Three identical calls in a row are blocked (opencode).
- **Undo:** `/undo` puts back the files the last turn changed with the edit tools, not what a shell command changed (Hermes, Kimi).
- **Agents:** `explore`, `plan` and `coder` plus your own Markdown files with front matter in `.agents/agents` or `.claude/agents` (`tools`, `disallowedTools`, `model`). `task` runs one helper and waits; `spawn_agent`, `wait_agent`, `send_input` and `close_agent` (Codex's tools) run up to 6 at once, refuse the same job twice, and a helper that finishes reports to the model by itself in a `<subagent_notification>`. A helper's edits go through your approvals (Kimi, Muse, Codex).
- **While it works:** a message you send mid-run joins the next step; `@path` attaches a file or folder; `ask_user` lets the agent ask you with answer buttons (pi, Hermes, Kimi).
- **Plan and goal:** `/plan` keeps the agent read-only until you approve its plan; `/goal` keeps it working across turns until it calls `goal_done` (Kimi).
- **Memory:** the agent keeps lasting notes per project and per user in `~/.manga-agent/memory`; see and edit them with `/memory` (Hermes).
- **Tools for work beyond the project:** `web_search` (Tavily or Brave when `TAVILY_API_KEY` or `BRAVE_API_KEY` is set, else DuckDuckGo), `web_fetch`, `web_download` (up to 50 MB into the workspace), and background jobs: `run_command` with `background` returns a job id, `job_output` reads what it printed and `job_stop` ends it (a server needs the network switch on to listen). A turn may take up to 300 steps and helpers 100; `max_steps` and `token_budget` (10M) in `profile.json` change that.
- **Smooth tool use:** replies stream live and read-only calls in one reply (reads, searches, `symbols`, web fetches, helpers) run in parallel while edits stay in order. The status line shows cache hit rate, cost (where a price is known) and model versus tool seconds. `symbols` outlines a file or finds a definition by name; `view_image` shows an image to models that can see (set `vision` in `profile.json` to force it); with more than 15 connected MCP tools they are found with `tool_search` instead of all being listed.
- **Safer edits:** `write_file` refuses to overwrite a file the agent never read, and any edit refuses a file that changed after it was read. A second helper that edits at the same time works in a private copy of the project and is merged back file by file; a file both sides changed keeps the project's version and the helper's goes to `.agent-conflicts/`. Set `isolate_writers` to `false` to queue editing helpers instead.
- **Approvals that learn:** commands that only look (`ls`, `git status`, `grep`...) never ask, "Luôn cho phép" saves a command prefix or a web host to `~/.manga-agent/settings.json` (never for risky commands), and the review mode lets a second model (`review_model`, else the session's) clear calls that match what you asked and ask you only when unsure.
- **Without the page:** `python -m app.agent.cli exec "task" --base URL --model M` (key in `AGENT_API_KEY`) runs one task, `resume ID "more"` continues it, and `--json` prints one event per line.
- **Long runs:** compaction cuts inside one long turn (keeps the last ~60k characters, summarises the rest in Goal/Progress/Decisions/Next-steps form and carries the lists of files read and changed from one summary to the next); a context-length error from the provider triggers a summary and a retry; timeouts, dropped connections and 5xx are retried with a jittered, growing pause; the token budget ignores cached tokens and ends with a report instead of silence.
- **Borrowed from Codex, pi, omp and DeepSeek-TUI:** review mode can also deny (with a reason and a "no workarounds" reply to the model) and stops the turn after 3 denials in a row or 10 in 50; `fan_out` runs up to 12 helper jobs, six at a time, and returns every report in one call; stream rules (`stream_rules` in `profile.json`, regex plus a reminder) stop a reply midway, drop it and ask again, and a built-in rule does that for a tool call too long to survive the output limit; a command that changes `.git` hooks or config has them put back; `/receipts` lists every file changed, command run and approval of the session.
- **Deeper ports:** the system prompt is frozen at the session's start so the provider cache keeps hitting; later changes (memory, todos, mode) arrive as a `<context_update>` message and `/cache` says which part of the prefix changed and when (DeepSeek-TUI). `edit_file` takes several edits in one call and forgives smart quotes, Unicode forms, trailing spaces and indentation, and names the closest text when nothing matches (pi). `/goal` continues with Codex's template: the objective, a check for no progress and the budget used. During a goal (or with `notes_context` in `profile.json`) `context_notes` keeps a notebook and `new_context` starts a fresh window from it, the last request and a short tail, archiving the old history (omp, Codex). `run_script` runs a Python script in the sandbox whose `tools.read_file(...)`, `tools.search(...)`, `tools.glob(...)` and other read-only tools are plain functions, so one step can do dozens of lookups (Codex code mode, omp eval); at most 200 calls and 600 s per script.
- **Plugins and profiles (the DeepSeek Harness way):** the pieces are swappable. A `profile.json` in `~/.manga-agent` (or `.agents` in a project, which may only switch things off) turns whole feature groups off with `"disable"`: `files`, `edit`, `shell`, `web`, `skills`, `mcp`, `todo`, `memory`, `ask_user`, `plan`, `goal`, `subagents`, `external`. A plugin is a Python file in `~/.manga-agent/plugins` (or `.agents/plugins`, which runs only after you trust its exact contents) with `register(api)` that can add tools (`api.tool`), slash commands, `pre_tool`/`post_tool` hooks, system-prompt text and a whole replacement step loop (`api.loop`, chosen by `"loop"` in the profile). `delegate` hands a standalone job to Codex (`codex exec`) or Claude Code (`claude -p`) when they are installed, or to any command listed under `external_agents`; it always asks first. For DeepSeek models the current turn's reasoning is sent back with tool results, as its thinking mode requires. `/plugins` shows what is loaded.
- **Built from the research:** old tool outputs are dropped in batches while the last 12 stay (as good as summaries at half the cost, per JetBrains' Complexity Trap); a command output over 12k characters is saved to a file the agent reads in pieces; `AGENTS.md` is read only up to 8k characters and `/init` asks for a short one, since long context files lowered success in the Evaluating AGENTS.md study; at most three skills load per task; an agent that stops after editing without running anything is nudged once to check; helpers that edit files run one after the other (a second one waits in line, it is not refused); and after the agent reads a web page or an MCP result, every edit, command or network call asks you first (turn off with `"untrusted_guard": false` in `profile.json`). `profile.json` also takes `"subagent_model"` and `"compact_model"` to run helpers and summaries on a cheaper model.
- **What keeps the model inside its job:** on Linux the kernel enforces it for every command (Landlock and seccomp): writes only in the workspace and temp, no network sockets at all when the network is off (TCP, UDP and others), no signals to processes outside, credential folders such as `~/.ssh`, `~/.aws` and browser profiles unreadable, no API keys in the environment, output capped at 8 MB, files at 1 GiB, and anything a command leaves running is killed with it. The server itself is marked undumpable so a command cannot read its memory or environment. Leaving the sandbox (`outside_sandbox`) and `delegate` always ask, even in Auto mode. Without Auto, a web page is fetched freely only from hosts you named or approved, and `rm -r`, `git push`, `git reset --hard`, `sudo`, `kill` and similar ask first. `.env` and `.git` are protected from the edit tools, and a workspace cannot be `/`, your home folder or a system folder. A turn stops at 3M tokens (`token_budget`). macOS gets the write, network and credential rules through Seatbelt (not tested here); Windows has no sandbox, so commands always ask. Still open: a command can read everything outside the credential list, plugins and MCP servers run with your full rights, and old kernels (Landlock before ABI 6) cannot block signals.
- **Long sessions:** conversations past about 200k characters are summarised by the model; sessions are saved under `data/agent` and can be reopened.

Agent routes answer only this machine, need an `X-Manga-Agent` header, and switch off with `MANGA_AGENT_MODE=0`.

### Revision-safe render and export

Rendered pages and chapter exports are tied to the canonical editorial state. If the editor changes relevant content while a render or export runs, the stale artifact is rejected instead of becoming the published result. Long webtoon slices are stitched back into their source pages on export.

## Quick start

### Requirements

- Python 3.12 (the installer brings its own)
- A CPU-capable machine, with about 5 GB free for the first install
- Chromium for Playwright URL ingestion and browser checks
- More RAM helps OCR and very large pages

The image path accepts up to **100,000,000 decoded pixels** per image.

### Install

**Windows:** download [MangaTranslator-Setup.exe](https://github.com/tuantran00541-spec/manga-translator/releases/latest/download/MangaTranslator-Setup.exe) and double-click it. It needs no admin rights; the first install takes 5–15 minutes and uses about 1.6 GB of disk. Uninstall it from Windows Settings → Apps; it asks whether to keep your translated chapters.

The setup is not code-signed yet, so Windows SmartScreen may show "Windows protected your PC". Click **More info → Run anyway** (Vietnamese Windows: **Thêm thông tin → Vẫn chạy**). The one-command install below does not go through SmartScreen.

Or with one command, nothing else to install first:

~~~powershell
# Windows: paste into PowerShell or cmd
powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/tuantran00541-spec/manga-translator/main/install.ps1 | iex"
~~~

~~~bash
# Linux / macOS
curl -LsSf https://raw.githubusercontent.com/tuantran00541-spec/manga-translator/main/install.sh | sh
~~~

On Windows, double-click the **Manga Translator** icon the installer puts on the desktop and in the Start menu: the app opens in its own window, with no console. Elsewhere, or from any **new** cmd or terminal window:

~~~bash
manga           # starts the app and opens it in the browser (or just opens it if it already runs)
manga --window  # the same in its own window (Windows and macOS)
manga update    # gets the latest version; models, chapters and settings stay
manga --version
~~~

The installer works in one folder (`%LOCALAPPDATA%\manga-translator` on Windows, `~/.local/share/manga-translator` elsewhere):

- downloads [uv](https://github.com/astral-sh/uv) (pinned, hash checked), which fetches Python 3.12 and installs the dependencies with CPU-only PyTorch;
- on Windows, installs the Microsoft Visual C++ runtime when it is missing or older than 14.40 (onnxruntime and PaddleOCR need it; Windows asks for admin once). `manga` checks it at every start, so a runtime removed later is put back instead of failing with WinError 126;
- downloads the LaMa model (hash checked, resumes broken downloads) and builds `ctd_seg.onnx`;
- adds a `manga` command to the user PATH and, on Windows, the desktop and Start menu icons.

A Windows user folder with accents (for example `C:\Users\Nguyễn`) breaks PaddleOCR, so the installer then uses `C:\ProgramData\manga-translator`. From a clone, `install.bat` / `./install.sh` install that clone in place.

To uninstall, delete that folder, the icons and the `manga` command (`%LOCALAPPDATA%\manga-translator\bin` on Windows, `~/.local/bin/manga` elsewhere), then remove its line from the user Path or shell start-up file.

Manual install (Python 3.10–3.12):

~~~bash
python -m venv .venv
# Linux / macOS: source .venv/bin/activate    Windows: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
playwright install chromium
~~~

### Models

Place these files in `models/`:

| File | Purpose |
| --- | --- |
| `kiuyha_text_1280.onnx` | Text detection: Kiuyha/Manga-Bubble-YOLO boxes (in the repository) |
| `ctd_seg.onnx` | Letter masks: comic-text-detector, exported fast and mask-only |
| `lama-manga-dynamic.onnx` | Preferred inpainting backend |
| `lama.onnx` | Fixed-resolution fallback |

The LaMa files are too large for Git and must be downloaded. Build `ctd_seg.onnx` from the released [comictextdetector.pt.onnx](https://github.com/zyddnys/manga-image-translator/releases/tag/beta-0.3): save it as `.cache/comictextdetector.pt.onnx` and run `python scripts/export_ctd_onnx.py` (needs `torch`, `onnx` and `onnx2torch` once; the app itself only needs onnxruntime). The **Kiuyha ONNX export** workflow re-exports [Kiuyha/Manga-Bubble-YOLO](https://huggingface.co/Kiuyha/Manga-Bubble-YOLO) and checks the ONNX boxes against the original model.

### Run

~~~bash
manga             # after the installer
python run.py     # from an activated environment
~~~

Open `http://127.0.0.1:8000`.

### Browser extension

Sites behind Cloudflare usually refuse Playwright. The extension in [`extension/`](extension) takes the images from the chapter you have open in Chrome or Edge instead:

1. Open `chrome://extensions` (or `edge://extensions`) and turn on **Developer mode**.
2. Click **Load unpacked** and pick the `extension` folder: in the app folder for a source checkout, or `%LOCALAPPDATA%\manga-translator\app\extension` after the Windows setup.
3. With the app running, open a chapter, then click the Manga Translator icon and **Gửi chương vào app**.

It scrolls the page so lazy images load, takes the wide images and canvases in reading order, converts formats the app does not read (GIF, AVIF) to PNG, and opens the new chapter in the app. It only talks to the app on this computer.

### Docker

~~~bash
docker compose up --build
~~~

The image expects the model files through the mounted `./models` directory.

## AI setup

Provider keys are set in the settings UI or through environment variables:

~~~text
GEMINI_API_KEY
GOOGLE_API_KEY
DEEPSEEK_API_KEY
OPENAI_API_KEY
OPENROUTER_API_KEY
EXPLABS_API_KEY
~~~

The settings UI discovers a provider's models through `/models`, selects exact model IDs, and registers custom OpenAI-compatible providers. Custom endpoints must be public HTTPS URLs, and credentials are never accepted inside the API URL.

### Manga Cloud (experimental, off by default)

Every feature, A.I mode included, is free with your own A.I key. Manga Cloud is for people without one: with `MANGA_TIERS=1` the app offers a Manga Cloud provider served by the gateway at `MANGA_CLOUD_URL`. Users sign in from the A.I mode panel with their email and a 6-digit code and top up a prepaid balance. Each A.I call of a chapter is charged at its real cost plus a 5 % fee (`GATEWAY_FEE_PERCENT`). A long webtoon chapter costs about $0.30, so a chapter needs about $0.32 of balance to start. Payment-provider fees are added to what the user pays, so the balance always gets the full top-up. The gateway stops a chapter at what the balance can pay for, and at $2 whatever the balance.

The gateway lives in `gateway/` and runs with `python -m gateway`. [docs/DEPLOY.md](docs/DEPLOY.md) puts it online behind Caddy with HTTPS and lists its limits and admin commands.

| Area | Variables |
| --- | --- |
| A.I upstream | `GATEWAY_UPSTREAM_BASE`, `GATEWAY_UPSTREAM_KEY`, `GATEWAY_UPSTREAM_MODEL`, `GATEWAY_PRICE_INPUT_PER_M`, `GATEWAY_PRICE_OUTPUT_PER_M`; for Azure OpenAI also `GATEWAY_UPSTREAM_AUTH=api-key` and `GATEWAY_UPSTREAM_MAX_TOKENS_FIELD=max_completion_tokens` |
| Login email | `GATEWAY_RESEND_API_KEY`, `GATEWAY_MAIL_FROM`, `GATEWAY_DEV_LOGIN=1` (shows the code instead of mailing it, local testing only) |
| Wallet | `GATEWAY_FEE_PERCENT` (5), `GATEWAY_CHAPTER_ESTIMATE_USD` (0.30), `GATEWAY_VND_PER_USD` (26000), `GATEWAY_TOPUPS_VND`, `GATEWAY_TOPUPS_USD` |
| payOS (VietQR) | `GATEWAY_PAYOS_CLIENT_ID`, `GATEWAY_PAYOS_API_KEY`, `GATEWAY_PAYOS_CHECKSUM_KEY`, `GATEWAY_PAYOS_FEE_PERCENT` (0); webhook `/v1/billing/payos/webhook` |
| Lemon Squeezy (cards) | `GATEWAY_LS_API_KEY`, `GATEWAY_LS_STORE_ID`, `GATEWAY_LS_VARIANT`, `GATEWAY_LS_WEBHOOK_SECRET`, `GATEWAY_LS_FEE_PERCENT` (5), `GATEWAY_LS_FEE_FIXED_USD` (0.50); webhook `/v1/billing/lemonsqueezy/webhook` |
| Other | `GATEWAY_DB`, `GATEWAY_ADMIN_KEY`, `GATEWAY_RETURN_URL`, `GATEWAY_HOST`, `GATEWAY_PORT`, `GATEWAY_TRUSTED_PROXIES` |

Signed-in users see their balance and every top-up and chapter charge, and can sign out on every device or delete their account from the A.I mode panel. A payOS top-up is credited once, when its webhook reports the exact amount. A Lemon Squeezy top-up is a one-time order at a custom price on one product variant; a refund takes the credit back.

## Runtime settings

Detection, inpainting, OCR and rendering thresholds are fixed constants in `app/parameters.py`. Only operational settings read environment variables:

| Area | Variables |
| --- | --- |
| Processing | `MANGA_PIPELINE_DEFAULT_WORKERS`, `MANGA_PIPELINE_SLICE_WORKER_LIMIT`, `MANGA_USE_DYNAMIC_LAMA`, `MANGA_INPAINT_PRELOAD`, `MANGA_FIXED_LAMA_*` |
| ONNX Runtime | `MANGA_ORT_PROVIDER`, `MANGA_ORT_REQUIRE_PROVIDER`, `MANGA_ORT_INTRA_OP_THREADS`, `MANGA_ORT_OPENVINO_*`, `MANGA_ORT_CPU_MEM_ARENA`, `MANGA_ORT_MEM_PATTERN`, `MANGA_ORT_SERIALIZE_INFERENCE` |
| OCR | `MANGA_PPOCRV6_TIER`, `MANGA_PPOCRV6_TEXTLINE_ORIENTATION`, `MANGA_OCR_TARGET_SELECTION`, `MANGA_OCR_IMAGE_CACHE_MB`, `MANGA_OCR_JOB_CONCURRENCY_LIMIT`, `MANGA_OCR_JOB_ACTIVE_LIMIT` |
| Agent | `MANGA_AGENT_MODE=0` turns the Agent screen's routes off |
| Access | `MANGA_ALLOWED_HOSTS`: extra host names the app answers to, comma separated (loopback always works; add the machine's LAN name or IP to open it from another device) |
| Network and AI | `MANGA_DOWNLOAD_WORKERS`, `MANGA_DOWNLOAD_JS_NAVIGATION_TIMEOUT_MS`, `MANGA_REMOTE_CONNECT_TIMEOUT_SECONDS`, `MANGA_TRANSLATION_CONNECT_TIMEOUT_SECONDS`, `MANGA_TRANSLATION_READ_TIMEOUT_SECONDS` |

## CPU-first design

The supported baseline is CPU execution, tuned for bounded desktop-class workloads instead of a GPU:

- two page workers by default, small enough that heavy inference kernels do not fight for the same memory bandwidth
- conservative ONNX Runtime threading
- OpenVINO for the text detector on x86-64 Windows and Linux; standard ONNX Runtime elsewhere and for the letter mask and LaMa, whose input sizes vary
- decoded-page caching for repeated OCR work
- bounded downloads with transient retries
- short manifest lock windows

## Safety by design

Safety is part of the processing model, not only of deployment. The app protects:

- remote URL imports against SSRF and unsafe address ranges
- managed files against path escape
- uploads and archives against oversized payloads
- decoded images against excessive pixel counts
- browser requests against unsafe remote targets
- custom AI endpoints against invalid or untrusted URLs
- rendered and exported artifacts against stale editor state

A deployment that faces the network still needs normal firewall and authentication controls.

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

For implementation details see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Text detection](docs/detection.md)
- [Comic font library](docs/comic-fonts.md)
- [Deploying the gateway](docs/DEPLOY.md)
- [UI guidelines](docs/UI_GUIDELINES.md)
- [Security history](docs/SECURITY_HISTORY.md)

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

## Current limitations

The project speeds up chapter production; it does not replace professional redraw and lettering.

- **Stylised sound effects are mostly left as art**: the letter model reads story lettering, not drawn SFX, and Korean SFX and props text can stay in a machine-only run.
- **A glowing title can leave a faint smear** where the glow was erased.
- **Text painted into the art** (signs, screens) may be translated as plain text and lettered over it.
- **Vietnamese compound words** can split across two lines.
- **Watermarks and site banners** are only partly erased.
- On large manga pages with heavy black art, the cleanup can take a stroke of art that the detector boxed as text; the restore brush puts it back.
- **Western comic pages are not the target**: the tool is tuned for webtoons. Small balloons that hug their text can be erased whole, and the text of balloons standing close together is joined into one block, so its translation runs over the art.
- Perspective or warped lettering, curved text, hand-drawn SFX recreation, ambiguous OCR, heavily protected reader sites and typography that needs artistic judgment still need a person.

The strongest use is a **reviewable production workstation with automation**, not a black-box batch converter.

## Philosophy

Detection, cleanup, OCR, translation, typography and AI help are all allowed to make mistakes. The editor can inspect, correct, reject, repaint, resize, retranslate or restyle any of them before anything becomes the published chapter.

> **The page belongs to the editor.**

## License

Copyright (C) 2026 tuantran00541-spec. The code is licensed under the [GNU Affero General Public License v3.0](LICENSE): you may use, change and share it, and anyone who runs a changed version as a network service must offer its source to that service's users.

Bundled third-party parts keep their own licenses:

- **Fonts:** SIL Open Font License, see `app/static/fonts/licenses/`.
- **Vietnamese word list:** MIT, see `app/render/vi_words.LICENSE.txt`.
- **Models:** downloaded at install time, under their own authors' terms.
