# Features

What the app does, how well, and where it still needs a person.

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

## Feature list

### Import and slicing

- PNG, JPEG, WEBP and BMP; ZIP and CBZ chapters
- The browser extension: open the chapter in your own Chrome or Edge, click once, and its images go into the app. Sites behind Cloudflare that block Playwright work, because your browser has already loaded the pages.
- Chapter URLs through HTTP and Playwright: relative URLs, srcset, lazy-load attributes and scroll-based discovery, with site adapters where a fast path is stable
- Bounded downloads and upload validation
- Long pages are cut in low-content bands, never at a fixed height. Source-page identity and stitch ownership are recorded, so the chapter is rebuilt with no duplicated or missing pixels.

### Detection

The Kiuyha ONNX text detector (`models/kiuyha-tuned-1023.onnx`) finds the text blocks of a slice. Each block gets an erase mask:

- letters read by the mask head of comic-text-detector (`models/ctd_seg.onnx`), at full and half size
- a thin outline round every letter, even one the same colour as the background
- glow and shadow: smooth pixels unlike the colours round the block, grown out from the letters
- art touching the letters stays: growth that runs off the block is dropped
- a box holding two separate captions becomes two texts
- text across a slice seam is detected once on a strip over the cut and joins the block it belongs to

Detection records keep their source model and review state. [detection.md](detection.md) explains how and why.

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

The renderer bundles **71 comic fonts** (69 under OFL-1.1, 40 covering Vietnamese) across dialogue, emphasis, thought, narration, skill, SFX, horror and romance. A.I mode letters each text in the font for its role and container (speech, narration, shouts, screens, SFX); every font choice is validated against the installed catalog rather than accepted as a file path. Wrapped lines are balanced so a line never leaves one word alone. See [comic-fonts.md](comic-fonts.md).

### AI providers

Built in: Google Gemini, DeepSeek, OpenAI and OpenRouter. The settings layer also registers custom **OpenAI-compatible** providers on validated HTTPS API bases. Each provider advertises its capabilities separately, so one registry serves model discovery and A.I mode.

### Revision-safe render and export

Rendered pages and chapter exports are tied to the canonical editorial state. If the editor changes relevant content while a render or export runs, the stale artifact is rejected instead of becoming the published result. Long webtoon slices are stitched back into their source pages on export.

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
