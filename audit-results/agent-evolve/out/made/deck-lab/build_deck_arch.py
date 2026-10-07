"""Manga Translator architecture deck."""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

SW_IN, SH_IN = 13.333, 7.5
ML = 0.7
CW = SW_IN - 2 * ML

INK = RGBColor(0x2B, 0x2B, 0x2B)
MUTED = RGBColor(0x6E, 0x7B, 0x8A)
NAVY = RGBColor(0x1B, 0x3A, 0x5C)
ACCENT = RGBColor(0x4A, 0x90, 0xD9)
ORANGE = RGBColor(0xE8, 0x79, 0x2B)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LGTINT = RGBColor(0x8F, 0xB4, 0xDD)
CARD_BG = RGBColor(0xED, 0xF1, 0xF7)
LINE = RGBColor(0xC9, 0xD4, 0xE0)

FONT = "DejaVu Sans"


def _tb(slide, x, y, w, h, text, size=15, color=INK, bold=False, align=PP_ALIGN.LEFT, spacing=1.0):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.font.name = FONT
    p.alignment = align
    p.line_spacing = spacing
    return box


def _rect(slide, x, y, w, h, fill, line=None, line_w=1.0):
    s = slide.shapes.add_shape(1, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(line_w)
    s.shadow.inherit = False
    return s


def _content_slide(prs, kicker, title):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _tb(slide, ML, 0.45, CW, 0.4, kicker.upper(), 11, ACCENT, True)
    _tb(slide, ML, 0.78, CW, 0.6, title, 26, NAVY, True)
    _rect(slide, ML, 1.45, 1.2, 0.03, ORANGE)
    return slide


def _card(slide, x, y, w, h, tag, body, tag_color=ACCENT, body_size=12, card_h=None):
    ch = card_h if card_h is not None else h - 0.15
    _rect(slide, x, y, w, ch, CARD_BG, line=LINE)
    _rect(slide, x, y, w, 0.07, tag_color)
    _tb(slide, x + 0.18, y + 0.16, w - 0.36, 0.45, tag, 13, NAVY, True)
    _tb(slide, x + 0.18, y + 0.58, w - 0.36, ch - 0.5, body, body_size, INK, spacing=1.05)


def _footer(slide, dark=False):
    c = LGTINT if dark else MUTED
    _tb(slide, SW_IN - 3.5, 6.85, 2.5, 0.6, "Manga Translator", 11, c, align=PP_ALIGN.RIGHT)


def build(path):
    prs = Presentation()
    prs.slide_width = int(SW_IN * 914400)
    prs.slide_height = int(SH_IN * 914400)

    # ---- 1. Title (dark) ----
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _rect(s, 0, 0, SW_IN, SH_IN, NAVY)
    _tb(s, ML, 2.0, CW, 0.4, "ARCHITECTURE BRIEFING", 13, ORANGE, True)
    _tb(s, ML, 2.4, CW, 1.5, "Manga Translator —", 34, WHITE, True)
    _tb(s, ML, 3.1, CW, 0.6, "architecture & module map", 34, WHITE, True)
    _rect(s, ML, 3.8, 2.2, 0.03, ORANGE)
    _tb(s, ML, 4.05, CW, 1.2, "FastAPI core · ONNX AI pipeline · vanilla-JS SPA · embedded coding agent · cloud gateway", 14, LGTINT, spacing=1.3)
    _footer(s, dark=True)

    # ---- 2. Repo at a glance ----
    s = _content_slide(prs, "Overview", "Repo at a glance: 7 top-level areas")
    cards = [
        ("app/", "FastAPI core: 13 modules, 14 routers, ~30k LOC"),
        ("gateway/", "Cloud gateway: auth, billing, LLM token quota"),
        ("extension/", "Chrome extension: import chapters from browser"),
        ("tests/", "54 test files: OCR, inpaint, render, security"),
        ("deploy/ · Dockerfile", "Compose, Caddy, MSI installers, run.py"),
        ("docs/", "ARCHITECTURE.md — Mermaid flows, source of this deck"),
    ]
    w0 = 3.93
    for i, (t, b) in enumerate(cards):
        col, row = i % 3, i // 3
        x = ML + col * (w0 + 0.12)
        y = 1.85 + row * 2.05
        _card(s, x, y, w0, 1.95, t, b, card_h=1.9)
    _footer(s)

    # ---- 3. Layered architecture ----
    s = _content_slide(prs, "Layers", "Five layers, top to bottom")
    rows = [
        ("Frontend", "index.html + static/js: 35 modules, ~8k LOC; canvas review & text editor", ACCENT),
        ("API layer", "FastAPI: 14 routers; RequestSizeLimitMiddleware guards /api/*", ACCENT),
        ("Orchestration", "ChapterPipeline mixins; manifest locks; state snapshots; thread pools", ORANGE),
        ("AI engine", "ONNX: Kiuyha detect, LaMa inpaint, MangaOCR/PaddleOCR v6, vision QC", ORANGE),
        ("Data on disk", "data/raw → data/processed → data/output; models/ holds .onnx", MUTED),
    ]
    y = 1.95
    for i, (name, desc, col) in enumerate(rows):
        _rect(s, ML, y, CW, 0.78, CARD_BG, line=LINE)
        _rect(s, ML, y, 0.08, 0.78, col)
        _tb(s, ML + 0.35, y + 0.06, 2.5, 0.65, f"{i+1}. {name}", 14, NAVY, True)
        _tb(s, ML + 0.35, y + 0.4, CW - 0.7, 0.5, desc, 12, INK)
        y += 0.95
    _footer(s)

    # ---- 4. app/ module map ----
    s = _content_slide(prs, "Deep dive · app/", "Each module has one job")
    mods = [
        ("routers/ (14)", "chapters · editor · export · ocr · render · translation · agent · ai_mode — thin HTTP → pipeline"),
        ("pipeline.py (817 L)", "ChapterPipeline: download → detect → inpaint → manifest, mixes in two helper classes"),
        ("page_processing.py", "Per-page worker: read, detect, NMS, LaMa, atomic clean write, stale-state guard"),
        ("pipeline_editing.py (1k L)", "Review mutations: boxes, excluded regions, manual mask, text objects"),
        ("manifest_utils.py (790 L)", "manifest.json + filelocks, atomic os.replace, artifact transactions, snapshots"),
        ("security.py", "SSRF / path-traversal / size limits on every URL and upload"),
        ("ocr/ (13 files)", "MangaOCR (ja) · PaddleOCR v6 (en/zh) · Korean rec; quality + reading order"),
        ("detector/ · inpaint/", "KiuyhaTextDetector + BubbleBox/NMS; LaMa clusters, smart fill, feather"),
        ("render/", "Text objects: auto font-size, bg color, stroke, wrap, font catalog"),
        ("downloader/ (8)", "bs4 adapter, else Playwright JS; slicer splits webtoon strips"),
        ("agent/ (38 files)", "Embedded coding agent: tools, sandbox, MCP server, guardrails"),
        ("translation/", "LLM glossary translation; ai_mode/ runs the 5-stage pipeline"),
    ]
    w0 = 3.93
    for i, (t, b) in enumerate(mods):
        col, row = i % 3, i // 3
        x = ML + col * (w0 + 0.12)
        y = 1.85 + row * 1.32
        _card(s, x, y, w0, 1.28, t, b, body_size=11.5, card_h=1.24)
    _footer(s)

    # ---- 5. Main processing flow ----
    s = _content_slide(prs, "Deep dive · flow", "process_pages: one pass, eight steps")
    steps = [
        "POST /api/process_pages — validate id + indices",
        "manifest lock + capture state snapshot",
        "thread pool (2–8) over selected pages",
        "read_image with pixel limit",
        "Kiuyha detect → boxes, NMS IoU 0.35",
        "filter boxes inside excluded regions",
        "LaMa inpaint + manual mask clusters",
        "state unchanged? atomic write, invalidate render",
    ]
    w0 = 5.6
    for i, t in enumerate(steps):
        col, row = i % 2, i // 2
        x = ML + col * (w0 + 0.6)
        y = 1.95 + row * 1.15
        _rect(s, x, y, 0.5, 0.5, NAVY)
        _tb(s, x, y + 0.09, 0.5, 0.35, str(i + 1), 16, WHITE, True, PP_ALIGN.CENTER)
        _tb(s, x + 0.7, y + 0.02, w0 - 0.9, 1.05, t, 13, INK, spacing=1.1)
    _tb(s, ML, 6.1, 5.6, 0.8, "Stale state? → discard, manifest untouched", 13, ORANGE, True)
    _footer(s)

    # ---- 6. Concurrency & data safety ----
    s = _content_slide(prs, "Deep dive · safety", "No lost updates: locks + snapshots + atomic writes")
    items = [
        ("File locks", "manifest.lock (30 s), page_XXX.lock (60 s) serialize writers across requests"),
        ("Atomic replace", "tmp file → os.replace; a crash never corrupts manifest or clean images"),
        ("State snapshots", "capture inputs before work, diff after; changed pages are discarded"),
        ("Artifact transactions", "half-finished tmp files recovered on startup"),
        ("Limits up front", "upload ≤ 500 MB / 300 files; body cap; image pixel cap"),
    ]
    y = 1.95
    for t, b in items:
        _rect(s, ML, y, CW, 0.78, CARD_BG, line=LINE)
        _tb(s, ML + 0.3, y + 0.1, 2.3, 0.65, t, 14, NAVY, True)
        _tb(s, ML + 2.8, y + 0.08, CW - 3.1, 0.8, b, 12, INK, spacing=1.1)
        y += 0.93
    _footer(s)

    # ---- 7. Frontend module map ----
    s = _content_slide(prs, "Deep dive · frontend", "Vanilla JS, no framework: 35 modules, ~8k LOC")
    groups = [
        ("Shell", "main.js bootstrap · ui-shell · toast · theme · api.js (all /api calls)"),
        ("Chapter flow", "upload · preview (slice review) · page-navigator · source-language"),
        ("Review canvas", "review.js + review-stitch/: overlays, brush, magic wand, proofing"),
        ("Text editor", "editor.js + inspector, history, geometry, presets, persistence"),
        ("AI features", "ai-mode.js 5-stage pipeline · agent.js coding-agent UI · visual QC"),
        ("Load order", "index.html: toast → api → upload → preview → review → editor → main"),
    ]
    w0 = 3.93
    for i, (t, b) in enumerate(groups):
        col, row = i % 3, i // 3
        x = ML + col * (w0 + 0.12)
        y = 1.85 + row * 2.05
        _card(s, x, y, w0, 2.0, t, b, card_h=1.95)
    _footer(s)

    # ---- 8. Around the core ----
    s = _content_slide(prs, "Deep dive · around the core", "Three satellites around the FastAPI app")
    cards = [
        ("app/agent/ · 38 files", "Embedded coding agent: tools, session, sandbox, MCP server; UI via routers/agent.py", ORANGE),
        ("gateway/ · 6 files", "Standalone service: email login, token quota, per-model pricing; behind Caddy in deploy/", ACCENT),
        ("extension/ · 3 files", "Chrome MV3: popup scans manga pages, sends URLs to /api/chapter; background keeps session", ACCENT),
    ]
    w0 = 3.93
    for i, (t, b, c) in enumerate(cards):
        x = ML + i * (w0 + 0.12)
        _card(s, x, 1.95, w0, 2.6, t, b, tag_color=c, card_h=2.6)
    _tb(s, ML, 5.2, CW, 0.95, "Also: deploy/ (Docker Compose + Caddy) · installer/ (MSI/InnoSetup) · scripts/ (20+ ops & sanity checks) · 54 test files pin every contract above.", 12.5, INK, spacing=1.15)
    _footer(s)

    prs.save(path)


if __name__ == "__main__":
    build("deck-lab/deck_arch.pptx")
