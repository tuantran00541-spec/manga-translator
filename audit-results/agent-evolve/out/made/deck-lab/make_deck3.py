"""Deck 3 (warm light theme): "Bugs We Could Have Prevented", 7 slides, 16:9.
Built strictly from .agents/skills/pptx/SKILL.md — warm-ink palette variant,
plus the add_fade_transitions.py post-step."""
import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# ---- warm light palette: cream bg, warm ink, burnt-orange accent, sand panels ----
INK    = RGBColor(0x2A, 0x23, 0x1C)   # near-black warm brown (12.9:1 on BG)
ACCENT = RGBColor(0xC4, 0x57, 0x1A)   # burnt orange (4.9:1 on BG, 4.4:1 on PANEL)
PANEL  = RGBColor(0xEF, 0xE6, 0xD8)   # sand
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
MUTED  = RGBColor(0x5C, 0x4F, 0x3E)   # warm gray (5.5:1 on BG)
BG     = RGBColor(0xFB, 0xF7, 0xEF)   # cream

FONT = "Segoe UI"
SLIDE_W, SLIDE_H = Inches(13.333), Inches(7.5)
MARGIN = Inches(0.6)
CONTENT_W = SLIDE_W - 2 * MARGIN


def blank_slide(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = BG
    return s


def rect(s, x, y, w, h, color, radius=False):
    st = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    sh = s.shapes.add_shape(st, x, y, w, h)
    sh.fill.solid()
    sh.fill.fore_color.rgb = color
    sh.line.fill.background()
    sh.shadow.inherit = False
    return sh


def text(s, x, y, w, h, runs, size=16, color=INK, bold=False,
         align=PP_ALIGN.LEFT, line_spacing=1.0, space_after=6):
    tb = s.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    first = True
    for para in runs:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.alignment = align
        p.line_spacing = line_spacing
        p.space_after = Pt(space_after)
        items = [(para, color, bold)] if isinstance(para, str) else para
        for t, c, b in items:
            r = p.add_run()
            r.text = t
            r.font.name = FONT
            r.font.size = Pt(size)
            r.font.bold = b
            r.font.color.rgb = c
    return tb


def header(s, kicker, title):
    text(s, MARGIN, Inches(0.42), CONTENT_W, Inches(0.3),
         [kicker.upper()], size=13, color=ACCENT, bold=True, space_after=0)
    text(s, MARGIN, Inches(0.72), CONTENT_W, Inches(0.75),
         [title], size=32, bold=True, space_after=0)
    rect(s, MARGIN, Inches(1.45), Inches(0.55), Inches(0.055), ACCENT)


def card_row(s, y, n, height, titles, bodies, title_size=20, body_size=14):
    gap = Inches(0.37)
    cw = CONTENT_W / n - gap if n > 1 else CONTENT_W
    for i in range(n):
        x = MARGIN + i * (cw + gap)
        rect(s, x, y, cw, height, PANEL, radius=True)
        text(s, x + Inches(0.3), y + Inches(0.25), cw - Inches(0.6), Inches(0.5),
             [titles[i]], size=title_size, bold=True, space_after=0)
        text(s, x + Inches(0.3), y + Inches(0.8), cw - Inches(0.6),
             height - Inches(1.0), [bodies[i]], size=body_size, color=MUTED,
             space_after=0, line_spacing=1.15)


prs = Presentation()
prs.slide_width = SLIDE_W
prs.slide_height = SLIDE_H

# ---- 1: title ----
s = blank_slide(prs)
rect(s, 0, 0, Inches(0.18), SLIDE_H, ACCENT)
text(s, MARGIN + Inches(0.4), Inches(2.5), CONTENT_W, Inches(0.4),
     ["A POSTMORTEM IN 7 SLIDES"], size=14, color=ACCENT, bold=True, space_after=0)
text(s, MARGIN + Inches(0.4), Inches(3.0), CONTENT_W, Inches(1.9),
     ["Bugs We Could\nHave Prevented"], size=50, bold=True, space_after=0, line_spacing=1.05)
text(s, MARGIN + Inches(0.4), Inches(4.9), Inches(11), Inches(0.9),
     ["Five production incidents, each one a single missing check — and the\nhabit that would have caught it before the deploy."],
     size=17, color=MUTED, space_after=0, line_spacing=1.15)

# ---- 2: cost of one hour ----
s = blank_slide(prs)
header(s, "Idea 1 — Cost", "One hour down, counted the honest way")
stats = [("3 h", "on-call engineer, half the night"), ("41×", "user-facing, per hour, in a support queue"),
         ("9", "minutes to rollback — we waited 2 hours"), ("$0", "paid by the test we did not write")]
x = MARGIN
for v, l in stats:
    rect(s, x, Inches(1.9), Inches(2.85), Inches(1.5), PANEL, radius=True)
    text(s, x + Inches(0.25), Inches(2.1), Inches(2.4), Inches(0.6),
         [v], size=28, color=ACCENT, bold=True, space_after=0)
    text(s, x + Inches(0.25), Inches(2.72), Inches(2.4), Inches(0.7),
         [l], size=13, color=MUTED, space_after=0, line_spacing=1.1)
    x += Inches(3.13)
text(s, MARGIN, Inches(4.1), CONTENT_W, Inches(2.2),
     ["The arithmetic", "Incident cost is not just the rollback: it is the on-call pager, the\n"
      "status page, the follow-up doc, and the next two weeks of extra caution. A single\n"
      "5-minute test usually costs less than the last of those line items."],
     size=15, space_after=10, line_spacing=1.2)

# ---- 3: incident 1 ----
s = blank_slide(prs)
header(s, "Idea 2 — The missing check", "Null that shipped in a release")
card_row(s, Inches(1.9), 3, Inches(2.4),
         ["What happened", "Where it hid", "The check"],
         ["A new field defaulted to None on one branch of the pricing code. No user hit that branch for weeks; the next deploy made it the main one.",
          "The unit test covered the happy path only; the branch was one `if not x` away.",
          "Property test: for 200 random price inputs, the output is never None. 30 lines."],
         body_size=14)
text(s, MARGIN, Inches(4.9), CONTENT_W, Inches(1.5),
     [[("Lesson: ", ACCENT, True),
       ("the branch that is rarely taken is the one that changes meaning —\n"
        "treat rare paths as the main test, not an exception.", INK, False)]],
     size=15, space_after=0, line_spacing=1.2)

# ---- 4: incident 2 ----
s = blank_slide(prs)
header(s, "Idea 3 — The missing check", "Float, money, rounding")
text(s, MARGIN, Inches(1.85), CONTENT_W, Inches(0.8),
     ["Invoice totals built on `float` arithmetic disagreed with the bank by 3 cents\n"
      "out of 100 — enough to fail the reconciliation report, not enough to alarm anyone."],
     size=16, color=MUTED, space_after=0, line_spacing=1.15)
card_row(s, Inches(3.0), 3, Inches(2.4),
         ["0.1 + 0.2", "Decimal(\"0.1\") + …", "Exactly 0.3"],
         bodies=["The Python repr that haunts every language’s money math.",
          "Decimal is the type, not the function — the sum is exactly 0.3.",
          "The diff is the bug; the test should assert on 0.3, not 0.30000000000000004."],
         body_size=14)
text(s, MARGIN, Inches(5.7), CONTENT_W, Inches(0.8),
     [[("Lesson: ", ACCENT, True), ("any code that touches money, or any value that must round cleanly,\n"
        "is a `Decimal` code path. No exceptions for speed — money bugs are not fast enough to skip." , INK, False)]],
     size=14, space_after=0, line_spacing=1.15)

# ---- 5: incident 3 ----
s = blank_slide(prs)
header(s, "Idea 4 — The missing check", "The race nobody saw")
card_row(s, Inches(1.9), 3, Inches(2.4),
         ["Symptom", "Root cause", "The check"],
         ["Two users, same invoice, both saw “updated”, one saw a $12 gap for a week.",
          "Read-modify-write with no row lock; fast users beat the slow commit.",
          "Integration test: two threads update the same row; assert one wins, one errors with 409."],
         body_size=14)
text(s, MARGIN, Inches(4.9), CONTENT_W, Inches(1.5),
     [[("Lesson: ", ACCENT, True), ("races are not caught in review — they are caught in integration tests\n"
        "that actually run two callers at the same time. “Should be fine in practice” is\n"
        "the exact phrase that precedes the next postmortem.", INK, False)]],
     size=15, space_after=0, line_spacing=1.2)

# ---- 6: the habits ----
s = blank_slide(prs)
header(s, "Idea 5 — The habits", "Four checks that cover most of these classes")
items = [
    ("Default to None", "Every new field gets a default; every code path is tested with the default present."),
    ("Money = Decimal", "No float in billing, pricing or aggregation paths. Lint rule: `float` in these directories."),
    ("Concurrency test", "Any endpoint that writes to a shared row gets a two-thread integration test in CI."),
    ("One rare branch", "The least-traveled path of any function gets its own named test, not a hidden `else`."),
]
y = Inches(1.9)
for t, d in items:
    rect(s, MARGIN, y, CONTENT_W, Inches(1.0), PANEL, radius=True)
    rect(s, MARGIN + Inches(0.2), y + Inches(0.28), Inches(0.44), Inches(0.44), ACCENT, radius=True)
    text(s, MARGIN + Inches(0.25), y + Inches(0.3), Inches(0.35), Inches(0.4),
         ["✓"], size=15, color=WHITE, bold=True, space_after=0)
    text(s, MARGIN + Inches(0.9), y + Inches(0.18), Inches(3.4), Inches(0.6),
         [t], size=16, bold=True, space_after=0)
    text(s, MARGIN + Inches(4.4), y + Inches(0.18), Inches(7.5), Inches(0.7),
         [d], size=14, color=MUTED, space_after=0, line_spacing=1.1)
    y += Inches(1.14)

# ---- 7: takeaway ----
s = blank_slide(prs)
header(s, "Takeaway", "The bug is cheaper to write than to find")
text(s, MARGIN, Inches(1.9), CONTENT_W, Inches(2.4),
     ["Most production incidents are preventable, cheaply:",
      "a default-value test, a Decimal swap, a two-thread test, one named branch test.",
      "None of them takes more than an hour. The postmortem always takes longer than the test did not."],
     size=17, space_after=12, line_spacing=1.3)
rect(s, MARGIN, Inches(4.6), CONTENT_W, Inches(1.6), ACCENT, radius=True)
text(s, MARGIN + Inches(0.3), Inches(4.85), CONTENT_W - Inches(0.6), Inches(1.1),
     [[("Write the test before the deploy. It is always cheaper than the rollback.", WHITE, True)]],
     size=20, space_after=0, line_spacing=1.15)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deck3.pptx")
prs.save(out)
print(out, len(prs.slides._sldIdLst), "slides")
