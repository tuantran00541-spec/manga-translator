# Deck 1: "How Code Reviews Actually Work" — light theme, 16:9, 7 slides.
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# ---- palette / metrics (learnings, refined by the checker) ----
INK = RGBColor(0x1F, 0x24, 0x30)     # near-black text on light
ACCENT = RGBColor(0x0F, 0x62, 0xFE)  # blue
PANEL = RGBColor(0xE8, 0xEE, 0xF9)   # light blue panel
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GRAY = RGBColor(0x5B, 0x64, 0x74)
BG = RGBColor(0xF7, 0xF9, 0xFC)

FONT = "Segoe UI"
SLIDE_W, SLIDE_H = Inches(13.333), Inches(7.5)
MARGIN = Inches(0.6)


def blank_slide(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = BG
    return s


def rect(s, x, y, w, h, color, radius=False):
    shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    sh = s.shapes.add_shape(shape_type, x, y, w, h)
    sh.fill.solid()
    sh.fill.fore_color.rgb = color
    sh.line.fill.background()
    sh.shadow.inherit = False
    return sh


def text(s, x, y, w, h, runs, size=18, color=INK, bold=False,
         align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, line_spacing=1.0,
         space_after=6, box=None):
    """runs: list of paragraphs; each is a str or a list of (text, color, bold)."""
    tb = s.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
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
    if box:
        rect(s, box[0], box[1], box[2], box[3], WHITE)
    return tb


def header(s, kicker, title):
    text(s, MARGIN, Inches(0.42), SLIDE_W - 2 * MARGIN, Inches(0.3),
         [kicker.upper()], size=13, color=ACCENT, bold=True, space_after=0)
    text(s, MARGIN, Inches(0.72), SLIDE_W - 2 * MARGIN, Inches(0.75),
         [title], size=32, bold=True, space_after=0)
    rect(s, MARGIN, Inches(1.45), Inches(0.55), Inches(0.055), ACCENT)


prs = Presentation()
prs.slide_width = SLIDE_W
prs.slide_height = SLIDE_H

# ---- 1: title ----
s = blank_slide(prs)
rect(s, 0, 0, SLIDE_W, SLIDE_H, WHITE)
rect(s, 0, 0, Inches(0.18), SLIDE_H, ACCENT)
text(s, MARGIN + Inches(0.4), Inches(2.3), Inches(11), Inches(0.4),
     ["A DECK-LAB STUDY IN 7 SLIDES"], size=14, color=ACCENT, bold=True, space_after=0)
text(s, MARGIN + Inches(0.4), Inches(2.7), Inches(11), Inches(2.1),
     ["How Code Reviews\nActually Work"], size=54, bold=True, space_after=0, line_spacing=1.05)
text(s, MARGIN + Inches(0.4), Inches(4.9), Inches(10), Inches(0.9),
     ["What a good review checks, how to comment without friction, and how to\nkeep the process fast enough that it never blocks the merge."],
     size=18, color=GRAY, space_after=0, line_spacing=1.1)

# ---- 2: agenda ----
s = blank_slide(prs)
header(s, "Agenda", "Five ideas, three minutes each")
items = [
    ("1", "The goal is risk reduction", "A review is a second set of eyes before production — not a style audit."),
    ("2", "Before you request it", "Small diffs, good titles, a checklist you already ran yourself."),
    ("3", "What a good comment looks like", "Concrete, actionable, kind; tied to a line and a reason."),
    ("4", "The author's side", "How to receive feedback and when to push back."),
    ("5", "Keeping the pipeline fast", "SLAs, async notes, and when a review is not needed."),
]
y = Inches(1.85)
for n, t, d in items:
    rect(s, MARGIN, y, Inches(12.13), Inches(0.92), WHITE, radius=True)
    text(s, MARGIN + Inches(0.25), y + Inches(0.16), Inches(0.5), Inches(0.6),
         [n], size=22, color=ACCENT, bold=True, space_after=0)
    text(s, MARGIN + Inches(0.85), y + Inches(0.12), Inches(3.6), Inches(0.7),
         [t], size=17, bold=True, space_after=0)
    text(s, MARGIN + Inches(4.6), y + Inches(0.12), Inches(7.3), Inches(0.7),
         [d], size=15, color=GRAY, space_after=0, line_spacing=1.05)
    y += Inches(1.08)

# ---- 3: goal ----
s = blank_slide(prs)
header(s, "Idea 1 — The goal", "Risk reduction, not nitpicking")
text(s, MARGIN, Inches(1.75), Inches(12.1), Inches(0.8),
     ["Every review question reduces the chance something ships broken:\n"
      "a subtle race, a wrong default, a missing edge case in the parser."],
     size=17, color=GRAY, space_after=0, line_spacing=1.15)
cards = [
    ("Correctness", "Does the code do what the title promises? Trace the new path yourself before commenting."),
    ("Tests", "Is the new behavior covered? A bug fix without a failing test is a re-shipping waiting to happen."),
    ("Longevity", "Will this confuse the next person at 2 a.m.? Optimize for the reader, not the author."),
]
x = MARGIN
for t, d in cards:
    rect(s, x, Inches(3.1), Inches(3.8), Inches(2.4), PANEL, radius=True)
    text(s, x + Inches(0.3), Inches(3.35), Inches(3.2), Inches(0.5), [t], size=20, bold=True, space_after=0)
    text(s, x + Inches(0.3), Inches(3.95), Inches(3.2), Inches(1.4), [d], size=15, color=GRAY,
         space_after=0, line_spacing=1.15)
    x += Inches(4.17)

# ---- 4: before requesting ----
s = blank_slide(prs)
header(s, "Idea 2 — Before you request", "Half the work happens before /review")
checks = [
    "Diff under ~400 changed lines; split the work if it is not.",
    "Title says the outcome, not the task: “Fix invoice rounding” not “math changes”.",
    "You already ran the tests, the linter, and read your own diff once.",
    "Left no “TODO: fix later” — finish the small thing or move it to an issue.",
]
y = Inches(1.85)
for c in checks:
    rect(s, MARGIN, y, Inches(12.13), Inches(0.78), WHITE, radius=True)
    rect(s, MARGIN + Inches(0.2), y + Inches(0.19), Inches(0.4), Inches(0.4), ACCENT, radius=True)
    text(s, MARGIN + Inches(0.25), y + Inches(0.2), Inches(0.35), Inches(0.4),
         ["✓"], size=16, color=WHITE, bold=True, space_after=0)
    text(s, MARGIN + Inches(0.85), y + Inches(0.16), Inches(11), Inches(0.5),
         [c], size=16, space_after=0)
    y += Inches(0.94)
text(s, MARGIN, Inches(5.9), Inches(12.1), Inches(0.5),
     ["Reviewers can only help with problems they can see in the diff."],
     size=15, color=GRAY, space_after=0)

# ---- 5: good comments ----
s = blank_slide(prs)
header(s, "Idea 3 — Comments", "Concrete, actionable, kind")
ex = [
    ("Bad",  "This is wrong.", GRAY),
    ("Bad",  "Why did you do it like this? Maybe just refactor everything.", GRAY),
    ("Good", "L42: the loop skips i = 0, so the first invoice is never billed. Add a test that bills order 1? (bug)", ACCENT),
    ("Good", "Consider `Decimal` here — floats make 0.1 + 0.2 ≠ 0.3. Happy to split it into a follow-up if that is a lot of work. (suggestion)", ACCENT),
]
y = Inches(1.85)
for tag, c, col in ex:
    rect(s, MARGIN, y, Inches(12.13), Inches(0.95), WHITE if col == GRAY else PANEL, radius=True)
    text(s, MARGIN + Inches(0.25), y + Inches(0.1), Inches(0.9), Inches(0.4),
         [tag], size=13, color=col, bold=True, space_after=0)
    text(s, MARGIN + Inches(1.15), y + Inches(0.2), Inches(10.7), Inches(0.6),
         [c], size=15, color=INK, space_after=0, line_spacing=1.05)
    y += Inches(1.08)

# ---- 6: author's side ----
s = blank_slide(prs)
header(s, "Idea 4 — The author's side", "Receiving feedback well")
text(s, MARGIN, Inches(1.8), Inches(5.8), Inches(4.6),
     ["Default to “thank you”",
      "Most feedback is noise; the useful 20% still saves time. Reply to every point, even “LGTM”.",
      "Push back with evidence",
      "“I measured: the cache hit is 40%, so the extra call is fine here” beats “works on my machine”.",
      "Fix it or explain it",
      "Unaddressed comments stall the merge more than a short written answer."],
     size=16, space_after=14, line_spacing=1.15)
rect(s, Inches(6.9), Inches(1.8), Inches(5.83), Inches(4.4), ACCENT, radius=True)
text(s, Inches(7.3), Inches(2.1), Inches(5.0), Inches(3.8),
     ["RULE OF THUMB", "If you cannot say it in two sentences, it does not belong in the thread.",
      "", "A 200-line comment becomes a new document — link it instead."],
     size=17, color=WHITE, bold=True, space_after=12, line_spacing=1.2)

# ---- 7: fast pipeline + takeaway ----
s = blank_slide(prs)
header(s, "Idea 5 — Speed", "Reviews that do not block the merge")
stats = [("24 h", "request-to-first-reply SLA"), ("3", "reviewers max; two is faster"),
         ("5 min", "per comment, or skip it"), ("100%", "of hotfixes may skip review")]
x = MARGIN
for v, l in stats:
    rect(s, x, Inches(1.85), Inches(2.85), Inches(1.5), WHITE, radius=True)
    text(s, x + Inches(0.25), Inches(2.05), Inches(2.4), Inches(0.6),
         [v], size=28, color=ACCENT, bold=True, space_after=0)
    text(s, x + Inches(0.25), Inches(2.65), Inches(2.4), Inches(0.6),
         [l], size=13, color=GRAY, space_after=0, line_spacing=1.05)
    x += Inches(3.13)
text(s, MARGIN, Inches(4.0), Inches(12.13), Inches(2.4),
     ["Takeaway", "A code review is a risk-reduction tool, not a gate.",
      "Small diffs, prepared authors, kind concrete comments, and a reply SLA keep it fast — and that is the point."],
     size=16, space_after=10, line_spacing=1.15)

import os
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deck1.pptx")
prs.save(out)
print(out, len(prs.slides._sldIdLst), "slides")
