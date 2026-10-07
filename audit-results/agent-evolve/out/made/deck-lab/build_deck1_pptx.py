"""Build 'The Deep Work Playbook' deck (7 slides, 16:9, 13.333x7.5 in)."""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn

NAVY = RGBColor(0x1B, 0x3A, 0x5C)
BLUE = RGBColor(0x4A, 0x90, 0xD9)
ORANGE = RGBColor(0xE8, 0x79, 0x2B)
INK = RGBColor(0x2B, 0x2B, 0x2B)
GRAY = RGBColor(0x6E, 0x7B, 0x8A)
LIGHT = RGBColor(0xF2, 0xF5, 0xF9)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BAR_COLORS = [NAVY, BLUE, RGBColor(0x8F, 0xB4, 0xDD), ORANGE, GRAY]

FONT = "DejaVu Sans"
SW, SH = Inches(13.333), Inches(7.5)
ML = Inches(0.7)          # left margin
CW = SW - 2 * ML          # content width


def new_pres():
    prs = Presentation()
    prs.slide_width, prs.slide_height = SW, SH
    return prs


def add_slide(prs, bg=WHITE):
    s = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = bg
    return s


def box(slide, x, y, w, h, fill=None, line=None):
    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    shp.shadow.inherit = False
    if fill is None:
        shp.fill.background()
    else:
        shp.fill.solid()
        shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
        shp.line.width = Pt(1)
    return shp


def text(slide, x, y, w, h, s, size, color=INK, bold=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, spacing=1.15, space_after=0):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    lines = s.split("\n")
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = spacing
        if space_after:
            p.space_after = Pt(space_after)
        r = p.add_run()
        r.text = line
        f = r.font
        f.size = Pt(size)
        f.bold = bold
        f.name = FONT
        f.color.rgb = color
    return tb


def footer(slide, num, title="", light=False):
    text(slide, ML, SH - Inches(0.42), CW, Inches(0.3), f"{title}   ·   {num}".replace("   ·", " ·").strip(),
         11, BLUE if light else GRAY, align=PP_ALIGN.RIGHT)


def title_block(slide, kicker, headline):
    text(slide, ML, Inches(0.45), CW, Inches(0.3), kicker.upper(), 11, ORANGE, bold=True)
    text(slide, ML, Inches(0.75), CW, Inches(0.75), headline, 26, NAVY, bold=True)
    box(slide, ML, Inches(1.5), Inches(1.2), Pt(3), fill=ORANGE)


# --- Slide 1: title ---
s = add_slide(prs := new_pres(), bg=NAVY)
box(s, Inches(0.7), Inches(2.5), Inches(0.6), Pt(4), fill=ORANGE)
text(s, Inches(0.7), Inches(2.8), Inches(12), Inches(1.3), "The Deep Work Playbook",
     40, WHITE, bold=True)
text(s, Inches(0.7), Inches(4.0), Inches(11), Inches(0.6),
     "Why focused attention is the superpower of the 21st century", 18, BLUE)
footer(s, 1, "", light=True)

# --- Slide 2: agenda ---
s = add_slide(prs)
title_block(s, "Agenda", "Today's path in four stops")
items = [
    ("1", "The four rules", "The system that makes deep work possible"),
    ("2", "The attention economy", "Why your focus was engineered away"),
    ("3", "The data", "What focus costs and what it returns"),
    ("4", "The start", "Your next three focused hours"),
]
y = Inches(2.0)
for num, head, sub in items:
    text(s, ML, y, Inches(0.7), Inches(0.7), num, 28, ORANGE, bold=True)
    text(s, ML + Inches(0.8), y + Inches(0.03), Inches(4.5), Inches(0.5), head, 20, NAVY, bold=True)
    text(s, ML + Inches(0.8), y + Inches(0.48), Inches(8.5), Inches(0.4), sub, 14, GRAY)
    y += Inches(1.15)

# --- Slide 3: four rules (cards) ---
s = add_slide(prs)
title_block(s, "The system", "Four rules that make deep work a habit")
cards = [
    ("Rule 1", "Work deeply", "Schedule it, defend it, count it."),
    ("Rule 2", "Embrace boredom", "Boredom is the gym for focus."),
    ("Rule 3", "Ruthless scheduling", "Know exactly where time goes."),
    ("Rule 4", "Drain the shallows", "Batch the shallow, protect the deep."),
]
cw = (CW - 3 * Inches(0.3)) / 4
for i, (tag, head, sub) in enumerate(cards):
    x = ML + i * (cw + Inches(0.3))
    box(s, x, Inches(2.0), cw, Inches(3.6), fill=LIGHT)
    box(s, x, Inches(2.0), cw, Inches(0.12), fill=ORANGE if i % 2 else BLUE)
    text(s, x + Inches(0.25), Inches(2.35), cw - Inches(0.5), Inches(0.4), tag.upper(), 11, GRAY, bold=True)
    text(s, x + Inches(0.25), Inches(2.75), cw - Inches(0.5), Inches(0.8), head, 18, NAVY, bold=True)
    text(s, x + Inches(0.25), Inches(3.7), cw - Inches(0.5), Inches(1.6), sub, 13, INK, spacing=1.2)

# --- Slide 4: attention economy (two columns) ---
s = add_slide(prs)
title_block(s, "The attention economy", "Your focus was engineered away on purpose")
box(s, ML, Inches(2.0), CW, Inches(2.2), fill=LIGHT)
text(s, ML + Inches(0.35), Inches(2.25), CW - Inches(0.7), Inches(0.4), "DESIGNERS OF DEPENDENCE", 11, BLUE, bold=True)
text(s, ML + Inches(0.35), Inches(2.65), CW - Inches(0.7), Inches(1.4),
     "Every feed is a slot machine: a variable reward pulls your thumb, your thumb pulls your focus.\n"
     "The product is your attention. The competitor is everything you were going to do.",
     15, INK, spacing=1.3)
text(s, ML + Inches(0.35), Inches(4.45), CW - Inches(0.7), Inches(0.4), "WHY WILLS POWER LOSES", 11, BLUE, bold=True)
text(s, ML + Inches(0.35), Inches(4.85), CW - Inches(0.7), Inches(1.4),
     "Decision fatigue is real: willpower is a budget, not a character trait.\n"
     "Design the environment instead of the intent - friction is a tool.",
     15, INK, spacing=1.3)

# --- Slide 5: cost of distraction (big number + stat cards) ---
s = add_slide(prs)
title_block(s, "The cost", "Distraction is expensive - and it compounds")
text(s, ML, Inches(1.9), Inches(4.2), Inches(1.4), "23 min", 64, ORANGE, bold=True)
text(s, ML + Inches(4.5), Inches(2.55), Inches(4.0), Inches(1.2),
     "average time to return after an interruption\n(Lab of Glennon Doyle-Murthy, 2013)", 12, GRAY, spacing=1.25)
stats = [
    ("40%", "drop in productivity when work is fragmented"),
    ("64%", "of knowledge workers call focus 'extremely important'"),
    ("0.4x", "value of shallow work vs. deep work per hour"),
]
sw_ = (CW - 2 * Inches(0.3)) / 3
for i, (n, d) in enumerate(stats):
    x = ML + i * (sw_ + Inches(0.3))
    box(s, x, Inches(4.2), sw_, Inches(2.0), fill=LIGHT)
    text(s, x + Inches(0.25), Inches(4.45), sw_ - Inches(0.5), Inches(0.7), n, 34, NAVY, bold=True)
    text(s, x + Inches(0.25), Inches(5.25), sw_ - Inches(0.5), Inches(0.9), d, 12, INK, spacing=1.2)

# --- Slide 6: productivity data (hand-drawn bar chart) ---
s = add_slide(prs)
title_block(s, "The payoff", "Focus pays back: deep vs. shallow output per hour")
chart_x, chart_y = ML + Inches(0.2), Inches(2.2)
chart_w, chart_h = Inches(7.4), Inches(3.0)
base = chart_y + chart_h - Inches(0.5)
data = [("Deep block", 0.78), ("Deep + shallow", 0.6), ("Shallow day", 0.35), ("Meetings + chat", 0.2), ("After 3rd interrupt", 0.12)]
max_v = 1.0
bw = Inches(0.9)
gap = (chart_w - 5 * bw) / 4
for i, (label, v) in enumerate(data):
    x = chart_x + i * (bw + gap)
    h = int(chart_h * v)
    box(s, x, base - h, bw, h, fill=BAR_COLORS[i])
    text(s, x - Inches(0.15), base + Inches(0.05), bw + Inches(0.3), Inches(0.5), label, 11, GRAY, align=PP_ALIGN.CENTER, spacing=1.0)
    text(s, x - Inches(0.15), base - h - Inches(0.35), bw + Inches(0.3), Inches(0.3),
         f"{int(v * 100)}%", 12, NAVY, bold=True, align=PP_ALIGN.CENTER)
box(s, chart_x, base + Inches(0.55), chart_w, Pt(1), fill=GRAY)
text(s, chart_x, base + Inches(0.55), chart_w, Inches(0.4), "relative output per hour, 1.0 = a full deep block", 11, GRAY)
ix = ML + Inches(8.0)
text(s, ix, Inches(2.2), Inches(4.6), Inches(0.4), "TWO WAYS TO READ IT", 11, BLUE, bold=True)
text(s, ix, Inches(2.65), Inches(4.6), Inches(1.2),
     "Every hour of shallow work is a 70% discount on the same hour of deep work.",
     18, NAVY, bold=True, spacing=1.25)
text(s, ix, Inches(4.0), Inches(4.6), Inches(1.6),
     "Interruptions are the worst offender: after a third interruption, the session returns only 12% of a deep block. Protect the last hour of the day.",
     15, INK, spacing=1.3)

# --- Slide 7: takeaway / start here ---
s = add_slide(prs, bg=NAVY)
box(s, Inches(0.7), Inches(1.6), Inches(0.6), Pt(4), fill=ORANGE)
text(s, Inches(0.7), Inches(2.0), Inches(11.9), Inches(1.2),
     "Start with the smallest unit you can defend", 34, WHITE, bold=True, spacing=1.2)
text(s, Inches(0.7), Inches(3.5), Inches(7.0), Inches(2.4),
     "1. Pick one deep goal for the next 90 days\n"
     "2. Block 4 hours, two days a week\n"
     "3. Switch the phone off, not on silent",
     20, BLUE, spacing=1.6)
text(s, Inches(0.7), Inches(5.6), Inches(11.9), Inches(0.5),
     "Deep work is a craft: rehearse it the same way you rehearse any skill - deliberately, daily, visibly.",
     14, WHITE)
footer(s, 7, "", light=True)

prs.save("deck-lab/deck1_deep_work.pptx")
print("saved deck-lab/deck1_deep_work.pptx")
