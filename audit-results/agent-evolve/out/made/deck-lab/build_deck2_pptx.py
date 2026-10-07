"""Build 'Cities Run Out of Water' deck - built strictly from .agents/skills/pptx/SKILL.md.

Palette (fresh, not deck 1's): ink #1F2933, teal #0E7C7B, gold #D9A441, slate #6B7A8F, light #EEF3F6, white.
"""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

INK = RGBColor(0x1F, 0x29, 0x33)
TEAL = RGBColor(0x0E, 0x7C, 0x7B)
TEAL_LT = RGBColor(0x7F, 0xC8, 0xC4)
GOLD = RGBColor(0xD9, 0xA4, 0x41)
SLATE = RGBColor(0x6B, 0x7A, 0x8F)
LIGHT = RGBColor(0xEE, 0xF3, 0xF6)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = "DejaVu Sans"
SW, SH = Inches(13.333), Inches(7.5)
ML, CW = Inches(0.7), Inches(11.93)


def new_pres():
    prs = Presentation()
    prs.slide_width, prs.slide_height = SW, SH
    return prs


def add_slide(prs, bg=WHITE):
    s = prs.slides.add_slide(prs.slide_layouts[6])
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
    shp.line.fill.background()
    return shp


def text(slide, x, y, w, h, s, size, color=INK, bold=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, spacing=1.15):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, line in enumerate(s.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = spacing
        r = p.add_run()
        r.text = line
        f = r.font
        f.size, f.bold, f.name, f.color.rgb = Pt(size), bold, FONT, color
    return tb


def footer(slide, num, dark=False):
    text(slide, ML, SH - Inches(0.42), CW, Inches(0.3), f"Water Risk  ·  {num}",
         11, TEAL_LT if dark else SLATE, align=PP_ALIGN.RIGHT)


def head(slide, kicker, headline):
    text(slide, ML, Inches(0.45), CW, Inches(0.3), kicker.upper(), 11, GOLD, bold=True)
    text(slide, ML, Inches(0.75), CW, Inches(0.75), headline, 26, INK, bold=True)
    box(slide, ML, Inches(1.5), Inches(1.2), Pt(3), fill=TEAL)


prs = new_pres()

# 1 - title (dark teal bg)
s = add_slide(prs, bg=TEAL)
box(s, ML, Inches(2.3), Inches(0.6), Pt(4), fill=GOLD)
text(s, ML, Inches(2.6), Inches(12), Inches(1.4), "Cities Run Out of Water", 40, WHITE, bold=True)
text(s, ML, Inches(3.9), Inches(11), Inches(0.6),
     "The quiet infrastructure crisis behind every drought", 18, TEAL_LT)
footer(s, 1, dark=True)

# 2 - agenda
s = add_slide(prs)
head(s, "Agenda", "Four stops to understand water risk")
items = [
    ("1", "The demand wall", "Cities are outgrowing their aquifers"),
    ("2", "Where it leaks", "The three places water disappears"),
    ("3", "The numbers", "Supply, demand, loss - by the decade"),
    ("4", "What moves", "Five interventions that actually work"),
]
y = Inches(2.0)
for num, h, sub in items:
    text(s, ML, y, Inches(0.7), Inches(0.7), num, 28, GOLD, bold=True)
    text(s, ML + Inches(0.8), y + Inches(0.03), Inches(5.5), Inches(0.5), h, 20, INK, bold=True)
    text(s, ML + Inches(0.8), y + Inches(0.48), Inches(9), Inches(0.4), sub, 14, SLATE)
    y += Inches(1.15)
footer(s, 2)

# 3 - four cards
s = add_slide(prs)
head(s, "The demand wall", "Four forces pushing city water use upward")
cards = [
    ("F1", "Population", "Metro regions keep absorbing the country's growth."),
    ("F2", "Per-capita", "More showers, more lawns, more dishwashers each."),
    ("F3", "Heat", "Drier, hotter summers: irrigation and AC both drink."),
    ("F4", "Industry", "Data centers and fabs hide in 'light' sectors."),
]
cw = (CW - 3 * Inches(0.3)) / 4
for i, (tag, h, sub) in enumerate(cards):
    x = ML + i * (cw + Inches(0.3))
    box(s, x, Inches(2.0), cw, Inches(3.6), fill=LIGHT)
    box(s, x, Inches(2.0), cw, Inches(0.12), fill=TEAL)
    text(s, x + Inches(0.25), Inches(2.35), cw - Inches(0.5), Inches(0.4), tag, 11, SLATE, bold=True)
    text(s, x + Inches(0.25), Inches(2.75), cw - Inches(0.5), Inches(0.8), h, 18, INK, bold=True)
    text(s, x + Inches(0.25), Inches(3.7), cw - Inches(0.5), Inches(1.6), sub, 13, INK, spacing=1.2)
footer(s, 3)

# 4 - two column
s = add_slide(prs)
head(s, "Where it leaks", "Water vanishes at three points, not one")
box(s, ML, Inches(2.0), CW, Inches(2.1), fill=LIGHT)
text(s, ML + Inches(0.35), Inches(2.25), CW - Inches(0.7), Inches(0.4), "PIPE NETWORKS", 11, TEAL, bold=True)
text(s, ML + Inches(0.35), Inches(2.65), CW - Inches(0.7), Inches(1.3),
     "Aging mains are a running leak: many mid-sized cities lose 20-30% of pumped water before a customer sees it.\n"
     "Every unmarked break is a small emergency, forever.", 15, INK, spacing=1.3)
text(s, ML + Inches(0.35), Inches(4.35), CW - Inches(0.7), Inches(0.4), "DEMAND MISMATCH", 11, TEAL, bold=True)
text(s, ML + Inches(0.35), Inches(4.75), CW - Inches(0.7), Inches(1.3),
     "Cities plan for normal years and run out in dry ones.\n"
     "Reservoirs, aquifers and reuse plants are sized to averages, not to the tail.", 15, INK, spacing=1.3)
footer(s, 4)

# 5 - section divider (dark)
s = add_slide(prs, bg=TEAL)
text(s, ML, Inches(2.6), Inches(12), Inches(0.5), "SECTION 2", 13, GOLD, bold=True)
text(s, ML, Inches(3.1), Inches(12), Inches(1.0), "The numbers", 36, WHITE, bold=True)
text(s, ML, Inches(4.1), Inches(10), Inches(0.6), "Supply, demand and loss - by the decade", 16, TEAL_LT)
footer(s, 5, dark=True)

# 6 - chart slide (hand-drawn bars, single palette: teal=supply, slate=demand, gold=headline callout)
s = add_slide(prs)
head(s, "The numbers", "Supply falls, demand rises - the scissors gap")
chart_x, chart_y = ML + Inches(0.2), Inches(2.0)
chart_w, chart_h = Inches(7.0), Inches(3.2)
base = chart_y + chart_h - Inches(0.5)
data = [("2010 supply", 0.85, TEAL), ("2020 supply", 0.72, TEAL),
        ("2030 supply", 0.55, TEAL), ("2010 demand", 0.60, SLATE),
        ("2020 demand", 0.70, SLATE), ("2030 demand", 0.82, SLATE)]
bw = Inches(0.95)
gap = (chart_w - 6 * bw) / 5
for i, (label, v, col) in enumerate(data):
    x = chart_x + i * (bw + gap)
    h = int(chart_h * v)
    box(s, x, base - h, bw, h, fill=col)
    text(s, x - Inches(0.1), base + Inches(0.05), bw + Inches(0.2), Inches(0.5),
         label, 11, SLATE, align=PP_ALIGN.CENTER, spacing=1.0)
    text(s, x - Inches(0.1), base - h - Inches(0.35), bw + Inches(0.2), Inches(0.3),
         f"{int(v * 100)}%", 12, INK, bold=True, align=PP_ALIGN.CENTER)
box(s, chart_x, base + Inches(0.55), chart_w, Pt(1), fill=SLATE)
# inline legend so color is never the only cue
text(s, chart_x, base + Inches(0.85), chart_w, Inches(0.4),
     "supply (teal) / demand (slate), index, 100 = full aquifer", 11, SLATE, align=PP_ALIGN.CENTER)
ix = ML + Inches(7.9)
text(s, ix, Inches(2.3), Inches(4.4), Inches(0.4), "HOW TO READ", 11, TEAL, bold=True)
text(s, ix, Inches(2.75), Inches(4.4), Inches(1.3),
     "By 2030 the two lines cross: demand runs 27 points past what the aquifer can give.",
     17, INK, bold=True, spacing=1.25)
text(s, ix, Inches(4.2), Inches(4.4), Inches(1.6),
     "Where the falling supply curve meets the rising demand curve, the gap stops being a forecast.",
     14, INK, spacing=1.3)
footer(s, 6)

# 7 - stat row
s = add_slide(prs)
head(s, "What moves", "Five interventions, ranked by water saved per dollar")
stats = [
    ("$1", "price water so scarcity is felt, not hidden"),
    ("0.8%", "loss cut per year with leak-only programs"),
    ("5x", "return on reuse vs. building new intake"),
    ("30%", "of peak demand shiftable off-peak, free"),
]
cw2 = (CW - 3 * Inches(0.3)) / 4
for i, (n, d) in enumerate(stats):
    x = ML + i * (cw2 + Inches(0.3))
    box(s, x, Inches(2.0), cw2, Inches(2.6), fill=LIGHT)
    box(s, x, Inches(2.0), cw2, Inches(0.12), fill=GOLD if i == 2 else TEAL)
    text(s, x + Inches(0.25), Inches(2.45), cw2 - Inches(0.5), Inches(0.8), n, 30, TEAL, bold=True)
    text(s, x + Inches(0.25), Inches(3.35), cw2 - Inches(0.5), Inches(1.1), d, 13, INK, spacing=1.2)
text(s, ML, Inches(4.95), CW, Inches(0.5),
     "Ranked by water saved per dollar: pricing, leak repair, reuse, peak shifting.", 14, SLATE)
footer(s, 7)

# 8 - takeaway (dark)
s = add_slide(prs, bg=TEAL)
box(s, ML, Inches(1.7), Inches(0.6), Pt(4), fill=GOLD)
text(s, ML, Inches(2.1), Inches(11.9), Inches(1.2),
     "Water is a schedule, not a tap", 34, WHITE, bold=True, spacing=1.2)
text(s, ML, Inches(3.6), Inches(7.5), Inches(2.4),
     "1. Name the gap: supply vs. demand, by 2030\n"
     "2. Fund leaks first, reuse second, price third\n"
     "3. Shrink the peak before you buy a plant",
     20, TEAL_LT, spacing=1.6)
text(s, ML, Inches(5.7), Inches(11.9), Inches(0.5),
     "The city that measures its leak today prices its own water tomorrow.",
     14, WHITE)
footer(s, 8, dark=True)

prs.save("deck-lab/deck2_water.pptx")
print("saved deck-lab/deck2_water.pptx")
