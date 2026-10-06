"""Deck 2 (dark theme): "The Physics of Espresso", 7 slides, 16:9.
Built strictly from .agents/skills/pptx/SKILL.md — dark palette variant."""
import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# ---- dark palette (SKILL.md "Colors" section) ----
INK    = RGBColor(0xE8, 0xED, 0xF5)   # near-white body text on dark
ACCENT = RGBColor(0x4D, 0x9F, 0xFF)   # light blue accent for dark bg
PANEL  = RGBColor(0x1B, 0x24, 0x36)   # card fill on dark bg
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
MUTED  = RGBColor(0xA8, 0xB4, 0xC8)    # secondary text on dark (>=3:1 on bg)
BG     = RGBColor(0x0F, 0x14, 0x20)

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


def card_row(s, y, n, height, titles, bodies, title_size=20, body_size=15):
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
     ["DECK-LAB DARK-THEME STUDY"], size=14, color=ACCENT, bold=True, space_after=0)
text(s, MARGIN + Inches(0.4), Inches(3.0), CONTENT_W, Inches(1.9),
     ["The Physics of Espresso"], size=54, bold=True, space_after=0, line_spacing=1.05)
text(s, MARGIN + Inches(0.4), Inches(4.9), Inches(11), Inches(0.9),
     ["Six variables — grind, pressure, temperature, dose, yield, time — that turn\nroasted beans into a cup, and which one to change when the taste is off."],
     size=17, color=MUTED, space_after=0, line_spacing=1.15)

# ---- 2: what goes in ----
s = blank_slide(prs)
header(s, "The system", "Nine ingredients, one pressure pump")
card_row(s, Inches(1.9), 3, Inches(2.2),
         ["Water", "Beans", "Steam"],
         ["98–99 °C, low-TDS mineral water. Hotter scorch the cup; cooler under-extracts.",
          "Roasted 7–28 days, 18 g in the basket, ground fine and even.",
          "Crema is CO₂ and oil emulsified at 9 bar over ~25 s."])

# ---- 3: pressure ----
s = blank_slide(prs)
header(s, "Variable 1 — Pressure", "9 bar is a target, not a truth")
text(s, MARGIN, Inches(1.85), CONTENT_W, Inches(0.75),
     ["Static pressure in the group head does not equal pressure at the bed.\n"
      "Back-pressure from tamping and grind makes the real 3–4 bar on the coffee."],
     size=16, color=MUTED, space_after=0, line_spacing=1.15)
card_row(s, Inches(2.9), 3, Inches(2.4),
         ["Under 8 bar", "9 bar", "Over 11 bar"],
         ["Channeling: water finds the path of least resistance, extraction uneven.",
          "Standard machine rating; paired with 25 s, the sweet spot for most recipes.",
          "Forces through fines and tannins; cup reads bitter and dry."],
         title_size=18, body_size=14)

# ---- 4: grind & tamp ----
s = blank_slide(prs)
header(s, "Variable 2 — Grind and Tamping", "The most sensitive pair on the machine")
left = ["Start: one click coarser per day; a visibly coarse powder, not dust.",
        "Tamp flat and even — 10–15 kg of force is enough; unevenness extracts unevenly.",
        "Consistency beats absolute force: a wobble reads as bitterness."]
text(s, MARGIN, Inches(1.9), Inches(5.6), Inches(4.3), left,
     size=15, space_after=14, line_spacing=1.2)
rect(s, Inches(6.6), Inches(1.9), Inches(6.13), Inches(4.2), PANEL, radius=True)
text(s, Inches(7.0), Inches(2.2), Inches(5.4), Inches(2.6),
     ["FIELD TEST",
      "Pull twice at the same dose. Faster shot → grind coarser.\n"
      "Slower, weak shot → grind finer.",
      "One variable at a time; log grind, dose, yield and time."],
     size=16, color=INK, bold=False, space_after=12, line_spacing=1.2)
text(s, Inches(7.0), Inches(5.0), Inches(5.4), Inches(0.8),
     [[("Bold line in blue:", ACCENT, True), (" use accent for the one line that matters.", INK, False)]],
     size=14, space_after=0)

# ---- 5: temperature & water ----
s = blank_slide(prs)
header(s, "Variable 3 — Heat and Water", "99 °C is a ceiling, not a target")
stats = [("93–96", "°C pour-in for modern recipes (brighter coffees like the heat)"),
         ("25 s", "extraction window at 9 bar; outside it the cup goes sour or bitter"),
         ("38–50", "g/L TDS water; hard water scales the boiler, soft water tastes flat")]
x = MARGIN
for v, l in stats:
    rect(s, x, Inches(1.9), Inches(3.8), Inches(1.7), PANEL, radius=True)
    text(s, x + Inches(0.25), Inches(2.1), Inches(3.4), Inches(0.6),
         [v], size=30, color=ACCENT, bold=True, space_after=0)
    text(s, x + Inches(0.25), Inches(2.72), Inches(3.4), Inches(0.8),
         [l], size=13, color=MUTED, space_after=0, line_spacing=1.1)
    x += Inches(4.17)
text(s, MARGIN, Inches(4.2), CONTENT_W, Inches(2.2),
     ["Why it matters", "Extraction runs ~18–22% of the dry mass at the pour-in target; temperature sets the\n"
      "rate of the two. Change it and every other variable — grind, dose, time — shifts with it."],
     size=15, space_after=10, line_spacing=1.2)

# ---- 6: ratio & timing ----
s = blank_slide(prs)
header(s, "Variable 4 — Dose, Yield, Time", "The 1:2 recipe is a starting point")
card_row(s, Inches(1.9), 3, Inches(2.5),
         ["18 g in / 36 g out", "25 s total", "Front / back"],
         ["Standard 1:2 espresso: 18 g of dry coffee, ~36 g of liquid, medium body.",
          "First 10 s run light (sour if it stays light); last 10 s bring the sugar.",
          "Taste both: front under-extracted, over-extracted tail reads bitter."],
         body_size=14)
text(s, MARGIN, Inches(4.9), CONTENT_W, Inches(1.6),
     [[("Scale-first: ", ACCENT, True),
       ("weigh the cup at 10 s and 25 s; a flat curve means the grind stopped letting water through.", INK, False)]],
     size=15, space_after=0, line_spacing=1.2)

# ---- 7: takeaway ----
s = blank_slide(prs)
header(s, "Takeaway", "Change one thing, taste, log it")
items = [
    "Grind and tamping are 80% of the fight — fix those before touching temperature.",
    "Log dose, yield and time every pull; the log is the experiment.",
    "A lighter roast wants a cooler pour-in and a finer grind; dark roasts the reverse.",
]
y = Inches(1.95)
for it in items:
    rect(s, MARGIN, y, CONTENT_W, Inches(0.9), PANEL, radius=True)
    rect(s, MARGIN + Inches(0.2), y + Inches(0.24), Inches(0.42), Inches(0.42), ACCENT, radius=True)
    text(s, MARGIN + Inches(0.25), y + Inches(0.28), Inches(0.35), Inches(0.35),
         ["●"], size=14, color=BG, bold=True, space_after=0)
    text(s, MARGIN + Inches(0.9), y + Inches(0.26), Inches(11), Inches(0.5),
         [it], size=15, space_after=0)
    y += Inches(1.05)
text(s, MARGIN, Inches(5.4), CONTENT_W, Inches(0.9),
     ["Espresso is a controlled extraction: nine variables, one cup."],
     size=16, color=ACCENT, bold=True, space_after=0)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deck2.pptx")
prs.save(out)
print(out, len(prs.slides._sldIdLst), "slides")
