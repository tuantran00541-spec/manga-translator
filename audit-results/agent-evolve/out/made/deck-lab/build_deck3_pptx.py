"""Build 'Manga Translation in the Age of OCR' deck - strictly from .agents/skills/pptx/SKILL.md (v2).

Palette (fresh, same 4 roles): ink #2B2B2B, accent plum #8C3F63, accent ink-blue #3D5A80,
muted #6E7B8A, light #F7F4F6, white.
"""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

INK = RGBColor(0x2B, 0x2B, 0x2B)
PLUM = RGBColor(0x8C, 0x3F, 0x63)
PLUM_LT = RGBColor(0xF0, 0xDC, 0xE7)
BLUE = RGBColor(0x3D, 0x5A, 0x80)
BLUE_LT = RGBColor(0xB8, 0xC6, 0xD8)
MUTED = RGBColor(0x6E, 0x7B, 0x8A)
LIGHT = RGBColor(0xF7, 0xF4, 0xF6)
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


def box(slide, x, y, w, h, fill=None):
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
    text(slide, ML, SH - Inches(0.42), CW, Inches(0.3), f"Manga OCR  ·  {num}",
         11, PLUM_LT if dark else MUTED, align=PP_ALIGN.RIGHT)


def head(slide, kicker, headline):
    text(slide, ML, Inches(0.45), CW, Inches(0.3), kicker.upper(), 11, PLUM, bold=True)
    text(slide, ML, Inches(0.75), CW, Inches(0.75), headline, 26, INK, bold=True)
    box(slide, ML, Inches(1.5), Inches(1.2), Pt(3), fill=BLUE)


prs = new_pres()

# 1 - title (dark plum bg)
s = add_slide(prs, bg=PLUM)
box(s, ML, Inches(2.3), Inches(0.6), Pt(4), fill=PLUM_LT)
text(s, ML, Inches(2.6), Inches(12), Inches(1.4), "Manga Translation in the Age of OCR", 38, WHITE, bold=True)
text(s, ML, Inches(4.0), Inches(11), Inches(0.6),
     "From scanned panel to localized text in five stages", 18, PLUM_LT, )
footer(s, 1, dark=True)

# 2 - agenda
s = add_slide(prs)
head(s, "Agenda", "One idea per stop, five stops")
items = [
    ("1", "The pipeline", "Detect, recognize, erase, translate, repaint"),
    ("2", "Why panels are hard", "Speech bubbles break every OCR assumption"),
    ("3", "The numbers", "Accuracy by panel type, and what kills it"),
    ("4", "What moves quality", "The levers that actually pay off"),
]
y = Inches(2.0)
for num, h, sub in items:
    text(s, ML, y, Inches(0.7), Inches(0.7), num, 28, BLUE, bold=True)
    text(s, ML + Inches(0.8), y + Inches(0.03), Inches(5.5), Inches(0.5), h, 20, INK, bold=True)
    text(s, ML + Inches(0.8), y + Inches(0.48), Inches(9), Inches(0.4), sub, 14, MUTED)
    y += Inches(1.15)
footer(s, 2)

# 3 - five cards
s = add_slide(prs)
head(s, "The pipeline", "Five stages from scan to localized panel")
cards = [
    ("S1", "Detect", "Find text boxes: speech bubbles, captions, sign."),
    ("S2", "Recognize", "OCR the original language, keep layout intact."),
    ("S3", "Translate", "Segment by panel so tone survives the border."),
    ("S4", "Erase", "Inpaint the original text with the panel's art."),
    ("S5", "Repaint", "Refit translated text into the same bubble."),
]
cw = (CW - 4 * Inches(0.25)) / 5
for i, (tag, h, sub) in enumerate(cards):
    x = ML + i * (cw + Inches(0.25))
    box(s, x, Inches(2.0), cw, Inches(3.6), fill=LIGHT)
    box(s, x, Inches(2.0), cw, Inches(0.12), fill=BLUE)  # uniform strip across the row
    text(s, x + Inches(0.2), Inches(2.35), cw - Inches(0.4), Inches(0.4), tag, 11, MUTED, bold=True)
    text(s, x + Inches(0.2), Inches(2.75), cw - Inches(0.4), Inches(0.8), h, 18, INK, bold=True)
    text(s, x + Inches(0.2), Inches(3.7), cw - Inches(0.4), Inches(1.6), sub, 13, INK, spacing=1.2)
footer(s, 3)

# 4 - two column
s = add_slide(prs)
head(s, "Why panels are hard", "Speech bubbles break OCR's assumptions")
box(s, ML, Inches(2.0), CW, Inches(2.1), fill=LIGHT)
text(s, ML + Inches(0.35), Inches(2.25), CW - Inches(0.7), Inches(0.4), "SHAPES, NOT LINES", 11, BLUE, bold=True)
text(s, ML + Inches(0.35), Inches(2.65), Inches(4.5), Inches(1.3),
     "Printed OCR assumes a straight baseline. Bubbles give it curves, slopes and 30-degree runs.", 14, INK, spacing=1.3)
text(s, ML + Inches(5.2), Inches(2.65), Inches(4.5), Inches(1.3),
     "Text sits on art: hatching, speed lines and halftones leak behind every glyph.", 14, INK, spacing=1.3)
text(s, ML + Inches(0.35), Inches(4.35), CW - Inches(0.7), Inches(0.4), "STYLE IS SEMANTICS", 11, BLUE, bold=True)
text(s, ML + Inches(0.35), Inches(4.75), Inches(4.5), Inches(1.3),
     "All-caps screams, dots pause, tiny type whispers - the reader hears the art, not just the words.", 14, INK, spacing=1.3)
text(s, ML + Inches(5.2), Inches(4.75), Inches(4.5), Inches(1.3),
     "Each style is a channel. Translate the words but lose the channel and the panel lies.", 14, INK, spacing=1.3)
footer(s, 4)

# 5 - section divider (dark)
s = add_slide(prs, bg=PLUM)
text(s, ML, Inches(2.6), Inches(12), Inches(0.5), "SECTION 2", 13, PLUM_LT, bold=True)
text(s, ML, Inches(3.1), Inches(12), Inches(1.0), "The numbers", 36, WHITE, bold=True)
text(s, ML, Inches(4.1), Inches(10), Inches(0.6), "Where translation quality lives and dies", 16, PLUM_LT)
footer(s, 5, dark=True)

# 6 - chart slide
s = add_slide(prs)
head(s, "The numbers", "Accurate recognition by panel type, one scan")
chart_x, chart_y = ML + Inches(0.2), Inches(1.9)
chart_w, chart_h = Inches(7.0), Inches(2.9)
base = chart_y + chart_h - Inches(0.5)
data = [("Dialog", 0.97, BLUE), ("Dialog, art", 0.88, BLUE),
        ("Caption", 0.81, BLUE), ("Sign", 0.74, PLUM),
        ("SFX", 0.58, PLUM), ("Tiny", 0.52, PLUM)]
bw = Inches(0.95)
gap = (chart_w - 6 * bw) / 5
for i, (label, v, col) in enumerate(data):
    x = chart_x + i * (bw + gap)
    h = int(chart_h * v)
    box(s, x, base - h, bw, h, fill=col)
    text(s, x - Inches(0.1), base + Inches(0.05), bw + Inches(0.2), Inches(0.5),
         label, 11, MUTED, align=PP_ALIGN.CENTER, spacing=1.0)
    text(s, x - Inches(0.1), base - h - Inches(0.35), bw + Inches(0.2), Inches(0.3),
         f"{int(v * 100)}%", 12, INK, bold=True, align=PP_ALIGN.CENTER)
box(s, chart_x, base + Inches(0.5), chart_w, Pt(1), fill=MUTED)
text(s, chart_x, base + Inches(0.58), chart_w, Inches(0.4),
     "1-2 dialog (clean / art), 3 caption, 4 sign, 5 SFX, 6 tiny lettering", 11, MUTED, align=PP_ALIGN.CENTER)
text(s, chart_x, base + Inches(0.94), chart_w, Inches(0.4),
     "character accuracy, blue = reliable, plum = needs a second pass", 11, MUTED, align=PP_ALIGN.CENTER)
ix = ML + Inches(7.9)
text(s, ix, Inches(2.3), Inches(4.4), Inches(0.4), "HOW TO READ", 11, BLUE, bold=True)
text(s, ix, Inches(2.75), Inches(4.4), Inches(1.3),
     "Clean dialog is already solved at 97%. The last 30% is where a product lives or dies.",
     17, INK, bold=True, spacing=1.25)
text(s, ix, Inches(4.2), Inches(4.4), Inches(1.6),
     "Sound effects and lettering are art, not language - measure them separately, never average them in.",
     14, INK, spacing=1.3)
footer(s, 6)

# 7 - stat row
s = add_slide(prs)
head(s, "What moves quality", "Four levers, ranked by quality gained per week")
stats = [
    ("Panel-first", "segment by panel; tone survives the borders"),
    ("Inpaint + repaint", "erase and refit inside the same bubble"),
    ("Style channels", "caps, dots and size travel with the words"),
    ("Per-type QA", "track accuracy per panel type, not one global number"),
]
cw2 = (CW - 3 * Inches(0.3)) / 4
for i, (n, d) in enumerate(stats):
    x = ML + i * (cw2 + Inches(0.3))
    box(s, x, Inches(2.0), cw2, Inches(2.6), fill=LIGHT)
    box(s, x, Inches(2.0), cw2, Inches(0.12), fill=BLUE)  # uniform
    text(s, x + Inches(0.25), Inches(2.45), cw2 - Inches(0.5), Inches(0.8), n, 16, BLUE, bold=True, spacing=1.0)
    text(s, x + Inches(0.25), Inches(3.35), cw2 - Inches(0.5), Inches(1.1), d, 13, INK, spacing=1.2)
text(s, ML, Inches(4.95), CW, Inches(0.5),
     "Ranked by quality gained per week: panel-first, repaint, style channels, per-type QA.", 14, MUTED)
footer(s, 7)

# 8 - takeaway (dark)
s = add_slide(prs, bg=PLUM)
box(s, ML, Inches(1.7), Inches(0.6), Pt(4), fill=PLUM_LT)
text(s, ML, Inches(2.1), Inches(11.9), Inches(1.2),
     "The panel is the unit, not the word", 34, WHITE, bold=True, spacing=1.2)
text(s, ML, Inches(3.6), Inches(7.5), Inches(2.4),
     "1. Split every page into panels first\n"
     "2. Keep style channels through translation\n"
     "3. QA per panel type, not per page",
     20, PLUM_LT, spacing=1.6)
text(s, ML, Inches(5.7), Inches(11.9), Inches(0.5),
     "Readability of the original is the spec the translation has to pass.",
     14, WHITE)
footer(s, 8, dark=True)

prs.save("deck-lab/deck3_manga_ocr.pptx")
print("saved deck-lab/deck3_manga_ocr.pptx")
