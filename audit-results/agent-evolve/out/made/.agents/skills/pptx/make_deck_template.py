"""deck-lab generator skeleton: helper functions + palette. Copy, edit content, run.
Usage: python make_deck.py  (expect output saved next to the script)."""
import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# ---------- palette: light theme (swap for dark: see SKILL.md) ----------
INK    = RGBColor(0x1F, 0x24, 0x30)   # body text on light
ACCENT = RGBColor(0x0F, 0x62, 0xFE)   # blue accent
PANEL  = RGBColor(0xE8, 0xEE, 0xF9)    # light blue card fill
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
GRAY   = RGBColor(0x5B, 0x64, 0x74)    # muted secondary text
BG     = RGBColor(0xF7, 0xF9, 0xFC)

FONT = "Segoe UI"

# ---------- grid constants (16:9) ----------
SLIDE_W, SLIDE_H = Inches(13.333), Inches(7.5)
MARGIN = Inches(0.6)
CONTENT_W = SLIDE_W - 2 * MARGIN          # 12.13in
HEADER_BOTTOM = Inches(1.6)               # nothing content starts above this
FOOTER_TOP = Inches(6.9)

LINE_IN = lambda pt, lines: pt * 1.3 / 72.0 * lines  # budget box height per text block


def blank_slide(prs, bg=BG):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = bg
    return s


def rect(s, x, y, w, h, color, radius=False):
    shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    sh = s.shapes.add_shape(shape_type, x, y, w, h)
    sh.fill.solid()
    sh.fill.fore_color.rgb = color
    sh.line.fill.background()
    sh.shadow.inherit = False
    return sh


def text(s, x, y, w, h, runs, size=16, color=INK, bold=False,
         align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, line_spacing=1.0,
         space_after=6):
    """runs: list of paragraphs; each is a str or a list of (text, color, bold).
    Budget h with LINE_IN(size, n_lines) so the checker does not flag overflow."""
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
    return tb


def header(s, kicker, title):
    text(s, MARGIN, Inches(0.42), CONTENT_W, Inches(0.3),
         [kicker.upper()], size=13, color=ACCENT, bold=True, space_after=0)
    text(s, MARGIN, Inches(0.72), CONTENT_W, Inches(0.75),
         [title], size=32, bold=True, space_after=0)
    rect(s, MARGIN, Inches(1.45), Inches(0.55), Inches(0.055), ACCENT)


def card_row(s, y, n, height, title_size=20, body_size=15):
    """n cards side by side; returns a callable(s, i, title, body) for filling each."""
    card_w = (CONTENT_W / n) - Inches(0.37) if n > 1 else CONTENT_W
    pitch = card_w + Inches(0.37)
    def fill(i, title, body):
        x = MARGIN + i * pitch
        rect(s, x, y, card_w, height, PANEL, radius=True)
        text(s, x + Inches(0.3), y + Inches(0.25), card_w - Inches(0.6),
             Inches(0.5), [title], size=title_size, bold=True, space_after=0)
        text(s, x + Inches(0.3), y + Inches(0.75), card_w - Inches(0.6),
             height - Inches(0.9), [body], size=body_size, color=GRAY,
             space_after=0, line_spacing=1.15)
    return fill


EMU_IN = 914400


def build(prs, slides):
    """slides: list of callables, one per content slide, each (s, prs) -> None.
    Slide 1 is assumed to be the title slide and is handled here."""
    s = blank_slide(prs, WHITE)
    rect(s, 0, 0, Inches(0.18), SLIDE_H, ACCENT)
    text(s, MARGIN + Inches(0.4), Inches(2.7), CONTENT_W, Inches(0.4),
         ["KICKER LINE"], size=14, color=ACCENT, bold=True, space_after=0)
    text(s, MARGIN + Inches(0.4), Inches(3.2), CONTENT_W, Inches(1.9),
         ["Deck Title Goes Here\nSecond line if needed"], size=50, bold=True,
         space_after=0, line_spacing=1.05)
    text(s, MARGIN + Inches(0.4), Inches(5.1), CONTENT_W, Inches(0.9),
         ["One-sentence subtitle describing the deck, the audience, and the takeaway."],
         size=17, color=GRAY, space_after=0, line_spacing=1.1)
    for fn in slides:
        fn(prs.slides.add_slide(prs.slide_layouts[6]), prs)


def main():
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H

    def slide_skeleton(s, prs):
        header(s, "Section kicker", "Slide title — one idea only")
        fill = card_row(s, Inches(1.9), 3, Inches(2.4))
        fill(0, "Card A title", "Two to three lines of body at 15pt, no more than ~90 chars per card.")
        fill(1, "Card B title", "Same shape, second idea. Keep cards parallel: same length, same depth.")
        fill(2, "Card C title", "Third card. If the text runs long, split the slide rather than shrinking the font.")

    build(prs, [slide_skeleton] * 6)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deck.pptx")
    prs.save(out)
    print(out, len(prs.slides._sldIdLst), "slides")


if __name__ == "__main__":
    main()
