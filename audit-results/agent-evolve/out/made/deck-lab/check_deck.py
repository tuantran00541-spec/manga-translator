"""Static + render QC for decks built with python-pptx.

Usage: python3 check_deck.py DECK.pptx [PNG_PREFIX]
Reads EMU layout data, calibrates each font's width (DejaVu/Liberation TTFs),
estimates text overflow and shape overlap, checks text/background contrast,
and - when PNG_PREFIX is given - verifies against a rendered page image.
"""
import colorsys
import glob
import os
import re
import subprocess
import sys
import zipfile

from PIL import Image, ImageFont
from pptx import Presentation
from pptx.enum.text import MSO_AUTO_SIZE
from pptx.util import Emu, Pt

EMU_PER_INCH = 914400
EMU_PER_PT = 12700
SLIDE_W_IN = 13.333333
SLIDE_H_IN = 7.5
LINE_FACTOR = 1.2
MIN_RUN_PT = 11
MIN_BODY_PT = 13

_FONT_CACHE = {}

FONT_FILES = {
    "DejaVu Sans": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "DejaVu Sans Bold": "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "Liberation Sans": "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "Liberation Sans Bold": "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
}





def _font(name, bold, size_pt):
    key = (name, bold, size_pt)
    if key not in _FONT_CACHE:
        base = FONT_FILES.get(name, FONT_FILES["DejaVu Sans"])
        if bold and name not in base:
            base = FONT_FILES.get(name + " Bold", FONT_FILES["DejaVu Sans Bold"])
        _FONT_CACHE[key] = ImageFont.truetype(base, max(6, round(size_pt * 4)))  # 4x for precision
    return _FONT_CACHE[key]


def text_width_emu(text, name, size_pt, bold):
    """Width in EMU of a single line of text. Measured from TTF, scaled."""
    f = _font(name, bold, size_pt)
    return f.getlength(text) / 4.0 * EMU_PER_PT / 100 * size_pt


def line_height_emu(size_pt):
    return size_pt * LINE_FACTOR * EMU_PER_PT


def _run_specs(p):
    """Yield (text, size_pt, name, bold, color_hex) per run, using defaults."""
    p_font = p.font
    for r in p.runs:
        yield (r.text, r.font.size.pt if r.font.size else (p_font.size.pt if p_font.size else 18),
               r.font.name or "DejaVu Sans",
               bool(r.font.bold),
               r.font.color.rgb if r.font.color and r.font.color.type is not None else None)


def para_metrics(p):
    """Estimated rendered height + width of one paragraph. Returns (height_emu, max_width_emu, color_hex, min_size_pt)."""
    specs = list(_run_specs(p))
    if not specs:
        specs = [(p.text if p.text else "", 18, "DejaVu Sans", False, None)]
    sizes = [s for _, s, *_ in specs]
    text = "".join(s[0] for s in specs)
    width = sum(text_width_emu(s[0], s[2], s[1], s[3]) for s in specs)
    return (max(sizes) * LINE_FACTOR * EMU_PER_PT, width, specs[-1][4], min(sizes))


def rel_lum(rgb):
    def chan(v):
        v /= 255.0
        return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
    r, g, b = rgb
    return 0.2126 * chan(r) + 0.7152 * chan(g) + 0.0722 * chan(b)


def contrast(rgb_a, rgb_b):
    la, lb = rel_lum(rgb_a), rel_lum(rgb_b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def slide_bg_emu_rect(shapes, slide, bg_rgb):
    """Find a full-bleed solid rectangle shape (the background block) if p:bg is ignored by LibreOffice."""
    for sh in shapes:
        if sh.shape_type is None:
            continue
        x, y, w, h = sh.left, sh.top, sh.width, sh.height
        if x is None:
            continue
        if w >= SLIDE_W_IN * EMU_PER_INCH * 0.98 and h >= SLIDE_H_IN * EMU_PER_INCH * 0.98:
            fill = getattr(sh.fill, "type", None)
            if fill is not None:
                from pptx.enum.dml import MSO_FILL
                if fill == MSO_FILL.SOLID:
                    return (sh, sh.fill.fore_color.rgb)
    return None


def check(pptx_path, png_prefix=None):
    prs = Presentation(pptx_path)
    issues = []
    EMU_PER_IN = EMU_PER_INCH
    SW, SH = SLIDE_W_IN * EMU_PER_IN, SLIDE_H_IN * EMU_PER_IN

    for si, slide in enumerate(prs.slides, 1):
        shapes = [s for s in slide.shapes if s.left is not None]

        # --- textboxes and autoshapes with text
        for sh in shapes:
            if not sh.has_text_frame:
                continue
            tf = sh.text_frame
            usable_w = sh.width - 0.2 * EMU_PER_IN  # ~0.1in margins each side
            usable_h = sh.height - 0.16 * EMU_PER_IN if sh.height else 1e15
            est_h = 0.0
            min_size = None
            overflow_w = 0.0
            for p in tf.paragraphs:
                h, w, color, smin = para_metrics(p)
                est_h += h
                overflow_w = max(overflow_w, w)
                if min_size is None or smin < min_size:
                    min_size = smin
                if tf.word_wrap is False and w > sh.width * 1.02:
                    issues.append(f"S{si}: unwrapped text {w / EMU_PER_INCH:.2f}in wider than box {sh.width / EMU_PER_INCH:.2f}in in '{sh.name}'")
            if est_h > usable_h * 1.05:
                issues.append(f"S{si}: text overflow est. {est_h / EMU_PER_INCH:.2f}in > box {usable_h / EMU_PER_INCH:.2f}in in '{sh.name}'")
            if min_size is not None and min_size < MIN_RUN_PT:
                issues.append(f"S{si}: font {min_size}pt < {MIN_RUN_PT}pt floor in '{sh.name}'")
            for p in tf.paragraphs:
                for run in p.runs:
                    sz = run.font.size.pt if run.font.size else 18
                    if sz < MIN_BODY_PT and not run.font.bold:
                        issues.append(f"S{si}: non-bold {sz}pt text in '{sh.name}'")
                        break

        # --- overlap detection among *fill* shapes (rectangles etc.), not text frames
        solid = [s for s in shapes if s.shape_type is not None and s.left is not None
                 and not s.has_text_frame]
        for i in range(len(solid)):
            for j in range(i + 1, len(solid)):
                a, b = solid[i], solid[j]
                ax1, ay1, ax2, ay2 = a.left, a.top, a.left + a.width, a.top + a.height
                bx1, by1, bx2, by2 = b.left, b.top, b.left + b.width, b.top + b.height
                ox, oy = max(0, min(ax2, bx2) - max(ax1, bx1)), max(0, min(ay2, by2) - max(ay1, by1))
                if ox > 0 and oy > 0:
                    area = ox * oy
                    sa, sb = a.width * a.height, b.width * b.height
                    if sa > 0 and sb > 0:
                        frac = area / min(sa, sb)
                        if frac > 0.3:
                            issues.append(f"S{si}: overlap {frac*100:.0f}% between '{a.name}' and '{b.name}'")

        # --- per-slide word count and left-margin sanity
        words = 0
        for sh in shapes:
            if sh.has_text_frame:
                words += len(sh.text_frame.text.split())
        if words > 220:
            issues.append(f"S{si}: dense slide, {words} words; target under 200")
        for sh in shapes:
            # skip the full-bleed background rect (0in,0in covering the whole slide)
            is_bg = (sh.left == 0 and sh.top == 0
                     and sh.width and sh.height
                     and sh.width >= SW * 0.98 and sh.height >= SH * 0.98)
            if is_bg:
                continue
            if sh.left is not None and sh.has_text_frame and sh.left < 0.3 * EMU_PER_INCH and sh.width and sh.width > 0.5 * EMU_PER_INCH:
                issues.append(f"S{si}: text '{sh.name}' starts at {sh.left/EMU_PER_INCH:.2f}in from left edge (< 0.3in margin)")

        # --- contrast
        bg = None
        fullbleed = slide_bg_emu_rect(shapes, slide, None)
        if fullbleed:
            bg = fullbleed[1]
        # try p:bg in xml
        import lxml.etree as ET
        sxml = slide._element
        P = "{http://schemas.openxmlformats.org/presentationml/2006/main}"
        cSld = sxml.find(f"{P}cSld")
        if cSld is not None:
            bgel = cSld.find(f"{P}bg")
            if bgel is not None:
                A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
                clr = bgel.find(f"{A}bgPr/{A}solidFill/{A}srgbClr")
                if clr is not None:
                    bg = clr.get("val")
        bg_rgb = None
        if bg:
            bg = str(bg)
            if len(bg) == 6:
                bg_rgb = tuple(int(bg[i:i + 2], 16) for i in (0, 2, 4))
        if bg_rgb is not None:
            for sh in shapes:
                if not sh.has_text_frame:
                    continue
                for p in sh.text_frame.paragraphs:
                    for spec in _run_specs(p):
                        c = spec[4]
                        if c is None:
                            continue
                        t_rgb = (int(str(c)[0:2], 16), int(str(c)[2:4], 16), int(str(c)[4:6], 16))
                        cr = contrast(bg_rgb, t_rgb)
                        if cr < 4.5:
                            issues.append(f"S{si}: contrast {cr:.2f} < 4.5 ('{sh.name}')")

    # --- rendered verification
    if png_prefix:
        pages = sorted(glob.glob(os.path.join(os.path.dirname(png_prefix) or ".", os.path.basename(png_prefix) + "-*.png")))
        if pages:
            scale = Image.open(pages[0]).width / SLIDE_W_IN  # px per inch
            for si, path in enumerate(pages, 1):
                img = Image.open(path).convert("RGB")
                im_h, im_w = img.height, img.width
                # sample bg from corners (avoid text)
                bg_px = [img.getpixel((3, 3)), img.getpixel((im_w - 3, 3)),
                         img.getpixel((3, im_h - 3)), img.getpixel((im_w - 3, im_h - 3))]
                if len(set(bg_px)) > 1:
                    issues.append(f"S{si}: uneven corners {bg_px} (bg may be wrong)")
                # check slide bounds: shapes that exceed the page
                slide = list(prs.slides)[si - 1]
                for sh in slide.shapes:
                    if sh.left is None or sh.width is None:
                        continue
                    right_in = (sh.left + sh.width) / EMU_PER_INCH
                    bottom_in = (sh.top + sh.height) / EMU_PER_INCH
                    if right_in > SLIDE_W_IN * 1.01 or bottom_in > SLIDE_H_IN * 1.01:
                        issues.append(
                            f"S{si}: shape '{sh.name}' extends off slide "
                            f"(right {right_in:.2f}in > {SLIDE_W_IN:.2f}in, "
                            f"bottom {bottom_in:.2f}in > {SLIDE_H_IN:.2f}in)")

    if issues:
        print(f"\n=== ISSUES ({len(issues)}) ===")
        for it in issues:
            print("  " + it)
    else:
        print("\n=== CLEAN ===")
    return issues


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    path = sys.argv[1]
    png_prefix = sys.argv[2] if len(sys.argv) > 2 else None
    issues = check(path, png_prefix)
    sys.exit(1 if issues else 0)


if __name__ == "__main__":
    main()
