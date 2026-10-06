"""deck-lab/check_deck.py — open a pptx, flag text overflow, shape overlaps,
small fonts, and low-contrast text; print a report. Usage: python check_deck.py deck.pptx"""
import sys, math
from pptx import Presentation
from pptx.util import Pt

# Rough char-width factors for common fonts at given pt size (em units).
def est_lines(text, width_in, pt, bold):
    """Estimate how many wrapped lines text needs in a box width_in inches."""
    cw = pt * (0.55 if bold else 0.5) / 72.0  # inches per char approx
    per_line = max(1, int(width_in / cw))
    lines = 0
    for seg in text.split("\n"):
        lines += max(1, math.ceil(len(seg) / per_line))
    return lines

def lum(rgb):
    r, g, b = [v / 255.0 for v in rgb]
    f = lambda x: x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4
    r, g, b = f(r), f(g), f(b)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def contrast(c1, c2):
    l1, l2 = lum(c1), lum(c2)
    if l1 < l2:
        l1, l2 = l2, l1
    return (l1 + 0.05) / (l2 + 0.05)

def bg_of(slide, shape):
    """Find the fill under a shape: the smallest filled shape that contains it, else slide bg."""
    candidates = []
    for sh in slide.shapes:
        if sh is shape or sh.top is None or sh.left is None or shape.left is None:
            continue
        try:
            if sh.fill.type != 1:
                continue
        except Exception:
            continue
        x1, y1, x2, y2 = sh.left, sh.top, sh.left + sh.width, sh.top + sh.height
        if x1 <= shape.left and shape.left + shape.width <= x2 + 10000 and y1 <= shape.top and shape.top + shape.height <= y2 + 10000:
            candidates.append((sh.width * sh.height, sh.fill.fore_color.rgb))
    if candidates:
        return min(candidates)[1]
    b = slide.background.fill
    if b.type is not None and b.type == 1:
        return b.fore_color.rgb
    return None

def main(path):
    prs = Presentation(path)
    EMU_IN = 914400
    problems = []
    for i, slide in enumerate(prs.slides, 1):
        shapes = [s for s in slide.shapes if s.top is not None]
        for s in shapes:
            x, y = s.left / EMU_IN, s.top / EMU_IN
            w, h = s.width / EMU_IN, s.height / EMU_IN
            if x + w > prs.slide_width / EMU_IN + 0.01 or y + h > prs.slide_height / EMU_IN + 0.01:
                problems.append(f"s{i}: shape '{s.shape_type}' exceeds slide bounds ({x+w:.2f}x{y+h:.2f})")
            # overlaps between text-bearing shapes
            for o in shapes:
                if o is s:
                    continue
                ox, oy = o.left / EMU_IN, o.top / EMU_IN
                ow, oh = o.width / EMU_IN, o.height / EMU_IN
                inter_w = min(x + w, ox + ow) - max(x, ox)
                inter_h = min(y + h, oy + oh) - max(y, oy)
                if inter_w > 0.15 and inter_h > 0.15 and s.has_text_frame and o.has_text_frame:
                    a = s.text_frame.text.strip()
                    b2 = o.text_frame.text.strip()
                    if a and b2:
                        problems.append(f"s{i}: textboxes overlap ({inter_w:.2f}x{inter_h:.2f}) '{a[:24]}' / '{b2[:24]}'")
            if not s.has_text_frame:
                continue
            tf = s.text_frame
            text = tf.text.strip()
            if not text:
                continue
            # overflow estimate
            need = 0.0
            for p in tf.paragraphs:
                runs = list(p.runs)
                pt = max((r.font.size.pt if r.font.size else 18) for r in runs) if runs else 18
                bold = any(r.font.bold for r in runs)
                t = "".join(r.text for r in runs)
                n = est_lines(t, w, pt, bold)
                ls = p.line_spacing or 1.0
                ls = ls if isinstance(ls, float) else 1.0
                need += n * pt * 1.25 * ls / 72.0 + (p.space_after.pt / 72.0 if p.space_after else 0)
            if need > h + 0.05:
                problems.append(f"s{i}: text likely overflows: needs ~{need:.2f}in, box {h:.2f}in ('{text[:40]}')")
            # min font + contrast
            for p in tf.paragraphs:
                for r in p.runs:
                    if not r.text.strip():
                        continue
                    sz = r.font.size.pt if r.font.size else None
                    if sz and sz < 12:
                        problems.append(f"s{i}: font too small ({sz}pt): '{r.text[:30]}'")
                    if r.font.color and r.font.color.type is not None:
                        try:
                            fg = tuple(r.font.color.rgb)
                        except Exception:
                            continue
                        bg = bg_of(slide, s)
                        if bg is not None:
                            cr = contrast(fg, tuple(bg))
                            if cr < 3.0:
                                problems.append(f"s{i}: low contrast {cr:.2f}: '{r.text[:30]}' fg={fg} bg={tuple(bg)}")
    print(f"checked {i} slides")
    if problems:
        for p in problems:
            print(" !", p)
    else:
        print("clean: no overflow/overlap/contrast issues detected")
    return 1 if problems else 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
