"""Check a .pptx for text overflow, overlaps, small fonts and low contrast."""
import sys
from pptx import Presentation
from pptx.util import Emu, Pt

FONT_W = 0.60  # avg char width as fraction of pt size (DejaVu Sans, mixed case)

def emu_in(v):
    return Emu(v).inches

def rel_luminance(rgb):
    c = [((rgb >> shift & 0xFF) / 255.0) ** 2.4 for shift in (16, 8, 0)]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]

def contrast_ratio(fg, bg):
    l1, l2 = sorted([rel_luminance(fg), rel_luminance(bg)])
    return (l1 + 0.05) / (l2 + 0.05)

def hex_of(rgb):
    try:
        return rgb.__int__()
    except Exception:
        return None

def main(path, min_font=12.0, min_contrast=4.5, overlap_pad_pt=2.0):
    prs = Presentation(path)
    sw_in, sh_in = emu_in(prs.slide_width), emu_in(prs.slide_height)
    bg_hex = 0xFFFFFF
    problems = []
    slides = list(prs.slides)
    for si, slide in enumerate(slides, 1):
        bg_hex = 0xFFFFFF
        try:
            if slide.background.fill.type is not None:
                bg_hex = hex_of(slide.background.fill.fore_color.rgb)
        except Exception:
            pass
        boxes = []
        for sh in slide.shapes:
            if not sh.has_text_frame:
                continue
            tf = sh.text_frame
            paras = []
            text_color = bg = None
            max_pt = 0.0
            n_chars = 0
            min_size = None
            for p in tf.paragraphs:
                for r in p.runs:
                    t = r.text
                    if not t.strip() and r is not p.runs[-1] and len(p.runs) > 1:
                        pass
                    n_chars += len(t)
                    if r.font.size:
                        sz = r.font.size.pt
                        min_size = sz if min_size is None else min(min_size, sz)
                        max_pt = max(max_pt, sz)
                    try:
                        text_color = hex_of(r.font.color.rgb)
                    except Exception:
                        pass
                    lines = t.count("\n") + 1
                    paras.extend([t[: t.find("\n")] if "\n" in t else t] * 0 + [seg for seg in t.split("\n")])
            if n_chars == 0:
                continue
            # estimate rendered height
            size = max_pt if max_pt else (min_size or 18.0)
            # use the smallest size present, since that is what overflows worst
            if min_size:
                size = min_size
            box_w_pt = emu_in(sh.width) * 72.0
            cpl = max(1.0, int((box_w_pt - 8) / (size * FONT_W)))
            total_lines = 0
            for p in tf.paragraphs:
                t = "".join(r.text for r in p.runs)
                psize = size
                for r in p.runs:
                    if r.font.size:
                        psize = min(psize, r.font.size.pt)
                n_txt = len(t)
                lines = max(1, -(-n_txt // cpl)) if n_txt else 1
                total_lines += lines
            line_h = size * 1.25
            need_in = total_lines * line_h / 72.0
            have_in = emu_in(sh.height)
            x0, y0 = emu_in(sh.left), emu_in(sh.top)
            x1, y1 = x0 + emu_in(sh.width), y0 + have_in
            # overflow: text taller than box, or box off-slide
            if need_in > have_in + 0.02:
                problems.append(f"slide {si}: text overflows box by ~{need_in - have_in:.2f}in (box {have_in:.2f}in, {total_lines} lines @~{size:.0f}pt) '{t[:30]}'")
            if x1 > sw_in + 0.01 or y1 > sh_in + 0.01 or x0 < -0.01 or y0 < -0.01:
                problems.append(f"slide {si}: shape off-slide ({x0:.2f},{y0:.2f})-({x1:.2f},{y1:.2f})")
            if min_size and min_size < min_font:
                problems.append(f"slide {si}: font {min_size:.0f}pt < {min_font:.0f}pt minimum")
            if text_color is not None:
                cr = contrast_ratio(text_color, bg_hex)
                if cr < min_contrast and min_size and min_size < 18:
                    problems.append(f"slide {si}: low contrast {cr:.2f}:1 for text {hex(text_color)} on {hex(bg_hex)} @~{min_size:.0f}pt")
            boxes.append((x0, y0, x1, y1, sh.name, min_size))
        # overlaps between sibling boxes with real text
        for a in range(len(boxes)):
            for b in range(a + 1, len(boxes)):
                ax0, ay0, ax1, ay1, an, ams = boxes[a]
                bx0, by0, bx1, by1, bn, bms = boxes[b]
                pad = overlap_pad_pt / 72.0
                if ax0 < bx1 - pad and bx0 < ax1 - pad and ay0 < by1 - pad and by0 < ay1 - pad:
                    # ignore nested (a fully inside b) - boxes are usually siblings
                    if ax0 >= bx0 and ax1 <= bx1:
                        continue
                    if bx0 >= ax0 and bx1 <= ax1:
                        continue
                    problems.append(f"slide {si}: boxes overlap: '{an}' vs '{bn}'")
    if problems:
        print(f"{path}: {len(problems)} problem(s)")
        for p in problems:
            print("  -", p)
        return 1
    print(f"{path}: clean (no overflow, no overlaps, no small fonts, no low-contrast text)")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1], min_font=float(sys.argv[2]) if len(sys.argv) > 2 else 12.0))
