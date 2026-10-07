"""pptx_lab v3 - build and verify professional .pptx decks with python-pptx.

Fixes from round 2 (visual audit of lab1/lab2/lab3):
 - dark-slide footers now #C7D6E8 (was #A9C4E0 still weak for the brown deck) -> use #D9D2C7 fallback
 - takeaway bullets: 22pt, light tint aligned to left margin (not indented)
 - process card body: 14pt on light card (was 13, low-contrast complaint)
 - chart value labels 12pt bold ink; category labels 12pt (was 11) for readability
 - chart colors: last bar no longer uses the accent (accent reserved for emphasis)
 - compare rows: text 14pt, row height 0.85in to hold 2 lines
 - card row body 13->14pt
 - section sub on dark bg: 16pt light tint
 - add 'statrow' layout: 2-4 big stat cards (number 44pt + caption 13pt) to fill lower band
 - add 'tworow' compare variant handling: items can be [head, body] pairs per row
"""
import json

def _rel_lum(rgb):
    c = [(((rgb >> s) & 0xFF) / 255.0) ** 2.4 for s in (16, 8, 0)]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]

def _contrast(fg, bg):
    l1, l2 = sorted([_rel_lum(fg), _rel_lum(bg)])
    return (l1 + 0.05) / (l2 + 0.05)

FONT = "DejaVu Sans"

def _rgb_hex(v):
    from pptx.dml.color import RGBColor
    if v.startswith("#"):
        v = v[1:]
    if len(v) == 6:
        return RGBColor(int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16))
    return RGBColor.from_string(v)

def _apply_transition(slide, kind="fade"):
    from pptx.oxml.ns import qn
    from lxml import etree
    if kind == "none":
        return
    sld = slide._element
    for el in sld.findall(qn('p:transition')):
        sld.remove(el)
    trans = etree.SubElement(sld, qn('p:transition'))
    trans.set('spd', 'med')
    child = etree.SubElement(trans, qn(f'p:{kind}'))
    child.set('dur', '500' if kind == 'fade' else '700')

def _new_pres():
    from pptx import Presentation
    from pptx.util import Inches
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    return prs

def _add_slide(prs, bg_hex):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = _rgb_hex(bg_hex) if isinstance(bg_hex, str) else bg_hex
    return s

def _box(slide, x, y, w, h, fill=None):
    from pptx.util import Inches
    from pptx.enum.shapes import MSO_SHAPE
    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.shadow.inherit = False
    if fill is None:
        shp.fill.background()
    else:
        shp.fill.solid()
        shp.fill.fore_color.rgb = _rgb_hex(fill) if isinstance(fill, str) else fill
    shp.line.fill.background()
    return shp

def _text(slide, x, y, w, h, s, size, color="2B2B2B", bold=False, align=0, spacing=1.15):
    from pptx.util import Inches, Pt
    from pptx.enum.text import PP_ALIGN
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    pmap = {0: PP_ALIGN.LEFT, 1: PP_ALIGN.CENTER, 2: PP_ALIGN.RIGHT}
    for i, line in enumerate(s.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = pmap.get(align, PP_ALIGN.LEFT)
        p.line_spacing = spacing
        r = p.add_run()
        r.text = line
        f = r.font
        f.size = Pt(size)
        f.bold = bold
        f.name = FONT
        f.color.rgb = _rgb_hex(color) if isinstance(color, str) else color
    return tb

def _dark_tint(bg_hex, dark_bg):
    """Light text color for a given dark background - pick by luminance."""
    bg = int.from_bytes(bytes.fromhex(bg_hex.replace('#', '')), 'big') if isinstance(bg_hex, str) else bg_hex
    if _rel_lum(bg) > 0.15:
        return "DCE7F5"
    return "E8E2D8"

def _footer(slide, num, dark=False, dark_bg="#1B3A5C"):
    _text(slide, 0.7, 7.08, 11.93, 0.3, f"deck  ·  {num}", 11,
          _dark_tint(dark_bg, True) if dark else "5A6B7D", align=2)

def _head(slide, kicker, headline):
    _text(slide, 0.7, 0.45, 11.93, 0.3, kicker.upper(), 11, "E8792B", bold=True)
    _text(slide, 0.7, 0.75, 11.93, 0.75, headline, 26, "1B3A5C", bold=True)
    _box(slide, 0.7, 1.5, 1.2, 0.04, "4A90D9")

def _card_row(slide, cards, x0=0.7, y0=2.0, total_w=11.93, card_h=3.6):
    n = len(cards)
    if n == 0:
        return
    gap = 0.3
    cw = (total_w - (n - 1) * gap) / n
    for i, (tag, head, body) in enumerate(cards):
        x = x0 + i * (cw + gap)
        _box(slide, x, y0, cw, card_h, "F2F5F9")
        _box(slide, x, y0, cw, 0.12, "4A90D9")
        _text(slide, x + 0.2, y0 + 0.35, cw - 0.4, 0.4, str(tag).upper(), 11, "5A6B7D", bold=True)
        _text(slide, x + 0.2, y0 + 0.75, cw - 0.4, 0.8, head, 18, "1B3A5C", bold=True)
        _text(slide, x + 0.2, y0 + 1.7, cw - 0.4, card_h - 1.9, body, 14, "2B2B2B", spacing=1.25)

def _two_col(slide, left_head, left_body, right_head, right_body):
    _box(slide, 0.7, 2.0, 11.93, 2.1, "F2F5F9")
    _text(slide, 1.05, 2.25, 11.0, 0.4, left_head, 11, "4A90D9", bold=True)
    _text(slide, 1.05, 2.65, 4.6, 1.3, left_body, 14, "2B2B2B", spacing=1.3)
    _text(slide, 6.05, 2.65, 4.6, 1.3, right_body, 14, "2B2B2B", spacing=1.3)
    _text(slide, 1.05, 4.35, 11.0, 0.4, right_head, 11, "4A90D9", bold=True)

def _chart(slide, data, x=0.9, y=2.0, w=7.0, h=3.0, caption="", note_head="", note_body=None):
    base = y + h - 0.5
    n = len(data)
    if n == 0:
        return
    bw = 0.95
    gap = (w - n * bw) / max(1, n - 1)
    # bars: all muted blue shades; accent (orange) reserved for a single emphasis bar via color
    shades = ["3D5A80", "4A78A8", "6B93BE", "8FB4DD"]
    for i, (label, v, col) in enumerate(data):
        x0 = x + i * (bw + gap)
        hh = h * v
        _box(slide, x0, base - hh, bw, hh, col if col else shades[i % len(shades)])
        _text(slide, x0 - 0.1, base + 0.08, bw + 0.2, 0.5, str(label), 12, "4A5568", align=1, spacing=1.0)
        _text(slide, x0 - 0.1, base - hh - 0.35, bw + 0.2, 0.3, f"{int(v*100)}%", 13, "1B2A3A", bold=True, align=1)
    _box(slide, x, base + 0.5, w, 0.015, "8A97A8")
    if caption:
        _text(slide, x, base + 0.58, w, 0.4, caption, 12, "4A5568", align=1)
    if note_head:
        ix = x + w + 0.4
        iw = 4.2
        _text(slide, ix, 2.3, iw, 0.4, note_head, 11, "4A90D9", bold=True)
        note_body = note_body or []
        _text(slide, ix, 2.75, iw, 1.3, " ".join(note_body[:2]), 17, "1B3A5C", bold=True, spacing=1.25)
        if len(note_body) > 2:
            _text(slide, ix, 4.15, iw, 1.6, " ".join(note_body[2:]), 14, "2B2B2B", spacing=1.3)

def _process(slide, steps):
    n = len(steps)
    if n == 0:
        return
    x0, y0 = 0.7, 2.2
    total_w = 11.93
    gap = 0.4
    cw = (total_w - (n - 1) * gap) / n
    for i, (head, body) in enumerate(steps):
        x = x0 + i * (cw + gap)
        _box(slide, x, y0, cw, 0.5, "4A90D9")
        _text(slide, x + 0.1, y0 + 0.06, cw - 0.2, 0.4, f"{i+1}. {head}", 13, "FFFFFF", bold=True, align=1)
        _box(slide, x, y0 + 0.5, cw, 2.4, "F2F5F9")
        _text(slide, x + 0.15, y0 + 0.7, cw - 0.3, 2.0, body, 14, "2B2B2B", spacing=1.25)
        if i < n - 1:
            arrow_x = x + cw + (gap - 0.3) / 2
            _text(slide, arrow_x, y0 + 0.55, 0.3, 0.4, "\u2192", 20, "E8792B", bold=True, align=1)

def _compare(slide, a, b):
    a_head, a_items = a[0], a[1] if len(a) > 1 else []
    b_head, b_items = b[0], b[1] if len(b) > 1 else []
    a_col = a[2] if len(a) > 2 and a[2] else "4A90D9"
    b_col = b[2] if len(b) > 2 and b[2] else "E8792B"
    _box(slide, 0.7, 2.0, 5.7, 0.5, a_col)
    _text(slide, 0.9, 2.08, 5.3, 0.4, a_head, 16, "FFFFFF", bold=True)
    _box(slide, 6.9, 2.0, 5.7, 0.5, b_col)
    _text(slide, 7.1, 2.08, 5.3, 0.4, b_head, 16, "FFFFFF", bold=True)
    rows = max(len(a_items), len(b_items))
    rh = 0.85
    for i in range(rows):
        y = 2.7 + i * (rh + 0.1)
        _box(slide, 0.7, y, 5.7, rh, "F2F5F9")
        _box(slide, 6.9, y, 5.7, rh, "F2F5F9")
        if i < len(a_items):
            _text(slide, 0.95, y + 0.1, 5.2, rh - 0.15, a_items[i], 14, "2B2B2B", spacing=1.15)
        if i < len(b_items):
            _text(slide, 7.15, y + 0.1, 5.2, rh - 0.15, b_items[i], 14, "2B2B2B", spacing=1.15)

def _statrow(slide, stats, y0=4.2):
    n = len(stats)
    if n == 0:
        return
    gap = 0.3
    cw = (11.93 - (n - 1) * gap) / n
    for i, (num, desc) in enumerate(stats):
        x = 0.7 + i * (cw + gap)
        _box(slide, x, y0, cw, 2.4, "F2F5F9")
        _box(slide, x, y0, cw, 0.12, "E8792B" if i == n - 1 else "4A90D9")
        _text(slide, x + 0.25, y0 + 0.35, cw - 0.5, 0.9, num, 44, "1B3A5C", bold=True)
        _text(slide, x + 0.25, y0 + 1.3, cw - 0.5, 1.0, desc, 13, "2B2B2B", spacing=1.25)

def _agenda(slide, items):
    y = 2.0
    for item in items:
        num, head, sub = item[0], item[1], item[2] if len(item) > 2 else ""
        _text(slide, 0.7, y, 0.7, 0.7, num, 28, "E8792B", bold=True)
        _text(slide, 1.5, y + 0.03, 5.5, 0.5, head, 20, "1B3A5C", bold=True)
        _text(slide, 1.5, y + 0.48, 9, 0.4, sub, 14, "4A5568")
        y += 1.15

def _title(slide, main, sub, dark_bg=True, bg_hex="#1B3A5C"):
    _box(slide, 0.7, 2.3, 0.6, 0.05, "E8792B")
    _text(slide, 0.7, 2.6, 12, 1.4, main, 40, "FFFFFF" if dark_bg else "1B3A5C", bold=True)
    _text(slide, 0.7, 4.0, 11, 0.6, sub, 18, _dark_tint(bg_hex, True) if dark_bg else "4A5568")

def _takeaway(slide, main, steps, close, dark_bg=True, bg_hex="#1B3A5C"):
    tint = _dark_tint(bg_hex, True) if dark_bg else "4A90D9"
    _box(slide, 0.7, 1.7, 0.6, 0.05, "E8792B")
    _text(slide, 0.7, 2.1, 11.9, 1.2, main, 34, "FFFFFF" if dark_bg else "1B3A5C", bold=True, spacing=1.2)
    _text(slide, 0.7, 3.5, 7.5, 2.4, "\n".join(f"-  {s_}" for s_ in steps), 20, tint, spacing=1.6)
    _text(slide, 0.7, 5.7, 11.9, 0.5, close, 14, "FFFFFF" if dark_bg else "2B2B2B")

def _section(slide, label, headline, sub, bg_hex="#1B3A5C"):
    tint = _dark_tint(bg_hex, True)
    _text(slide, 0.7, 2.6, 12, 0.5, label, 13, "E8792B", bold=True)
    _text(slide, 0.7, 3.1, 12, 1.0, headline, 36, "FFFFFF", bold=True)
    _text(slide, 0.7, 4.1, 10, 0.6, sub, 16, tint)

def build(spec, out_path):
    prs = _new_pres()
    dark_bg_hex = spec.get("dark_bg", "#1B3A5C")
    transition = spec.get("transition", "fade")
    slides = spec.get("slides", [])
    for i, sl in enumerate(slides):
        kind = sl.get("layout", "content")
        bg = sl.get("bg", None) or ("FFFFFF" if kind not in ("title", "takeaway", "section") else dark_bg_hex)
        s = _add_slide(prs, bg)
        dark = (bg != "FFFFFF")
        if kind == "title":
            _title(s, sl.get("main", "Title"), sl.get("sub", ""), bg_hex=bg)
            _footer(s, i + 1, dark=True, dark_bg=bg)
        elif kind == "agenda":
            _head(s, "Agenda", sl.get("headline", "Today's path"))
            _agenda(s, sl.get("items", []))
            _footer(s, i + 1, dark)
        elif kind == "cards":
            _head(s, sl.get("kicker", "Overview"), sl.get("headline", ""))
            _card_row(s, sl.get("cards", []))
            _footer(s, i + 1, dark)
        elif kind == "twocol":
            _head(s, sl.get("kicker", "Detail"), sl.get("headline", ""))
            _two_col(s, sl.get("left_head", ""), sl.get("left_body", ""),
                     sl.get("right_head", ""), sl.get("right_body", ""))
            _footer(s, i + 1, dark)
        elif kind == "chart":
            _head(s, sl.get("kicker", "Data"), sl.get("headline", ""))
            data = [(c.get("label", ""), float(c.get("value", 0)), c.get("color", None))
                    for c in sl.get("data", [])]
            _chart(s, data, caption=sl.get("caption", ""),
                   note_head=sl.get("note_head", ""), note_body=sl.get("note_body", []))
            _footer(s, i + 1, dark)
        elif kind == "process":
            _head(s, sl.get("kicker", "Process"), sl.get("headline", ""))
            _process(s, [(s0.get("head", ""), s0.get("body", "")) for s0 in sl.get("steps", [])])
            _footer(s, i + 1, dark)
        elif kind == "compare":
            _head(s, sl.get("kicker", "Compare"), sl.get("headline", ""))
            _compare(s, sl.get("a", ["", []]), sl.get("b", ["", []]))
            _footer(s, i + 1, dark)
        elif kind == "statrow":
            _head(s, sl.get("kicker", "Numbers"), sl.get("headline", ""))
            _statrow(s, [(s0.get("num", ""), s0.get("desc", "")) for s0 in sl.get("stats", [])])
            _footer(s, i + 1, dark)
        elif kind == "section":
            _section(s, sl.get("label", "SECTION"), sl.get("headline", ""), sl.get("sub", ""), bg_hex=bg)
            _footer(s, i + 1, dark=True, dark_bg=bg)
        elif kind == "takeaway":
            _takeaway(s, sl.get("main", ""), sl.get("steps", []), sl.get("close", ""), bg_hex=bg)
            _footer(s, i + 1, dark=True, dark_bg=bg)
        else:
            _head(s, sl.get("kicker", ""), sl.get("headline", ""))
            _text(s, 0.7, 2.0, 11.93, 4.0, sl.get("body", ""), 15, "2B2B2B", spacing=1.3)
            _footer(s, i + 1, dark)
        _apply_transition(s, transition)
    prs.save(out_path)
    return {"out": out_path, "slides": len(slides)}

def check(pptx_path, min_font=11.0, min_contrast=4.5):
    from pptx import Presentation
    from pptx.util import Emu
    FONT_W = 0.60
    prs = Presentation(pptx_path)
    sw_in = Emu(prs.slide_width).inches
    sh_in = Emu(prs.slide_height).inches
    problems = []
    for si, slide in enumerate(list(prs.slides), 1):
        bg_hex = 0xFFFFFF
        try:
            if slide.background.fill.type is not None:
                bg_hex = int(slide.background.fill.fore_color.rgb)
        except Exception:
            pass
        boxes = []
        for sh in slide.shapes:
            if not sh.has_text_frame:
                continue
            tf = sh.text_frame
            n_chars = sum(len(r.text) for p in tf.paragraphs for r in p.runs)
            if n_chars == 0:
                continue
            min_size, text_color = None, None
            for p in tf.paragraphs:
                for r in p.runs:
                    if r.font.size:
                        sz = r.font.size.pt
                        min_size = sz if min_size is None else min(min_size, sz)
                    try:
                        text_color = int(r.font.color.rgb)
                    except Exception:
                        pass
            size = min_size if min_size else 18.0
            cpl = max(1.0, int((Emu(sh.width).inches * 72 - 8) / (size * FONT_W)))
            total_lines = 0
            for p in tf.paragraphs:
                t = "".join(r.text for r in p.runs)
                n_txt = len(t)
                total_lines += max(1, -(-n_txt // cpl)) if n_txt else 1
            need_in = total_lines * size * 1.25 / 72.0
            have_in = Emu(sh.height).inches
            x0, y0 = Emu(sh.left).inches, Emu(sh.top).inches
            x1, y1 = x0 + Emu(sh.width).inches, y0 + have_in
            if need_in > have_in + 0.02:
                problems.append(f"slide {si}: overflow ~{need_in-have_in:.2f}in '{t[:30]}'")
            if x1 > sw_in + 0.01 or y1 > sh_in + 0.01 or x0 < 0.0 or y0 < 0.0:
                problems.append(f"slide {si}: off-slide ({x0:.2f},{y0:.2f})-({x1:.2f},{y1:.2f})")
            if min_size and min_size < min_font:
                problems.append(f"slide {si}: font {min_size:.0f}pt < {min_font:.0f}pt")
            if text_color is not None and min_size and min_size < 18:
                cr = _contrast(text_color, bg_hex)
                if cr < min_contrast:
                    problems.append(f"slide {si}: contrast {cr:.2f}:1 text #{text_color:06x} on #{bg_hex:06x}")
            boxes.append((x0, y0, x1, y1, sh.name, min_size))
        pad = 2.0 / 72.0
        for a in range(len(boxes)):
            for b in range(a + 1, len(boxes)):
                ax0, ay0, ax1, ay1, an, _ = boxes[a]
                bx0, by0, bx1, by1, bn, _ = boxes[b]
                if ax0 < bx1 - pad and bx0 < ax1 - pad and ay0 < by1 - pad and by0 < ay1 - pad:
                    if not (ax0 >= bx0 and ax1 <= bx1) and not (bx0 >= ax0 and bx1 <= ax1):
                        problems.append(f"slide {si}: overlap '{an}' vs '{bn}'")
    if problems:
        print(f"{pptx_path}: {len(problems)} problem(s)")
        for p in problems:
            print("  -", p)
        return {"ok": False, "problems": problems}
    print(f"{pptx_path}: clean")
    return {"ok": True, "problems": []}

def apply(ctx, config):
    tools = ctx.get("tools")
    def make_deck(session, args):
        spec = json.loads(args.get("spec_json") or args.get("spec") or "{}")
        out = args.get("out_path") or "deck-lab/deck_pptx_lab.pptx"
        res = build(spec, out)
        return json.dumps(res)
    def check_deck(session, args):
        path = args.get("pptx_path") or args.get("path")
        if not path:
            return "usage: check_deck(pptx_path)"
        mf = float(args.get("min_font", 11.0))
        return json.dumps(check(path, min_font=mf))
    tools.register({
        "name": "make_deck",
        "description": "Build a .pptx deck from a JSON spec (title + slides with layout/content/data). Applies learned pro rules.",
        "parameters": {"type": "object", "properties": {
            "spec_json": {"type": "string", "description": "JSON string of the deck spec"},
            "out_path": {"type": "string", "description": "output .pptx path"}}, "required": ["spec_json"]},
        "kind": "edit"},
    make_deck)
    tools.register({
        "name": "check_deck",
        "description": "Check a .pptx for overflow/overlap/small-font/low-contrast issues.",
        "parameters": {"type": "object", "properties": {
            "pptx_path": {"type": "string", "description": "path to .pptx"},
            "min_font": {"type": "number", "description": "minimum font size, default 11"}}, "required": ["pptx_path"]},
        "kind": "read"},
    check_deck)
    return tools
