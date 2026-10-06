# pptx_lab v1: make_pptx + check_pptx built from deck-lab learnings.
# make_pptx: JSON spec -> .pptx (16:9 grid, verified palettes, 6 layouts incl. charts, fade transitions).
# check_pptx: overflow, overlap, font size, contrast, margins, density, transition audit.

import json, math, os, re, zipfile
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE

PALETTES = {
    "light": dict(bg=(0xF7, 0xF9, 0xFC), ink=(0x1F, 0x24, 0x30), accent=(0x0F, 0x62, 0xFE),
                  panel=(0xE8, 0xEE, 0xF9), muted=(0x5B, 0x64, 0x74)),
    "dark": dict(bg=(0x0F, 0x14, 0x20), ink=(0xE8, 0xED, 0xF5), accent=(0x4D, 0x9F, 0xFF),
                 panel=(0x1B, 0x24, 0x36), muted=(0xA8, 0xB4, 0xC8)),
    "warm": dict(bg=(0xFB, 0xF7, 0xEF), ink=(0x2A, 0x23, 0x1C), accent=(0xC4, 0x57, 0x1A),
                 panel=(0xEF, 0xE6, 0xD8), muted=(0x5C, 0x4F, 0x3E)),
}
FONT = "Segoe UI"
SW, SH, MARGIN = 13.333, 7.5, 0.6

def rgb(t): return RGBColor(*t)

def _lum(t):
    r, g, b = [v / 255.0 for v in t]
    f = lambda x: x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4
    r, g, b = f(r), f(g), f(b)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def contrast(c1, c2):
    l1, l2 = sorted([_lum(c1), _lum(c2)], reverse=True)
    return (l1 + 0.05) / (l2 + 0.05)

def _rect(s, x, y, w, h, color, radius=False):
    st = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    sh = s.shapes.add_shape(st, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb = color
    sh.line.fill.background(); sh.shadow.inherit = False
    return sh

def _text(s, x, y, w, h, paras, size=16, color=None, bold=False, align=PP_ALIGN.LEFT,
          line_spacing=1.0, space_after=6):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    first = True
    for para in paras:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        d = para if isinstance(para, dict) else dict(t=para, size=size, color=color, bold=bold)
        p.alignment = d.get("align", align); p.line_spacing = line_spacing
        p.space_after = Pt(space_after)
        r = p.add_run(); r.text = d.get("t", "")
        r.font.name = FONT; r.font.size = Pt(d.get("size", size)); r.font.bold = d.get("bold", bold)
        dc = d.get("color", color)
        if dc:
            r.font.color.rgb = rgb(dc) if isinstance(dc, (tuple, list)) else dc
    return tb

def _header(s, P, kicker, title):
    _text(s, MARGIN, 0.42, SW - 2 * MARGIN, 0.3, [dict(t=(kicker or "").upper(), size=13,
          color=P["accent"], bold=True)], space_after=0)
    _text(s, MARGIN, 0.72, SW - 2 * MARGIN, 0.75, [dict(t=title, size=32, bold=True)], space_after=0)
    _rect(s, MARGIN, 1.45, 0.55, 0.055, P["accent"])

def _footer(s, P, idx, total):
    _text(s, MARGIN, SH - 0.45, SW - 2 * MARGIN, 0.3,
          [dict(t=f"{idx} / {total}", size=11, color=P["muted"], align=PP_ALIGN.RIGHT)], space_after=0)

def _layout_title(s, P, spec):
    _rect(s, 0, 0, 0.18, SH, P["accent"])
    _text(s, MARGIN + 0.4, 2.5, SW - 2 * MARGIN - 0.4, 0.4,
          [dict(t=(spec.get("kicker") or "DECK-LAB STUDY").upper(), size=14, color=P["accent"], bold=True)],
          space_after=0)
    _text(s, MARGIN + 0.4, 3.0, SW - 2 * MARGIN - 0.4, 1.9,
          [dict(t=spec["title"], size=46, bold=True)], line_spacing=1.05, space_after=0)
    if spec.get("subtitle"):
        _text(s, MARGIN + 0.4, 5.0, SW - 2 * MARGIN - 0.8, 0.9,
               [dict(t=spec["subtitle"], size=17, color=P["muted"])], line_spacing=1.15, space_after=0)

def _layout_cards(s, P, spec):
    _header(s, P, spec.get("kicker", ""), spec["title"])
    cards = spec.get("cards", [])
    if not cards or len(cards) > 3:
        raise ValueError("cards layout needs 1-3 cards")
    n = len(cards)
    cw = (SW - 2 * MARGIN - 0.37 * (n - 1)) / n if n > 1 else SW - 2 * MARGIN
    height = 2.5
    for i, c in enumerate(cards):
        x = MARGIN + i * (cw + 0.37)
        _rect(s, x, 1.9, cw, height, P["panel"], radius=True)
        _text(s, x + 0.3, 2.15, cw - 0.6, 0.5, [dict(t=c["title"], size=18, bold=True)], space_after=0)
        _text(s, x + 0.3, 2.7, cw - 0.6, height - 0.9,
               [dict(t=c.get("body", ""), size=15, color=P["muted"])], line_spacing=1.15, space_after=0)
    if spec.get("footnote"):
        _text(s, MARGIN, SH - 0.95, SW - 2 * MARGIN, 0.4,
               [dict(t=spec["footnote"], size=14, color=P["accent"], bold=True)], space_after=0)

def _layout_steps(s, P, spec):
    _header(s, P, spec.get("kicker", ""), spec["title"])
    steps = spec.get("steps", [])
    if not steps or len(steps) > 4:
        raise ValueError("steps layout needs 1-4 steps")
    n = len(steps)
    gap = 0.25
    cw = (SW - 2 * MARGIN - gap * (n - 1)) / n if n > 1 else SW - 2 * MARGIN
    y, h = 1.85, min(4.4, (SH - 1.85 - 0.75) / max(1, n))
    for i, st in enumerate(steps):
        x = MARGIN + i * (cw + gap)
        _rect(s, x, y, cw, h, P["panel"], radius=True)
        _rect(s, x + 0.25, y + 0.25, 0.45, 0.45, P["accent"], radius=True)
        _text(s, x + 0.22, y + 0.28, 0.5, 0.4, [dict(t=str(i + 1), size=15, color=P["bg"],
              bold=True, align=PP_ALIGN.CENTER)], space_after=0)
        _text(s, x + 0.85, y + 0.3, cw - 1.0, 0.5, [dict(t=st["title"], size=16, bold=True)], space_after=0)
        _text(s, x + 0.85, y + 0.85, cw - 1.0, h - 1.0,
               [dict(t=st.get("body", ""), size=14, color=P["muted"])], line_spacing=1.15, space_after=0)

def _layout_compare(s, P, spec):
    _header(s, P, spec.get("kicker", ""), spec["title"])
    a, b = spec["cols"]
    cw = (SW - 2 * MARGIN - 0.5) / 2
    for i, col in enumerate([a, b]):
        x = MARGIN + i * (cw + 0.5)
        _rect(s, x, 1.9, cw, 4.3, P["panel"], radius=True)
        _rect(s, x, 1.9, cw, 0.62, P["accent"])
        _text(s, x + 0.25, 1.99, cw - 0.5, 0.5, [dict(t=col["title"], size=17, color=P["bg"],
              bold=True)], space_after=0)
        paras = [dict(t=t, size=14, color=P["muted"]) for t in col.get("items", [])]
        _text(s, x + 0.25, 2.75, cw - 0.5, 3.2, paras, line_spacing=1.2, space_after=9)

def _layout_chart(s, P, spec):
    _header(s, P, spec.get("kicker", ""), spec["title"])
    ch = spec["chart"]
    data = CategoryChartData()
    data.categories = ch.get("categories", [])
    for series in ch.get("series", []):
        data.add_series(series.get("name", "series"), series["values"])
    xmap = {"column": XL_CHART_TYPE.COLUMN_CLUSTERED, "bar": XL_CHART_TYPE.BAR_CLUSTERED,
            "line": XL_CHART_TYPE.LINE, "pie": XL_CHART_TYPE.PIE}
    ctype = ch.get("type", "column")
    if ctype not in xmap:
        raise ValueError("chart type must be column, bar, line or pie")
    gf = s.shapes.add_chart(xmap[ctype], Inches(MARGIN), Inches(1.9),
                            Inches(SW - 2 * MARGIN), Inches(4.2), data)
    chart = gf.chart
    chart.font.size = Pt(12); chart.font.name = FONT; chart.font.color.rgb = rgb(P["ink"])
    chart.has_title = bool(ch.get("title", False))
    if ch.get("title"):
        chart.chart_title.text_frame.text = ch["title"]
    try:
        for plot in list(chart.plots):
            for sidx, ser in enumerate(plot.series):
                ser.format.fill.solid()
                ser.format.fill.fore_color.rgb = rgb(P["accent"] if sidx % 2 == 0 else P["muted"])
    except Exception:
        pass
    if ch.get("caption"):
        _text(s, MARGIN, SH - 0.85, SW - 2 * MARGIN, 0.35,
               [dict(t=ch["caption"], size=12, color=P["muted"])], space_after=0)

def _layout_stats(s, P, spec):
    _header(s, P, spec.get("kicker", ""), spec["title"])
    stats = spec.get("stats", [])
    if not stats or len(stats) > 4:
        raise ValueError("stats layout needs 1-4 stats")
    n = len(stats)
    gap = 0.28
    cw = (SW - 2 * MARGIN - gap * (n - 1)) / n if n > 1 else SW - 2 * MARGIN
    for i, st in enumerate(stats):
        x = MARGIN + i * (cw + gap)
        _rect(s, x, 1.9, cw, 1.7, P["panel"], radius=True)
        _text(s, x + 0.25, 2.05, cw - 0.5, 0.6, [dict(t=st["value"], size=28, color=P["accent"],
              bold=True)], space_after=0)
        _text(s, x + 0.25, 2.7, cw - 0.5, 0.8, [dict(t=st["label"], size=13, color=P["muted"])],
              line_spacing=1.1, space_after=0)
    if spec.get("note"):
        _text(s, MARGIN, 4.3, SW - 2 * MARGIN, 2.2, [dict(t=spec["note"], size=16)],
              line_spacing=1.25, space_after=10)

LAYOUTS = {"title": _layout_title, "cards": _layout_cards, "steps": _layout_steps,
           "compare": _layout_compare, "chart": _layout_chart, "stats": _layout_stats}

FADE = ('<p:transition xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" '
        'spd="med"><p:fade/></p:transition>')

def _add_transitions(path):
    zin = zipfile.ZipFile(path)
    entries = zin.namelist()
    slides = [n for n in entries if re.match(r"ppt/slides/slide\d+\.xml$", n)]
    out = zipfile.ZipFile(path + ".tmp", "w", zipfile.ZIP_DEFLATED)
    for name in entries:
        data = zin.read(name)
        if name in slides:
            xml = data.decode("utf-8")
            m = re.search(r"</p:cSld>\s*", xml)
            if m and "<p:transition" not in xml:
                xml = xml[:m.end()] + FADE + "\n" + xml[m.end():]
            data = xml.encode("utf-8")
        out.writestr(name, data)
    out.close(); zin.close()
    os.replace(path + ".tmp", path)

def make_pptx(session, args):
    """Build a .pptx from a JSON spec. Spec keys:
    deck (output filename, e.g. "deck-lab/out/p.pptx"), theme: light|dark|warm,
    transitions (default true), slides: list of {layout, title, ...}.
    Layouts: title {title, kicker, subtitle}; cards {title, kicker, cards:[{title, body}], footnote};
    steps {title, kicker, steps:[{title, body}]}; compare {title, kicker, cols:[{title, items:[]}x2]};
    chart {title, kicker, chart:{type: column|bar|line|pie, categories:[] , series:[{name, values:[]}], caption}};
    stats {title, kicker, stats:[{value, label}], note}. Applies the deck-lab grid and palettes."""
    spec = json.loads(args) if isinstance(args, str) else args
    theme = spec.get("theme", "light")
    if theme not in PALETTES:
        return f"error: unknown theme '{theme}' (use light, dark or warm)"
    P = {k: tuple(v) for k, v in PALETTES[theme].items()}
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(SW), Inches(SH)
    slides = spec.get("slides", [])
    if not slides:
        return "error: spec has no slides"
    total = len(slides)
    for i, ss in enumerate(slides):
        layout = LAYOUTS.get(ss.get("layout", "cards"))
        if layout is None:
            return f"error: slide {i+1} unknown layout '{ss.get('layout')}'"
        s = prs.slides.add_slide(prs.slide_layouts[6])
        s.background.fill.solid()
        s.background.fill.fore_color.rgb = rgb(P["bg"])
        try:
            layout(s, P, ss)
        except KeyError as e:
            return f"error: slide {i+1} ({ss.get('layout')}): missing key {e}"
        except ValueError as e:
            return f"error: slide {i+1} ({ss.get('layout')}): {e}"
        if ss.get("layout") != "title":
            _footer(s, P, i + 1, total)
    out = spec.get("deck", "deck.pptx")
    out = os.path.join(os.getcwd(), out) if not os.path.isabs(out) else out
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    prs.save(out)
    if spec.get("transitions", True):
        _add_transitions(out)
    return f"wrote {out} ({total} slides, theme={theme}, transitions={spec.get('transitions', True)})"

def check_pptx(session, args):
    """Check a .pptx: text overflow, text-box overlap, fonts under 12pt, contrast below
    WCAG (4.5 body / 3.0 large), margin breaches, slide density (>8 text blocks), and
    presence of fade transitions / entry animation. args: {path} or a path string."""
    path = args["path"] if isinstance(args, dict) else args
    path = os.path.join(os.getcwd(), path) if not os.path.isabs(path) else path
    if not os.path.exists(path):
        return f"error: not found: {path}"
    prs = Presentation(path)
    EMU_IN = 914400
    issues, info = [], []
    for i, slide in enumerate(prs.slides, 1):
        shapes = [s for s in slide.shapes if s.top is not None and s.left is not None]
        for s in shapes:
            x, y = s.left / EMU_IN, s.top / EMU_IN
            w, h = s.width / EMU_IN, s.height / EMU_IN
            if x < -0.01 or y < -0.01 or x + w > SW + 0.02 or y + h > SH + 0.02:
                issues.append(f"s{i}: out of bounds ({x:.2f},{y:.2f} {w:.2f}x{h:.2f})")
            if s.has_text_frame and s.text_frame.text.strip() and x + w > SW - MARGIN + 0.15:
                issues.append(f"s{i}: text box near right edge (ends {x+w:.2f}in)")
        tboxes = [s for s in shapes if s.has_text_frame and s.text_frame.text.strip()]
        for a in tboxes:
            for b in tboxes:
                if a is b:
                    continue
                iw = (min(a.left + a.width, b.left + b.width) - max(a.left, b.left)) / EMU_IN
                ih = (min(a.top + a.height, b.top + b.height) - max(a.top, b.top)) / EMU_IN
                if iw > 0.15 and ih > 0.15:
                    issues.append(f"s{i}: overlap '{a.text_frame.text[:18]}' / '{b.text_frame.text[:18]}'")
        for s in shapes:
            if not s.has_text_frame or not s.text_frame.text.strip():
                continue
            need = 0.0
            for p in s.text_frame.paragraphs:
                runs = list(p.runs)
                pt = max((r.font.size.pt if r.font.size else 16) for r in runs) if runs else 16
                bold = any(r.font.bold for r in runs)
                t = "".join(r.text for r in runs)
                cw = pt * (0.56 if bold else 0.5) / 72.0
                per_line = max(1, int((s.width / EMU_IN) / cw))
                lines = sum(max(1, math.ceil(len(seg) / per_line)) for seg in t.split("\n"))
                ls = p.line_spacing if isinstance(p.line_spacing, float) else 1.0
                need += lines * pt * 1.3 * ls / 72.0 + (p.space_after.pt / 72.0 if p.space_after else 0)
            if need > s.height / EMU_IN + 0.05:
                issues.append(f"s{i}: overflow '{s.text_frame.text[:28]}' needs ~{need:.2f}in, box {s.height/EMU_IN:.2f}in")
            for p in s.text_frame.paragraphs:
                for r in p.runs:
                    if r.font.size and r.font.size.pt < 12:
                        issues.append(f"s{i}: font {r.font.size.pt}pt < 12 ('{r.text[:18]}')")
        if len(tboxes) > 8:
            issues.append(f"s{i}: dense ({len(tboxes)} text blocks); split the slide")
        for s in tboxes:
            bg, best = None, None
            for sh2 in shapes:
                if sh2 is s or sh2.top is None or sh2.left is None:
                    continue
                try:
                    if sh2.fill.type != 1:
                        continue
                except Exception:
                    continue
                x1, y1 = sh2.left / EMU_IN, sh2.top / EMU_IN
                if (x1 <= s.left / EMU_IN and s.left / EMU_IN + s.width / EMU_IN <= x1 + sh2.width / EMU_IN + 0.01
                        and y1 <= s.top / EMU_IN and s.top / EMU_IN + s.height / EMU_IN <= y1 + sh2.height / EMU_IN + 0.01):
                    area = (sh2.width / EMU_IN) * (sh2.height / EMU_IN)
                    if best is None or area < best:
                        best, bg = area, sh2.fill.fore_color.rgb
            if bg is None:
                try:
                    bg = tuple(slide.background.fill.fore_color.rgb)
                except Exception:
                    bg = PALETTES["light"]["bg"]
            for p in s.text_frame.paragraphs:
                for r in p.runs:
                    if not r.text.strip():
                        continue
                    try:
                        fg = tuple(r.font.color.rgb)
                    except Exception:
                        continue
                    sz = r.font.size.pt if r.font.size else 16
                    cr = contrast(fg, bg)
                    minr = 3.0 if sz >= 18 else 4.5
                    if cr < minr:
                        issues.append(f"s{i}: contrast {cr:.2f} < {minr} '{r.text[:18]}' (fg={fg})")
    z = zipfile.ZipFile(path)
    sfiles = [n for n in z.namelist() if re.match(r"ppt/slides/slide\d+\.xml$", n)]
    ntrans = sum(1 for n in sfiles if b"<p:transition" in z.read(n))
    nanim = sum(1 for n in sfiles if b"<p:timing" in z.read(n))
    z.close()
    if ntrans:
        info.append(f"fade transitions on {ntrans}/{len(sfiles)} slides")
    else:
        issues.append(f"no fade transitions on any of {len(sfiles)} slides (expected by default)")
    info.append(f"entry animation on {nanim}/{len(sfiles)} slides" + (" (optional)" if nanim == 0 else ""))
    print(f"checked {i} slides in {path}")
    for x in issues + [f"info: {x}" for x in info]:
        print((" !" if x.startswith("s") or x.startswith("no ") else " i"), x)
    if not issues:
        print("clean: no overflow/overlap/size/contrast/margin/density issues")
    return "clean" if not issues else f"{len(issues)} issue(s) - see lines above"

inject = ["tools"]

defaults = {}

def apply(ctx, config):
    ctx.get("tools").register(
        dict(name="make_pptx",
             description=("Build a .pptx from a JSON spec: {deck, theme: light|dark|warm, transitions: true|false, "
                          "slides: [{layout: title|cards|steps|compare|chart|stats, title, kicker, ...}]} and "
                          "write the file with deck-lab grid, palettes and size rules, plus fade transitions. "
                          "Returns the written path or an error."),
             kind="edit"),
        make_pptx, "make_pptx")
    ctx.get("tools").register(
        dict(name="check_pptx",
             description=("Check a .pptx for text overflow, shape overlap, fonts under 12pt, WCAG contrast, "
                          "margin breaches, slide density and whether fade transitions / entry animation are "
                          "present. args: {path} or a path string. Returns 'clean' or an issue count with lines."),
             kind="read"),
        check_pptx, "check_pptx")
