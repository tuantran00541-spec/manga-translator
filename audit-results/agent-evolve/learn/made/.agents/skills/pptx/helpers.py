"""Hàm dùng chung theo SKILL.md (mục 4, 6) — dựng slide bằng python-pptx 16:9.

Các bộ slide ngày 2 (a7, a8) chỉ dùng helper trong file này để chứng minh skill
chạy được; không viết lại logic mỗi lần."""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from lxml import etree
from pptx.oxml.ns import qn

# Nạp 2 layout picture và table (nâng cấp ngày 3, định nghĩa dưới cùng file này)

WHITE = RGBColor(0xFF, 0xFF, 0xFF)
INK = RGBColor(0x1A, 0x1A, 0x1A)
NAVY = RGBColor(0x1F, 0x2A, 0x44)
BLUE = RGBColor(0x0B, 0x5E, 0xD7)
TEAL = RGBColor(0x0F, 0x7D, 0x70)
GRAY = RGBColor(0x59, 0x59, 0x59)
LIGHT = RGBColor(0xF2, 0xF5, 0xFA)
LINEC = RGBColor(0xD0, 0xD9, 0xE8)
MUTED = RGBColor(0x5A, 0x6B, 0x85)  # day-1 fix: white-on 5.4:1
LNAVY1 = RGBColor(0xB0, 0xC4, 0xDE)
LNAVY2 = RGBColor(0x8F, 0xA3, 0xC7)
FONT = 'DejaVu Sans'


def new_prs():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    return prs


def add_slide(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def fade(slide):
    el = slide._element
    t = el.find(qn('p:transition'))
    if t is None:
        t = etree.Element(qn('p:transition'))
        anchor = el.find(qn('p:clrMapOvr'))
        el.insert(list(el).index(anchor), t)
    for ch in list(t):
        t.remove(ch)
    etree.SubElement(t, qn('p:fade'), attrib={'spd': 'med'})


def txbox(slide, left, top, width, height, text, size, color=INK, bold=False,
          align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, spacing=1.0):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Emu(0)
    tf.margin_top = tf.margin_bottom = Emu(0)
    for i, line in enumerate(text.split('\n')):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = spacing
        r = p.add_run()
        r.text = line
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.color.rgb = color
        r.font.name = FONT
    return box


def rect(slide, left, top, width, height, fill, line=None, shape=MSO_SHAPE.RECTANGLE):
    sp = slide.shapes.add_shape(shape, Inches(left), Inches(top), Inches(width), Inches(height))
    sp.fill.solid()
    sp.fill.fore_color.rgb = fill
    if line is None:
        sp.line.fill.background()
    else:
        sp.line.color.rgb = line
    return sp


def slide_title(slide, text):
    """Action title theo SKILL.md mục 1: top 0.5in, cao 0.8in, 30pt bold, căn trái."""
    return txbox(slide, 0.7, 0.5, 11.9, 0.8, text, 30, color=INK, bold=True)


def source_line(slide, text):
    """Dòng nguồn/chú thích: 12pt, xám, vùng an toàn cuối slide (SKILL.md mục 1)."""
    return txbox(slide, 0.7, 6.7, 11.9, 0.4, text, 12, color=GRAY)


# ---- 8 mẫu slide (SKILL.md mục 4) ----

def slide_cover(prs, title, subtitle, footer, bg=NAVY):
    s = add_slide(prs)
    rect(s, 0, 0, 13.333, 7.5, bg)
    # Tiêu đề đặt cao hơn và cao hơn box (1.6in) để 3 dòng wrap không đè subtitle (vòng 3 ngày 4)
    txbox(s, 0.7, 1.9, 11.9, 1.6, title, 36, color=WHITE, bold=True)
    txbox(s, 0.7, 3.9, 11.9, 0.8, subtitle, 22, color=LNAVY1)
    txbox(s, 0.7, 6.7, 11.9, 0.5, footer, 14, color=LNAVY2)
    fade(s)
    return s


def slide_agenda(prs, title, items, accent=BLUE):
    s = add_slide(prs)
    slide_title(s, title)
    for i, item in enumerate(items[:6]):
        top = 1.8 + i * 0.95
        txbox(s, 0.8, top, 0.9, 0.7, f'{i + 1:02d}', 24, color=accent, bold=True)
        txbox(s, 1.9, top + 0.02, 9.5, 0.7, item, 20, color=INK, anchor=MSO_ANCHOR.MIDDLE)
    fade(s)
    return s


def slide_cards(prs, title, cards, accent=BLUE, bg=LIGHT):
    s = add_slide(prs)
    slide_title(s, title)
    for i, (big, label, desc) in enumerate(cards[:3]):
        left = 0.7 + i * 4.13
        rect(s, left, 1.7, 3.8, 4.4, bg, LINEC)
        # (vòng 7 ngày 6) ô số lớn: chữ 48pt wrap 2 dòng (như "Quá nhiều dòng") cần
        # ~1.6in chiều cao; trước đây box chỉ cao 1.2in nên chữ wrap vượt đáy box, che
        # ô nhãn ngay dưới (bẫy thật mà check thứ 9 ngày 5 phát hiện trên a18). Đặt cao
        # hơn 0.4in, hạ ô nhãn xuống 3.7in (trước đó 3.3in) cho đủ chỗ.
        txbox(s, left + 0.3, 2.0, 3.2, 1.6, big, 48, color=accent, bold=True,
               align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        txbox(s, left + 0.3, 3.7, 3.2, 0.5, label, 18, color=INK, bold=True,
               align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        # (vòng 5 ngày 4) mô tả: cao hơn (1.8in → 2.0in) và đặt sát đáy card (hết card 6.1in)
        # để chữ wrap nhiều dòng vẫn nằm trong card, không che card kế bên.
        txbox(s, left + 0.3, 4.1, 3.2, 2.0, desc, 14, color=GRAY, spacing=1.2)
    fade(s)
    return s


def slide_twocol(prs, title, cols, source=None):
    s = add_slide(prs)
    slide_title(s, title)
    for i, col in enumerate(cols[:2]):
        head, pts = col[0], col[1:]
        left = 0.7 + i * 6.2
        rect(s, left, 1.6, 5.6, 0.8, NAVY if i == 1 else MUTED)
        txbox(s, left + 0.3, 1.7, 5.0, 0.6, head, 24, color=WHITE, bold=True,
               anchor=MSO_ANCHOR.MIDDLE)
        for j, pt in enumerate(pts[:3]):
            txbox(s, left + 0.3, 2.8 + j * 0.9, 5.0, 0.7, '· ' + str(pt), 18, color=INK)
    if source:
        source_line(s, source)
    fade(s)
    return s


def slide_process(prs, title, steps, accent=BLUE):
    s = add_slide(prs)
    slide_title(s, title)
    steps = steps[:6]
    n = len(steps)
    # Khoảng cách động để 6 bước (tối đa) luôn nằm trong khung 13.333in.
    usable_w = 11.93
    step_w = usable_w / n
    circ_w = min(1.2, step_w * 0.6)
    for i, st in enumerate(steps):
        left = 0.7 + i * step_w + (step_w - circ_w) / 2
        circ = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(left), Inches(2.2), Inches(circ_w), Inches(circ_w))
        circ.fill.solid()
        circ.fill.fore_color.rgb = accent
        circ.line.fill.background()
        txbox(s, left, 2.2, circ_w, circ_w, str(i + 1), 30, color=WHITE, bold=True,
               align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        if i < n - 1:
            x2 = 0.7 + (i + 1) * step_w + ((step_w - circ_w) / 2)
            conn = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,
                                         Inches(left + circ_w), Inches(2.8),
                                         Inches(x2), Inches(2.8))
            conn.line.color.rgb = LINEC
            conn.line.width = Pt(2)
        txbox(s, left + (circ_w - 2.35) / 2 + 0.15, 3.9, 2.35, 1.4, st, 16, color=INK,
               align=PP_ALIGN.CENTER, spacing=1.2)
    fade(s)
    return s


def slide_barchart(prs, title, data, accent=BLUE, source=None, ymax=None):
    s = add_slide(prs)
    slide_title(s, title)
    base_y, max_h, chart_left, chart_w = 5.4, 3.0, 0.9, 11.5
    ymax = ymax or max(v for _, v in data)
    n = len(data)
    slot = chart_w / n
    for i, (label, val) in enumerate(data):
        cx = chart_left + i * slot + slot * 0.15
        bw = slot * 0.7
        bh = val / ymax * max_h
        by = base_y - bh
        rect(s, cx, by, bw, bh, accent if i == n - 1 else MUTED)
        # Nhãn giá trị: đặt trong thân bar (chữ trắng, căn giữa) khi bar đủ cao,
        # trên bar (chữ INK) khi bar ngắn — tránh chữ đè lên nhau (bẫy ngày 4, vòng 1).
        if bh > 0.9:
            txbox(s, cx, by + 0.05, bw, 0.4, f'{val}', 16, color=WHITE, bold=True,
                   align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        else:
            txbox(s, cx, by - 0.45, bw, 0.4, f'{val}', 16, color=INK, bold=True,
                   align=PP_ALIGN.CENTER)
        txbox(s, cx - 0.35, base_y + 0.15, bw + 0.9, 0.7, label, 13, color=GRAY,
               align=PP_ALIGN.CENTER)
    if source:
        source_line(s, source)
    fade(s)
    return s


def slide_panels(prs, title, cols, source=None):
    """So sánh hai ô (panel) có viền, mỗi ô một tiêu đề + gạch đầu dòng kèm chấm màu
    (vòng 1 ngày 4: khắc phục bố cục twocol trước — gạch chữ lơ lửng trong vùng trắng,
    không có ô nền rõ ràng để mắt bám). Mỗi mục trong cols là (header, [pts...])."""
    s = add_slide(prs)
    slide_title(s, title)
    for i, col in enumerate(cols[:2]):
        if isinstance(col, (list, tuple)) and len(col) > 1 and isinstance(col[1], (list, tuple)):
            head, pts = col[0], col[1]
        else:
            head, pts = col[0], list(col[1:])
        left = 0.7 + i * 6.2
        rect(s, left, 1.6, 5.6, 4.4, LIGHT if i == 0 else WHITE, LINEC)
        rect(s, left, 1.6, 5.6, 0.8, NAVY if i == 1 else MUTED)
        txbox(s, left + 0.3, 1.7, 5.0, 0.6, head, 24, color=WHITE, bold=True,
               anchor=MSO_ANCHOR.MIDDLE)
        for j, pt in enumerate(pts[:3]):
            top = 2.7 + j * 1.05
            dot = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(left + 0.35), Inches(top + 0.12),
                                     Inches(0.2), Inches(0.2))
            dot.fill.solid()
            dot.fill.fore_color.rgb = NAVY if i == 1 else MUTED
            dot.line.fill.background()
            txbox(s, left + 0.72, top, 4.55, 0.95, str(pt), 16, color=INK, spacing=1.05)
    if source:
        source_line(s, source)
    fade(s)
    return s


def slide_quote(prs, lead, quote, attribution, bg=NAVY):
    s = add_slide(prs)
    rect(s, 0, 0, 13.333, 7.5, bg)
    txbox(s, 1.2, 2.2, 11.0, 1.0, lead, 26, color=LNAVY1)
    txbox(s, 1.2, 3.1, 11.0, 2.2, quote, 44, color=WHITE, bold=True)
    txbox(s, 1.2, 5.6, 11.0, 0.6, attribution, 18, color=LNAVY2)
    fade(s)
    return s


def slide_closing(prs, message, sub):
    s = add_slide(prs)
    txbox(s, 0.7, 1.7, 11.9, 2.2, message, 34, color=INK, bold=True,
           align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    txbox(s, 0.7, 4.6, 11.9, 0.8, sub, 20, color=GRAY, align=PP_ALIGN.CENTER)
    fade(s)
    return s


def slide_diagram(prs, title, top, branches, evidence, source=None, accent=BLUE):
    """Sơ đồ kim tự tháp: một tiêu đề (nền sẫm) ở trên, 2-4 nhánh hỗ trợ ở giữa,
    và 2-4 ô bằng chứng (nền trắng có viền) ở dưới. Mỗi nhánh và mỗi ô bằng chứng
    cùng cột; đường nối mảnh LINEC 1pt."""
    s = add_slide(prs)
    slide_title(s, title)
    top_left, top_w = 3.2, 6.9
    rect(s, top_left, 1.7, top_w, 1.0, NAVY)
    txbox(s, top_left, 1.7, top_w, 1.0, top, 22, color=WHITE, bold=True,
           align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    conn = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,
                                  Inches(top_left + top_w / 2), Inches(2.7),
                                  Inches(top_left + top_w / 2), Inches(3.4))
    conn.line.color.rgb = LINEC
    conn.line.width = Pt(2)
    n = len(branches)
    usable = 11.5
    gap = 0.3
    box_w = (usable - gap * (n - 1)) / n
    col_lefts = [0.9 + i * (box_w + gap) for i in range(n)]
    for left, label in zip(col_lefts, branches):
        rect(s, left, 3.4, box_w, 0.9, LIGHT, LINEC)
        txbox(s, left, 3.4, box_w, 0.9, label, 16, color=INK,
               align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        conn = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,
                                      Inches(left + box_w / 2), Inches(4.3),
                                      Inches(left + box_w / 2), Inches(5.0))
        conn.line.color.rgb = LINEC
        conn.line.width = Pt(1)
    for left, label in zip(col_lefts, evidence):
        rect(s, left, 5.0, box_w, 0.7, WHITE, LINEC)
        txbox(s, left, 5.0, box_w, 0.7, label, 14, color=GRAY,
               align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    if source:
        source_line(s, source)
    fade(s)
    return s


def slide_picture(prs, title, caption, source=None, color=LIGHT):
    """Ảnh lớn chiếm 70-80% slide, caption nhỏ, text tối thiểu (SlideModel: Picture)."""
    s = add_slide(prs)
    slide_title(s, title)
    rect(s, 0.7, 1.7, 8.6, 4.5, color, LINEC)
    txbox(s, 0.7, 1.7, 8.6, 4.5, '(ảnh minh họa)', 24, color=GRAY,
           align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    txbox(s, 9.6, 1.7, 3.0, 4.5, caption, 16, color=INK, spacing=1.2)
    if source:
        source_line(s, source)
    fade(s)
    return s


def slide_table(prs, title, header, rows, source=None):
    """Bảng: mỗi hàng một thông tin, chữ 14-16pt, có dòng nguồn (SKILL.md mục 4)."""
    s = add_slide(prs)
    slide_title(s, title)
    n_cols = len(header)
    left = 0.7
    usable_w = 11.93
    gap = 0.2
    col_w = (usable_w - gap * (n_cols - 1)) / n_cols
    row_h = 0.7
    top_start = 1.7
    for i, h in enumerate(header):
        left_i = left + i * (col_w + gap)
        rect(s, left_i, top_start, col_w, row_h, NAVY)
        txbox(s, left_i, top_start, col_w, row_h, h, 16, color=WHITE, bold=True,
               align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    for j, row in enumerate(rows[:6]):
        top_j = top_start + row_h + 0.15 + j * (row_h + 0.15)
        for i, cell in enumerate(row):
            left_i = left + i * (col_w + gap)
            fill = LIGHT if j % 2 == 0 else WHITE
            rect(s, left_i, top_j, col_w, row_h, fill, LINEC)
            txbox(s, left_i + 0.1, top_j, col_w - 0.2, row_h, str(cell), 14, color=INK,
                   align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    if source:
        source_line(s, source)
    fade(s)
    return s


def slide_bento(prs, title, cells, accent=BLUE):
    """Lưới bento: ô 0 (hero) chiếm 2 cột hàng 1; ô 1-5 là 5 ô thường (1 hàng 2).
    Mỗi ô: số lớn 32pt (màu nhấn), nhãn 15pt, mô tả 12pt (SKILL.md mục 4, nâng cấp ngày 3)."""
    s = add_slide(prs)
    slide_title(s, title)
    usable_w = 11.93
    col_gap, row_h = 0.3, 1.4
    cell_w = (usable_w - col_gap * 2) / 3

    def place(idx):
        if idx == 0:
            return 0.7, 1.7, cell_w * 2 + col_gap, row_h
        if idx == 1:
            return 0.7 + 2 * (cell_w + col_gap), 1.7, cell_w, row_h
        j = idx - 2
        col = j % 3
        return 0.7 + col * (cell_w + col_gap), 1.7 + row_h + col_gap, cell_w, row_h

    for idx, (big, label, desc) in enumerate(cells[:6]):
        left, top, w, h = place(idx)
        rect(s, left, top, w, h, LIGHT, LINEC)
        txbox(s, left + 0.15, top + 0.1, w - 0.3, h * 0.4, big, 32, color=accent, bold=True,
               align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        txbox(s, left + 0.15, top + h * 0.45, w - 0.3, h * 0.3, label, 15, color=INK, bold=True,
               align=PP_ALIGN.CENTER)
        txbox(s, left + 0.15, top + h * 0.68, w - 0.3, h * 0.3, desc, 12, color=GRAY,
               align=PP_ALIGN.CENTER)
    fade(s)
    return s


def slide_timeline(prs, title, events, accent=BLUE, source=None):
    """Dải thời gian nằm ngang (layout thứ 14, ngày 5): 3-5 mốc (when, what, desc),
    một trục kẻ ngang, mỗi mốc một chấm tròn màu nhấn. Khác 'process' (bước tuần tự
    có số thứ tự) ở chỗ timeline nhấn vào mốc thời gian/nhóm (mốc sự kiện, quý, phiên
    bản), không phải thứ tự làm việc."""
    s = add_slide(prs)
    slide_title(s, title)
    events = events[:5]
    n = len(events)
    usable_w = 11.5
    slot = usable_w / n
    mid_y = 3.6
    rect(s, 0.9, mid_y, usable_w, 0.06, LINEC)
    for i, (when, what, desc) in enumerate(events):
        cx = 0.9 + i * slot + slot / 2
        dot = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(cx - 0.14), Inches(mid_y - 0.14),
                                 Inches(0.28), Inches(0.28))
        dot.fill.solid()
        dot.fill.fore_color.rgb = accent
        dot.line.fill.background()
        txbox(s, cx - slot / 2 + 0.2, mid_y - 1.1, slot - 0.4, 0.9, when, 20,
               color=INK, bold=True, align=PP_ALIGN.CENTER)
        txbox(s, cx - slot / 2 + 0.2, mid_y + 0.4, slot - 0.4, 0.6, what, 17,
               color=accent, bold=True, align=PP_ALIGN.CENTER)
    if source:
        source_line(s, source)
    fade(s)
    return s


def slide_kpi(prs, title, kpis, accent=BLUE, source=None):
    """Ô KPI (layout thứ 15, ngày 6): 1-4 chỉ số lớn kèm mũi tên xu hướng tăng (▲) hoặc
    giảm (▼) và phần trăm thay đổi. `kpis` là list [value, label, delta].
    delta nhận dạng tự động hướng: (a) nếu delta có ký tự ▲/▼/↑/↓ (do người dùng tự ghi
    sẵn), dùng nguyên văn, không thêm mũi tên kép; (b) nếu delta bắt đầu bằng "-" (âm),
    coi là giảm, thêm mũi tên ▼ màu đỏ đất; (c) nếu bắt đầu bằng "+" hoặc không có dấu
    (đơn thuần con số), coi là tăng, thêm mũi tên ▲ màu teal. Khác slide_cards/bento:
    KPI nói rõ hướng biến động, không chỉ hiện số đứng yên."""
    s = add_slide(prs)
    slide_title(s, title)
    kpis = kpis[:4]
    n = len(kpis)
    usable_w = 11.93
    gap = 0.3
    card_w = (usable_w - gap * (n - 1)) / n
    up = RGBColor(0x0F, 0x7D, 0x70)   # teal, tăng
    down = RGBColor(0xB4, 0x3B, 0x3B) # đỏ đất, giảm
    for i, (value, label, delta) in enumerate(kpis):
        left = 0.7 + i * (card_w + gap)
        rect(s, left, 1.9, card_w, 3.4, LIGHT, LINEC)
        txbox(s, left + 0.2, 2.1, card_w - 0.4, 1.2, str(value), 56, color=INK, bold=True,
               align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        d = str(delta).strip()
        if any(ch in d for ch in '▲▼↑↓'):
            arrow, dcolor, text = '', (up if '▲' in d or '↑' in d else down), d
        elif d.startswith('-'):
            arrow, dcolor, text = '▼ ', down, d[1:]
        elif d.startswith('+'):
            arrow, dcolor, text = '▲ ', up, d[1:]
        else:
            # con số không có dấu: coi là giá trị tuyệt đối, dùng mũi tên tăng (trung tính
            # nhất, không giả định hướng sai nếu người dùng bỏ sót dấu)
            arrow, dcolor, text = '▲ ', up, d
        txbox(s, left + 0.2, 3.4, card_w - 0.4, 0.6, arrow + text, 20, color=dcolor,
               bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        txbox(s, left + 0.2, 4.1, card_w - 0.4, 1.0, str(label), 15, color=GRAY,
               align=PP_ALIGN.CENTER, spacing=1.1)
    if source:
        source_line(s, source)
    fade(s)
    return s
