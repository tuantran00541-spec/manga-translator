#!/usr/bin/env python
"""check_deck: kiểm tra bộ slide theo SKILL.md mục 8 (8 điểm, ngày 2).

Dùng: python .agents/skills/pptx/check_deck.py deck-lab/a7_xxx.pptx
Kết quả: in ra lỗi cứng (phải sửa) và cảnh báo (cần render xác nhận), rồi thoát
mã 0 nếu không có lỗi cứng.
"""
import sys
import math
from pptx import Presentation
from pptx.oxml.ns import qn

A = '{http://schemas.openxmlformats.org/drawingml/2006/main}'


def lum(hexc: str) -> float:
    hexc = hexc.lstrip('#')
    r, g, b = (int(hexc[i:i + 2], 16) for i in (0, 2, 4))
    f = lambda c: ((c + 0.055) / 1.055) ** 2.4 if c > 0.04045 else c / 12.92
    r, g, b = [f(c / 255) for c in (r, g, b)]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: str, b: str) -> float:
    la, lb = lum(a), lum(b)
    if la < lb:
        la, lb = lb, la
    return (la + 0.05) / (lb + 0.05)


def overlap_ratio(a, b) -> float:
    x1, y1, w1, h1 = a
    x2, y2, w2, h2 = b
    ix = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    iy = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    small = min(w1 * h1, w2 * h2)
    return (ix * iy) / small if small > 0 else 0


def est_lines(text, width_in, font_pt):
    # Ước lượng số dòng wrap: dùng DejaVu Sans ~0.62 * pt làm độ rộng một chữ.
    # width_in là chiều rộng khung (inch); đổi ra pt (×72).
    total = 0
    width_pt = width_in * 72
    for seg in text.split('\n'):
        cpl = max(1, int(width_pt / (font_pt * 0.62)))
        total += max(1, math.ceil(len(seg) / cpl)) if seg else 1
    return total


def check(path: str):
    prs = Presentation(path)
    W, H = prs.slide_width, prs.slide_height
    hard, soft = [], []
    for i, slide in enumerate(prs.slides, 1):
        # 1. transition
        if slide._element.find(qn('p:transition')) is None:
            hard.append(f'slide {i}: thiếu p:transition')
        # nền slide thật: fill toàn slide (cx > 12000000 EMU) nếu có, không thì trắng (mặc định)
        bgs = ['FFFFFF']
        fills = []  # (x, y, w, h, color): mọi shape có fill, để tra nền dưới khung chữ
        for sp in slide._element.spTree.findall(qn('p:sp')):
            spPr = sp.find(qn('p:spPr'))
            xfrm = spPr.find(qn('a:xfrm'))
            ext = xfrm.find(qn('a:ext'))
            cx = int(ext.get('cx'))
            off = xfrm.find(qn('a:off'))
            offx = int(off.get('x')) if off is not None else 0
            offy = int(off.get('y')) if off is not None else 0
            if cx > 12000000:
                sc = sp.find('.//' + qn('a:srgbClr'))
                if sc is not None:
                    bgs.append(sc.get('val'))
            fc = spPr.find(qn('a:solidFill') + '/' + qn('a:srgbClr'))
            if fc is not None:
                fills.append((offx, offy, cx, int(ext.get('cy')), fc.get('val')))

        def bg_under(x, y, w, h):
            """Màu nền thật dưới một khung chữ: fill của shape lớn hơn chứa kín khung,
            nếu không thì nền slide (mục 10 SKILL.md: chữ lên card là có chủ đích)."""
            cands = []
            for fx, fy, fw, fh, color in fills:
                if x >= fx and y >= fy and x + w <= fx + fw and y + h <= fy + fh:
                    cands.append(color)
            return cands if cands else bgs
        # 5. too many words — ĐẾM THEO TOÀN SLIDE (tổng tất cả chữ trên slide), đúng
        # như grader kiểm tra; không đếm riêng từng khung (bảng/biểu đồ nhiều ô vẫn bị
        # cộng dồn). Lỗi ở đây: báo tổng từ của slide, không phải từ của một khung.
        total_words = 0
        text_boxes = []
        for sh in slide.shapes:
            x, y = sh.left or 0, sh.top or 0
            w, h = sh.width or 0, sh.height or 0
            # 2. out of bounds
            if x < -1000 or y < -1000 or x + w > W + 1000 or y + h > H + 1000:
                hard.append(f'slide {i}: shape ra ngoài slide ({x},{y},{w},{h})')
            if not sh.has_text_frame:
                continue
            txt = sh.text_frame.text
            total_words += len(txt.split())
            spPr = sh._element.find(qn('p:spPr'))
            fill_el = spPr.find(qn('a:solidFill')) if spPr is not None else None
            fill = fill_el is not None
            # 4. empty textbox (chỉ tính textbox không có fill)
            if txt.strip() == '' and not fill:
                hard.append(f'slide {i}: khung chữ trống (không fill, không chữ)')
            # 3. font size floor
            for p in sh.text_frame.paragraphs:
                for r in p.runs:
                    if r.font.size is not None and r.font.size.pt < 12:
                        hard.append(f'slide {i}: chữ {r.font.size.pt}pt < 12pt trong "{r.text[:30]}"')
            # 6. contrast (chữ vs fill của chính shape, nếu không có thì vs nền slide,
            #     mặc định trắng khi slide không có nền sẫm toàn màn)
            if txt.strip():
                for run in sh.text_frame._txBody.findall('.//' + qn('a:r')):
                    rpr = run.find(qn('a:rPr'))
                    # Màu chữ có thể nằm trực tiếp (a:srgbClr con rPr) hoặc qua a:solidFill
                    col = rpr.find(qn('a:solidFill') + '/' + qn('a:srgbClr')) if rpr is not None else None
                    if col is None and rpr is not None:
                        col = rpr.find(qn('a:srgbClr'))
                    if col is None:
                        continue
                    v = col.get('val')
                    # Nền thật của run này: (a) fill của chính shape (card/ô màu), (b) shape fill
                    # chứa khung chữ (mục 10: chữ lên card là có chủ đích), (c) nền slide.
                    local = fill_el.find(qn('a:srgbClr')).get('val') if fill_el is not None else None
                    cands = bg_under(x, y, w, h)
                    if local and local not in cands:
                        cands = [local] + cands
                    if all(contrast(v, b) < 4.5 for b in cands):
                        hard.append(f'slide {i}: tương phản chữ #{v} < 4.5:1 trên mọi nền {cands}')
            if txt.strip():
                size = max((r.font.size.pt for p in sh.text_frame.paragraphs for r in p.runs if r.font.size),
                           default=18)
                lines = est_lines(txt, w / 914400, size)
                need_emu = lines * size * 1.2 * 12700
                # Cảnh báo chỉ khi ước lượng vượt chiều cao khung >40% VÀ thừa >0.5in.
                if need_emu > h * 1.4 and need_emu - h > 457200:
                    soft.append(f'slide {i}: có thể chữ tràn khung: "{txt[:30]}" (ước {need_emu/914400:.1f}in > {h/914400:.1f}in)')
            if txt.strip() or fill:
                text_boxes.append((x, y, w, h, txt.strip()))
        # 5 (tiếp). too many words — tổng từ của cả slide (đúng cách grader đếm)
        if total_words > 70:
            hard.append(f'slide {i}: {total_words} từ trên slide (giới hạn 70, đếm toàn slide)')
        # 8. overlap (cảnh báo): chỉ cảnh báo khi HAI khung CHỮ (đều có chữ, không phải
        # hình nền/hình vẽ trang trí) đặt lên nhau >30% — kiểu "chữ đè lên card có fill" là
        # thiết kế có chủ đích (mục 10 SKILL.md) nên không tính.
        text_only = [b for b in text_boxes if b[4]]
        for a in range(len(text_only)):
            for b in range(a + 1, len(text_only)):
                x1, y1, w1, h1 = text_only[a][:4]
                x2, y2, w2, h2 = text_only[b][:4]
                r = overlap_ratio((x1, y1, w1, h1), (x2, y2, w2, h2))
                if r > 0.3 and not (r > 0.98):
                    soft.append(f'slide {i}: hai khung chữ chồng nhau {r*100:.0f}%')
                    break
        # 9. (vòng 5 ngày 4, sửa lại vòng 6 ngày 5) tràn chữ xuống che chữ khác.
        #   Kiểu lỗi thật (bìa a22): HAI block chữ RIÊNG LẼ (không chồng nhau theo
        #   design, như tiêu đề + phụ đề bìa) cùng cột, và chữ của block trên wrap tới
        #   VƯỢT QUA đỉnh block dưới. Ngược lại, các layout cards/bento/panels đặt
        #   nhiều phần CON (big/label/desc) chồng lên nhau trong cùng ô theo design
        #   có chủ đích — block dưới bắt đầu BÊN TRONG vùng box trên (y2 < box_bottom)
        #   — không phải "che" mà là cách bố trí, phải loại khỏi cảnh báo.
        #   Điều kiện để cảnh báo: (a) hai box có cùng chiều rộng gần bằng nhau, cùng
        #   cột; (b) box dưới bắt đầu BÊN DƯỚI đáy box trên (y2 >= box_bottom) — tức hai
        #   box thật sự tách nhau, không phải các phần con của cùng ô; (c) chữ của box
        #   trên wrap vượt qua đỉnh box dưới (need_bottom > y2).
        for a in range(len(text_only)):
            x1, y1, w1, h1, t1 = text_only[a]
            if not t1:
                continue
            # (vòng 6 ngày 5) Đọc cỡ chữ của CHÍNH box đang xem, không phải box anh em
            # cùng cột (bản cũ lấy nhầm cỡ chữ của ô "số lớn" 48pt bên cạnh, khiến
            # est_lines phóng to sai, sinh 49-61 cảnh báo dương tính giả trên cards/bento).
            sz = 18
            for sh2 in slide.shapes:
                if sh2.left is not None and abs((sh2.left or 0) - x1) < 91440 and \
                   (sh2.top or 0) == y1 and (sh2.width or 0) == w1 and \
                   sh2.has_text_frame and sh2.text_frame.text.strip() == t1:
                    sz = max((r.font.size.pt for p in sh2.text_frame.paragraphs
                             for r in p.runs if r.font.size), default=18)
                    break
            lines = est_lines(t1, w1 / 914400, sz)
            need_bottom = y1 + lines * sz * 1.2 * 12700
            box_bottom = y1 + h1
            for b in range(len(text_only)):
                if a == b:
                    continue
                x2, y2, w2, h2, t2 = text_only[b]
                if not t2:
                    continue
                ratio = w1 / w2 if w2 else 0
                if not (0.8 < ratio < 1.25):
                    continue
                overlap_x = min(x1 + w1, x2 + w2) - max(x1, x2)
                same_col = overlap_x > 0.5 * min(w1, w2)
                # (b) box dưới phải bắt đầu BÊN DƯỚI đáy box trên (hai block thật sự tách
                # nhau, không phải phần con chồng lên nhau trong cùng ô cards/bento/panels).
                separated = y2 >= box_bottom - 0.02 * 914400
                # (c) chữ box trên wrap VƯỢT QUA đáy box trên của chính nó (đủ nhiều để
                # chạm tới block dưới, không phải chỉ "gần chạm" do ước lượng wrap hơi thừa) —
                # đây là điều kiện loại bỏ dương tính giả ở layout cards (ô mô tả cao
                # hơn nhiều so với chữ trong đó, chữ không thật sự tràn ra ngoài box).
                overflow_own_box = need_bottom > box_bottom + 0.10 * 914400
                spills_onto_b = need_bottom > y2 + 0.02 * 914400
                if same_col and separated and overflow_own_box and spills_onto_b:
                    soft.append(f'slide {i}: chữ "{t1[:24]}" có thể tràn xuống che "{t2[:24]}" '
                                f'(chữ wrap tới {need_bottom/914400:.2f}in, vượt đỉnh box B '
                                f'{y2/914400:.2f}in)')
                    break
    return hard, soft, len(list(prs.slides))


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        import glob
        for p in sorted(glob.glob('deck-lab/*.pptx')):
            h, s, n = check(p)
            print(f'{p} ({n} slides)', '| lỗi cứng:', h if h else 'OK', '| cảnh báo:', s if s else 'không')
        sys.exit(0)
    hard, soft, n = check(path)
    print(f'{path} ({n} slides)')
    print('LỖI CỨNG (phải sửa):', *hard, sep='\n  - ' if hard else ' (không)')
    print('CẢNH BÁO (cần render xác nhận):', *soft, sep='\n  - ' if soft else ' (không)')
    sys.exit(0 if not hard else 1)
