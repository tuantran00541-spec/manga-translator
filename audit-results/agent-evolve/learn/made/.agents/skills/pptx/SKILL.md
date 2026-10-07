---
name: pptx
description: Xây bộ slide PowerPoint chuẩn chuyên gia bằng python-pptx (và tool mcp__deck__*) — lưới, chữ, màu, biểu đồ, kể chuyện; dùng khi làm slide mới hoặc tự kiểm slide đã làm.
---

# SKILL: làm slide chuẩn chuyên gia (v1)

Skill này gói lại các con số đo được trong `deck-lab/RESEARCH.md` thành quy trình thực thi được
với python-pptx 1.0.2 (16:9, slide 13.333 × 7.5 inch). Mọi mẫu dưới đây có thể sao chép thành
function; dùng `mcp__deck__make_deck` khi có (criterion C) hoặc viết code python-pptx trực tiếp.

Nguồn đã đọc (mcp__tinyfish__fetch, trích toàn bộ URL):
1. https://guykawasaki.com/the_102030_rule/
2. https://idratherbewriting.com/2018/10/15/ideal-number-of-slides-for-an-hour-long-presentations/
3. https://deckary.com/blog/consulting-slide-standards
4. https://deckary.com/blog/powerpoint-layout-ideas
5. https://slidemodel.com/types-of-slides/
6. https://www.storytellingwithdata.com/blog/2017/8/9/my-guiding-principles
7. https://placeplate.com/blog/slide-design-tips-for-presentations/
8. https://loadfix.github.io/ooxml-reference-corpus/family/pptx__slide-transition.html

## 1. Bố cục và lưới

- Khung slide: 13.333 × 7.5 inch (12192000 × 6858000 EMU), tỉ lệ 16:9.
- Lề an toàn: 0.7 inch bốn cạnh → vùng nội dung 11.93 × 6.1 inch. Không đặt gì quan trọng ngoài
  vùng này; đặc biệt 10% chiều cao cuối (vùng 6.75–7.5 inch) để nguồn/chú thích nhỏ.
- Tiêu đề slide: chạy ngang, top 0.5 inch, cao 0.8 inch, căn trái.
- Lưới cột: 1, 2 (chia đôi, khe 0.5 inch), 3 (bento, khe 0.3 inch), 4 (quy trình ngang).
- Mô hình đọc Z: tiêu đề trên-trái, thông điệp chính/dữ liệu ở giữa hoặc dưới-phải.
- Mỗi slide đúng một thông điệp; nếu tiêu đề chứa chữ "và" → tách hai slide (Deckary).

## 2. Thang cỡ chữ (pt)

Vai trò | pt | chú thích
- Action title (tiêu đề kết luận): 28–32pt, bold, tối đa 2 dòng, ≤15 từ
- Tiêu đề section / bìa chính: 40–44pt
- Số lớn hero stat: 48–80pt, kèm nhãn 14–18pt ngay dưới
- Body / gạch đầu dòng: 18–20pt
- Ghi chú, nguồn, phụ đề: 12–14pt (không bao giờ dưới 12pt)
- Sàn tuyệt đối: 12pt cho mọi chữ trên slide (check_deck sẽ bắt lỗi).

Không đổi cỡ chữ của cùng một vai trò giữa các slide — mọi action title cùng 30pt, mọi body cùng
18pt xuyên suốt (Deckary).

## 3. Màu theo vai trò (mã hex, đã kiểm tương phản)

Bảng màu dùng trong các bộ slide của khóa học (đã tính lại bằng code WCAG):

| Vai trò | Hex | Tương phản trên nền tương ứng |
|---|---|---|
| Nền sáng | #FFFFFF | — |
| Nền sẫm (bìa/trích dẫn) | #1F2A44 (navy) | chữ trắng trên nó = 14.3:1 |
| Chữ chính trên nền sáng | #1A1A1A | trên trắng = 17.4:1 |
| Chữ phụ, nguồn | #595959 | trên trắng = 7.0:1 |
| Màu nhấn (số liệu, CTA, nút) | #0B5ED7 | trên trắng = 5.8:1 |
| Nhấn thứ hai (đội xanh lục, đa dạng hóa) | #0F7D70 (teal) | trên trắng = 5.1:1 |
| Nền card nhạt | #F2F5FA (xám xanh rất nhạt) | chữ #1A1A1A trên nó ≈16:1 |
| Cột "sai/cũ" (so sánh) | #5A6B85 (xám sẫm) | chữ trắng trên nó = 5.4:1 — TỰA CHUYỆN: trước dùng #8A97B0 chỉ 2.9:1, đã sửa ngày 1 |

Quy tắc: tối đa 3 màu (nền, chữ, nhấn). Không dùng màu sắc duy nhất để truyền tải dữ liệu
(mù màu); luôn kèm nhãn. Màu nhấn chỉ dùng cho số liệu quan trọng nhất trong một biểu đồ, phần
còn lại dùng xám (#8A97B0 cho nhãn, không phải chữ).

## 4. Ít nhất 8 kiểu slide mẫu (công thức dựng từng loại)

Mỗi mẫu dưới đây có bố cục, số lượng phần tử tối đa và khung chữ cụ thể; sao chép thành
`add_*_slide(prs, ...)` trong `make_deck`.

1. **Bìa (title)**: nền sẫm toàn slide; tiêu đề 40pt + phụ đề 22pt + dòng tác giả/ngày 14pt;
   không quá 3 dòng chữ.
2. **Mục lục (agenda)**: tiêu đề 30pt + 4–6 dòng, mỗi dòng "số 24pt màu nhấn + chữ 20pt",
   mỗi dòng cao ≤0.9 inch, không quá 6 dòng.
3. **Thẻ / bento (cards)**: tiêu đề + tối đa 3 thẻ ngang; mỗi thẻ: nền nhạt, số lớn 48pt,
   nhãn 18pt, mô tả 14pt ≤3 dòng; thẻ cao ~4.4 inch, rộng ~3.8 inch, khe 0.3 inch.
4. **So sánh hai cột (two-column)**: tiêu đề hành động + 2 khung ngang; tiêu đề khung 22pt nền
   sẫm (navy cho "đúng", xám #5A6B85 cho "cũ"); mỗi khung 3 gạch · 18pt, không quá 3 gạch.
5. **Quy trình (process)**: tiêu đề + 3–5 vòng tròn số (1.2 inch, nền màu nhấn, số 30pt trắng)
   nối bằng đường kẻ; nhãn mỗi bước 16pt ≤2 dòng, chính giữa dưới vòng.
6. **Biểu đồ số liệu (chart)**: tiêu đề hành động + vùng biểu đồ chiếm 70–80% diện tích;
   trục, nhãn 12–14pt, chỉ highlight đúng 1 thanh/số quan trọng nhất bằng màu nhấn, còn lại
   xám; có dòng nguồn 12pt dưới cùng. Với python-pptx: dựng bằng shape (rectangle cho cột)
   vì API chart thật khó kiểm soát tương phản/label.
7. **Trích dẫn (quote)**: nền sẫm toàn slide; câu 36–44pt trắng + dòng nguồn 18pt màu xám
   nhạt; không thêm gì khác.
8. **Kết luận / CTA (closing)**: nền sáng; một câu 36–40pt chính giữa + 1 dòng phụ đề 20pt;
   không liệt kê lại nhiều ý.

Ngoài 8 mẫu trên, có thể dùng: **sơ đồ (diagram)** — kim tự tháp hoặc hai nhánh gặp nhau, mỗi
ô 14–22pt, dùng đường nối mảnh (#D0D9E8, 1–2pt).

## 5. Biểu đồ (chọn loại + quy tắc dựng)

- Cột (bar): so sánh rời rạc, trục ngang là nhóm → dùng `MSO_SHAPE.RECTANGLE`, cùng chiều cao
  khung, bar cao = giá trị/max × cao vùng biểu đồ; nhãn giá trị 16–18pt ngay trên bar.
- Đường (line): xu hướng thời gian → đường nối các điểm, một trục thời gian.
- Mảnh (pie): chỉ khi nói "tỷ lệ trong một tổng", tối đa 3–4 mảnh.
- Nguyên tắc Knaflic (storytellingwithdata): (1) chọn đúng loại theo trục; (2) dọn mọi thứ thừa
  (lưới, viền, 3D); (3) khoanh vùng nên nhìn — highlight đúng một số; (4) mỗi biểu đồ có tiêu
  đề + nguồn; (5) không ghép quá 2 trục vào 1 biểu đồ.
- Dựng bằng shape trong python-pptx để kiểm soát màu/label theo mục 3; đặt trục baseline cố
  định (ví dụ y = 5.4 inch trên slide 7.5 inch), vùng biểu đồ cao 3.2 inch.

## 6. Chuyển trang và animation

- python-pptx không có API transition (đã kiểm `hasattr(slide,'transition') == False`).
- Cách làm: chèn `<p:transition><p:fade spd="med"/></p:transition>` vào XML slide, **đúng thứ
  tự: trước phần tử `p:clrMapOvr`** (ECMA-376), dùng lxml, không dùng SubElement (làm sai
  thứ tự). Mọi slide phải có, check_deck sẽ bắt slide thiếu chuyển trang.
- Fade là lựa chọn an toàn; tránh fly-in, spin, bounce (PlacePlate). Không dùng animation
  xuất hiện cho nội dung quan trọng — chỉ fade nhẹ nếu cần.
- Hàm dùng chung trong các script của khóa học:

```python
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
```

## 7. Tiếng Việt có dấu

- Font: luôn đặt `run.font.name = 'DejaVu Sans'` cho mọi run (có đủ glyph tiếng Việt).
- Không bỏ dấu để né lỗi render; kiểm tra bằng render (`soffice` → PDF → `pdftoppm` → PNG) rồi
  xem ảnh bằng view_image.
- Nội dung tiếng Việt viết theo văn phong tự nhiên (điểm số, % , dấu phẩy trong câu), không
  cần gượng ép số; chỉ cần dấu đủ và không bị lỗi font.
- Kiểm tra: sau khi render, nhìn qua từng trang PNG; chữ như "ệ", "ữ", "ầ" phải hiển thị đúng,
  không ô vuông/tofu.

## 8. Quy trình tự kiểm (check_deck, 8 điểm)

Chạy sau mỗi bộ slide, trước khi báo "xong":

1. **Thiếu chuyển trang**: mọi slide phải có `<p:transition>`.
2. **Shape ra ngoài slide**: kiểm tra `left`, `top`, `left+width`, `top+height` trong [0,
   12192000] × [0, 6858000] EMU (dung sai 1000 EMU).
3. **Chữ dưới 12pt**: mọi `font.size.pt < 12` là lỗi.
4. **Khung chữ trống**: textbox không có solidFill mà text rỗng → lỗi. (Khung hình trang trí có
   fill không tính.)
5. **Slide quá 70 chữ (tính theo từ, `text.split()`)**: báo lỗi, cần cắt hoặc tách slide.
6. **Tương phản dưới 4.5:1**: so màu chữ (mọi run) với nền thực (fill của shape chứa chữ, hoặc
   nền slide) bằng công thức WCAG; xem `deck-lab/RESEARCH.md` mục 5 và bảng màu mục 3.
7. **Chữ tràn khung**: ước lượng số dòng wrap theo chiều rộng khung và cỡ chữ; nếu số dòng ×
   leading vượt chiều cao khung >40% là cảnh báo (không sai cứng, cần render xác nhận).
8. **Khung chữ chồng nhau**: hai textbox có vùng chứa giao nhau >30% diện tích khung nhỏ hơn →
   cảnh báo (nhiều trường hợp hợp lệ là chữ đặt lên nền card, cần render xác nhận).

Hai điểm 7–8 là "cảnh báo" (báo nhưng không loại bộ), điểm 1–6 là "lỗi cứng" (phải sửa trước
khi báo xong). Script thực thi: xem `.agents/skills/pptx/check_deck.py` (sẽ viết tại phiên
này).

## 9. Quy trình làm việc (theo skill này)

1. Chọn 8 mẫu (mục 4) sẽ dùng, sắp thành mạch kể chuyện: bìa → mục lục → 4–6 mẫu nội dung
   (thẻ/so sánh/quy trình/biểu đồ/sơ đồ/trích dẫn) → kết luận.
2. Dựng từng slide bằng hàm dùng chung (txbox, rect, fade, mục 6).
3. Chạy `check_deck` (mục 8); sửa hết lỗi cứng, ghi chú cảnh báo còn lại.
4. Render + view_image từng trang; kiểm tra glyph tiếng Việt (mục 7), đặc biệt các slide có
   chữ có dấu nhiều.
5. Sau mỗi bộ: ghi bài học mới vào SKILL.md (v2, v3...) — ví dụ ngày 1 học được: màu #8A97B0
   không đạt, phải dùng #5A6B85; heuristic overflow/false-positive cần render xác nhận.

## 10. Các bẫy đã gặp (từ lỗi ngày 1, ghi thật)

- python-pptx 1.0.2 không có API transition/animation — phải chèn XML tay, đúng thứ tự
  (trước `clrMapOvr`).
- `etree.SubElement` thêm transition vào cuối `p:sld` gây lỗi schema — phải dùng
  `el.insert(list(el).index(anchor), t)` với anchor là `p:clrMapOvr`.
- Khung chữ đặt đè lên hình nền (card) bị heuristic "chồng khung" hiểu nhầm là lỗi — phải
  chỉ báo cảnh báo, không tính lỗi cứng.
- Màu #8A97B0 (xám-xanh nhạt) với chữ trắng chỉ đạt 2.94:1 — dưới sàn 4.5:1; sửa bằng #5A6B85
  (5.42:1).
- Chữ tiếng Việt gõ nhầm nghĩa (không phải lỗi font) — phải đọc lại và sửa bằng render +
  view_image, không chỉ chạy script.

## 11. Nâng cấp sau bộ a7 (SKILL.md v2)

Bài học từ `deck-lab/a7_review.slide.pptx` (xây bằng `helpers.py` thuần, không sửa code khi dựng):

1. **Typo tự gõ trong nội dung tiếng Việt vẫn là lỗi lớn nhất** — a7 bị "Rô lưới" (sai ý, đọc
   như "rô" thay vì "lưới lề") và dấu nháy thừa trong bullet so sánh hai cột (chữ `"` trong
   content Python va chạm, render ra `
` trong string). Mọi lần: đọc lại content bằng
   `repr()` trước khi chạy, và render + view_image bắt hết.
2. **`helpers.py` (mục 4) đủ dùng, không cần sửa logic** — cả 8 mẫu dựng được đúng bố cục
   từ SKILL.md v1; điều sửa được chỉ là *nội dung truyền vào*, không phải *hàm dựng*.
   Điều này xác nhận đúng thiết kế skill: thay đổi giữa các bộ nên nằm ở data (list, string),
   không nằm ở code.
3. **Biểu đồ: highlight thanh quan trọng nhất bằng màu nhấn, còn lại xám `MUTED`** — a7 slide 6
   (thanh teal cuối = 35, cao nhất) và a8 slide 5 (thanh xanh cuối = 63) đều đúng nguyên tắc
   Knaflic mục 5; không cần thêm gì khác vào `slide_barchart`.
4. **Cảnh báo "chồng khung >30%" trong `check_deck.py` (mục 8) xác nhận là dương tính giả
   có chủ đích trong 2 layout** (nền card dưới chữ, vòng tròn quy trình dưới nhãn) — giữ là
   cảnh báo, không nâng thành lỗi cứng; đã ghi rõ trong mục 10.
5. **Bộ a7 (tiếng Việt có dấu) và a8 (màu theo vai trò) đều đạt "lỗi cứng: OK" trước render** —
   chứng minh check_deck chạy được như bước chặn cuối cùng của SKILL.md mục 9; bước render +
   view_image vẫn là bắt buộc, không thể bỏ qua.

## 12. Nâng cấp sau bộ a8 (SKILL.md v2.1)

Bài học từ `deck-lab/a8_maulung.cusos.pptx`:

1. **Palette 3 màu (trắng/navy + nhấn + xám) áp cho bộ có số liệu cho kết quả sạch hơn** — a8
   dùng `MUTED` (#5A6B85) thay vì `BLUE` cho 4 thanh nền, chỉ `accent` (xanh teal hoặc xanh
   dương, mỗi bộ một màu nhấn duy nhất) cho thanh quan trọng nhất; đúng quy tắc "màu nhấn
   chỉ dùng cho số quan trọng nhất" (mục 5, Knaflic), kiểm được bằng mắt trên render.
2. **Không nhất thiết phải có slide trích dẫn (quote) ở mọi bộ** — a8 bỏ slide quote, thay bằng
   `slide_closing` trực tiếp (6 slide thay vì 8); bộ nhỏ hơn vẫn đủ mạch chuyện nếu tiêu đề
   hành động mạnh. Cập nhật: 8 mẫu (mục 4) là *bộ công cụ*, không phải *checklist bắt buộc*
   mỗi bộ phải dùng đủ 8.
3. **`slide_cards` với "1" làm số lớn (hero stat) hoạt động tốt** — 3 thẻ "1 nền / 1 nhấn / 1
   xám" (a8 slide 3) là cách nén quy tắc màu thành 1 slide, thay vì liệt kê dài; nguyên tắc:
   thẻ dùng cho quy tắc, không dùng cho số liệu thật.
4. **Cảnh báo tràn khung của check_deck vẫn dương tính giả với `slide_cards` (ô mô tả 14pt)** —
   giữ nguyên mức "cảnh báo, cần render" (mục 8, mục 10); không cần sửa code helpers.
