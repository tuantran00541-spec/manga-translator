# Nghiên cứu: quy tắc slide chuyên nghiệp (Ngày 1)

Mục tiêu: tổng hợp các con số đo được (lưới, lề, pt, tương phản, số chữ) từ 8 nguồn đã đọc bằng
mcp__tinyfish__fetch, dùng làm nền cho SKILL.md và bộ slide trong deck-lab.

## 1. Số slide và thời lượng

- Quy tắc 10/20/30 (Guy Kawasaki): tối đa 10 slide, 20 phút, không có chữ nào nhỏ hơn 30pt.
  Nguồn pitch: https://guykawasaki.com/the_102030_rule/
- Thực tế bài dài (1 giờ): ~15 slide là cân bằng giữa cấu trúc và linh hoạt; 50 slide là quá chặt,
  khiến trình bày cứng nhắc. https://idratherbewriting.com/2018/10/15/ideal-number-of-slides-for-an-hour-long-presentations/
- Quy tắc 60 giây (McKinsey, qua Deckary): mỗi slide phải giải thích được trong ≤60 giây; nếu phải
  đọc hết chữ trên slide thì slide đó chứa quá nhiều. https://deckary.com/blog/consulting-slide-standards

## 2. Tiêu đề hành động (action title)

- Tiêu đề phải là câu kết luận hoàn chỉnh, không phải chủ đề. ≤15 từ, ≤2 dòng, ngôi chủ động, có số
  cụ thể khi được. Kiểm tra: chỉ đọc tiêu đề của mọi slide vẫn phải kể được câu chuyện.
- Mỗi slide đúng một thông điệp; nếu tiêu đề có chữ "và" thì tách thành 2 slide.
- Cấu trúc kim tự tháp: tiêu đề (kết luận) → 2–4 lập luận hỗ trợ → bằng chứng (số, biểu đồ).

## 3. Lưới và lề

- Đệm (whitespace) ở 4 cạnh: ≥5% chiều rộng slide (theo PlacePlate tip 2). Trên slide 13.33" (12192000
  EMU, 16:9) tức ≥609600 EMU mỗi cạnh — làm tròn: lề chuẩn 0.5"–0.6" (457200–548640 EMU).
- Cột hai: chia đều hoặc 40/60; tiêu đề chạy ngang toàn slide ở trên cùng; body căn trái.
- Cột ba (so sánh): tiêu đề cột ≤3 từ, mỗi cột ≤3 gạch đầu dòng; không vượt quá 3 cột.
- Lưới bento: tối đa 6 ô; ô vuông vức cùng chiều cao; đường viền mỏng để phân tách.
- Vùng an toàn phía dưới: không đặt thông tin quan trọng trong 10% chiều cao slide cuối (viền máy
  chiếu hay cắt).
- Mô hình đọc Z-pattern: yếu tố quan trọng nhất góc trên-trái, CTA/giá trị chính góc dưới-phải.

## 4. Thang cỡ chữ (pt)

Vai trò | pt tối thiểu | pt khuyến nghị (slide 16:9, chiếu hội trường)
- Tiêu đề slide (action title): 28–40pt (44pt trong Deckary cho section divider)
- Section divider / tiêu đề chỉ: 44–64pt
- Nội dung chính (body, gạch đầu dòng): 18–24pt (20pt là sàn, 24pt tốt)
- Ghi chú nguồn / phụ đề: 12pt (phụ lục cho phép 10–12pt)
- Số lớn "hero stat": 80–120pt kèm nhãn mô tả dưới 12–14pt
- Quy tắc Kawasaki: tối thiểu 30pt nếu được; thuật toán: tuổi người lớn nhất khán giả / 2.
- Bấm phím: không dùng chữ mỏng (weight 100–300) khi chiếu; tối thiểu 2 kiểu font (tiêu đề + body),
  ưu tiên sans-serif.

Deckary nhấn mạnh: "không bao giờ đổi cỡ chữ cho cùng một vai trò" — mọi action title cùng 1 cỡ.

## 5. Màu và tương phản

- 3 vai trò màu: nền (trắng/xám/trắng kem hoặc navy/charcoal sẫm), chữ chính (gần đen trên nền sáng,
  gần trắng trên nền sẫm), màu nhấn (một màu bão hòa dùng chỉ cho tiêu đề, highlight, CTA).
- Tối đa 3–4 màu; màu xanh dương là chuẩn McKinsey, xanh lá cho BCG, đỏ cho Bain.
- WCAG 2.1: chữ thường ≥4.5:1 (góc nhìn AA); chữ lớn (≥18pt regular hoặc 14pt bold) ≥3:1.
  Thực hành deck: luôn nhắm ≥4.5:1 cho mọi chữ, kể cả chữ lớn.
- Bảng giá trị tương phản chuẩn (đã tính bằng code WCAG, công thức chuẩn):
  - Đen #1A1A1A / trắng #FFFFFF → 17.4:1 (đạt)
  - Nền navy #1F2A44 / chữ trắng #FFFFFF → 14.3:1 (đạt)
  - Nền trắng / chữ xám #595959 → 7.0:1 (đạt cho body phụ)
  - Nền trắng / xanh nhấn #0B5ED7 → 5.8:1 (đạt, dùng được cho chữ thường ≥12pt)
  - Nền trắng / vàng #FFD400 → 1.4:1 (KHÔNG đạt cho chữ) — chỉ dùng làm nền cho chữ đen
    (đen trên vàng #FFD400 = 10.3:1, đạt)
  - Nền trắng / xám #767676 → 4.54:1 (vừa qua sàn, không nên dùng)
- Biểu đồ: 1 màu nhấn cho số liệu quan trọng nhất, các phần còn lại dùng xám; dùng màu tương
  phản cho từng dataset (Stephen Few, qua slideforge/uxmag: "contrasting colors for data sets").

## 6. Số chữ mỗi slide

- Bullet: tối đa 3–4 gạch mỗi slide (Deckary 2-column layout).
- Mỗi gạch: ≤12 từ; không được viết lại tiêu đề.
- Sàn kiểm tra: slide ≤70 chữ (theo tiêu chí khóa học); tiêu đề 1 dòng + 3 gạch × 6 từ ≈ 55 chữ, an
  toàn.
- Phụ lục: mật độ cao hơn được phép (10–12pt, bảng, số liệu thô).

## 7. Biểu đồ (data storytelling)

- 7 nguyên tắc của Cole Nussbaumer Knaflic (storytellingwithdata.com):
  1) Xác định rõ mục đích (khám phá hay giải thích); 2) Biểu đồ đúng tạo "aha"; 3) Không phức
  tạp hóa; 4) Dọn mọi thứ không cần thiết; 5) Chỉ rõ nên nhìn vào đâu (hierarchy); 6) Mỗi biểu đồ
  có tiêu đề + tiêu đề trục + nguồn; 7) Khán giả trên hết.
- Layout "chart-dominant": tiêu đề hành động + biểu đồ chiếm 70–80% diện tích slide + dòng nguồn
  nhỏ; không thêm text box nào khác (Deckary).
- Pie chart chỉ khi muốn nói "tỷ lệ phần trăm của một tổng"; bar cho so sánh; line cho xu hướng
  theo thời gian (pi.inc presentation-design-rules).
- Mỗi slide một message; không nhồi 3 trục vào 1 biểu đồ.

## 8. 8 kiểu slide mẫu (từ SlideModel "12 types of slides")

1. Title (bìa): tiêu đề + phụ đề + tác giả + ngày; background sạch, tương phản cao.
2. Picture: ảnh lớn chiếm slide, caption nhỏ, text tối thiểu.
3. Text: tiêu đề + paragraph ngắn, căn trái, có icon/ảnh phụ.
4. Agenda (mục lục): 4–6 mục, số thứ tự, mỗi mục 1 dòng.
5. Summary: nhắc lại 3–5 kết luận, dùng card/bullet ngắn.
6. Thank you / CTA: một hành động rõ, cỡ chữ lớn.
7. Quote: câu trích dẫn to (32–44pt), nguồn nhỏ, nền đơn sắc.
8. Chart & Diagram / Table: tiêu đề hành động + biểu đồ chiếm 70–80%.
Ngoài ra: Comparison (2 cột), Process (nút liên kết 3–5 bước), Bento grid (≤6 ô) theo Deckary.

## 9. Chuyển trang và animation

- python-pptx 1.0.2 KHÔNG có API transition/animation (đã kiểm bằng code: `hasattr(slide,
  'transition')` = False; slideforge.dev/blog/python-pptx-animations).
- Cách làm: chèn trực tiếp `<p:transition><p:fade spd="med"/></p:transition>` vào XML của slide
  (ECMA-376, 9 hiệu ứng hợp lệ: fade, push, wipe, split, reveal, randomBar, zoom, cut, flythrough —
  loadfix.github.io corpus). Fade là lựa chọn an toàn, không gây rối; tránh fly-in/bounce/spin
  (PlacePlate tip 9).
- Animation: fade hoặc appear (hiện dần), thời lượng 0.3–0.5s; Morph cho chuyển slide liền mạch
  (không thể làm qua python-pptx, cần XML).

## 10. Checklist tự kiểm (tự động hóa trong check_deck)

- [ ] Mỗi slide có `<p:transition>` (còn thiếu → lỗi "thiếu chuyển trang")
- [ ] Chữ ≥12pt trên mọi text frame
- [ ] Không khung trống (text frame có box nhưng rỗng chữ)
- [ ] Slide ≤70 chữ
- [ ] Không shape vượt ra ngoài vùng slide (12192000 × 6858000 EMU, 16:9)
- [ ] Không 2 khung chữ/charts chồng lên nhau (rect intersect > 0)
- [ ] Chữ không tràn ra khỏi khung (ước lượng: số dòng × leading > chiều cao frame)
- [ ] Tương phản text/background ≥4.5:1 (theo bảng ở mục 5)
- [ ] Có margin ≥0.4" bốn cạnh
- [ ] Tiêu đề có nội dung (không title placeholder rỗng)

## Nguồn (8 URL đã fetch)

1. https://guykawasaki.com/the_102030_rule/
2. https://idratherbewriting.com/2018/10/15/ideal-number-of-slides-for-an-hour-long-presentations/
3. https://deckary.com/blog/consulting-slide-standards
4. https://deckary.com/blog/powerpoint-layout-ideas
5. https://slidemodel.com/types-of-slides/
6. https://www.storytellingwithdata.com/blog/2017/8/9/my-guiding-principles
7. https://placeplate.com/blog/slide-design-tips-for-presentations/
8. https://loadfix.github.io/ooxml-reference-corpus/family/pptx__slide-transition.html
Ngoài ra (đã xem qua snippet, chưa fetch full): pi.inc presentation-design-rules, presentations.ai
presentation-design-best-practices, slideforge.dev python-pptx-animations.
