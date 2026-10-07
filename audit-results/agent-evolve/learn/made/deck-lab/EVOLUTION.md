# Tiến hóa plugin (check_deck + make_deck) — mỗi vòng một điểm yếu thật

Mỗi vòng: tìm điểm yếu bằng ảnh render + check_deck, viết lại plugin/helper, dựng lại bộ,
đo số trước/sau. Ghi thật, không phải "đã cải thiện".

## Vòng 1 (ngày 4): điểm yếu bố cục "so sánh hai cột" + lỗi render barchart

Điểm yếu (xem từ render a4 slide 4, a8 slide 3 và a20 lần dựng đầu):
- `slide_twocol`: gạch đầu dòng lơ lửng trong vùng trắng, không có ô nền/
viền nào giữ chúng → mắt người đọc không biết bullet thuộc cột nào khi hai cột
ngắn dài khác nhau. Không có chấm màu, không có ô panel.
- `slide_barchart`: nhãn giá trị luôn đặt *trên* bar (`by - 0.45`); khi bar cao
đến gần đáy vùng tiêu đề, chữ số có thể dính vào tiêu đề hoặc che mất cạnh bar
(bắt gặp ở slide barchart bộ a13, a16 — render xem thấy "50" nằm rất sát đầu bar).
- `slide_panels` (mới, thay thế cho twocol khi nội dung nhiều bullet): lần dựng
đầu a20, truyền `columns` dạng phẳng `[[Lỗi, p1, p2, p3], ...]` khiến helper
hiểu nhầm là "một dòng văn bản" (render ra `['...']` nguyên khối, không tách
bullet riêng) — đã sửa cách nhận `cols` (nằm ở helper, không phải lỗi dữ liệu).

Sửa (helpers.py, deck_tools.py, plugin schema):
- Thêm `slide_panels`: hai ô có viền (nền LIGHT/WHITE) + thanh tiêu đề màu
(MUTED/NAVY) + bullet có chấm tròn màu, mỗi bullet một dòng riêng.
- `slide_barchart`: nhãn giá trị đặt *bên trong* thân bar (chữ trắng, bold) khi
bar cao (>0.9in), chỉ đặt trên bar (chữ INK) khi bar ngắn.
- Plugin: thêm kind "panels" (schema) và wire `slide_panels` trong deck_tools.py.

Dựng lại: a20_phan.tich.loi (bộ chuyên về chủ đề này, dùng panels + barchart sửa
mới), 7 slide, check_deck "LỖI CỨNG: không", render slide 3 (barchart) và slide 4
(panels) — thấy bullet tách riêng, chấm màu đúng vị trí, nhãn số nằm gọn trong bar.

Số đo trước/sau (slide panels của a20):
- Trước (v1): 1 textbox chứa cả 3 bullet + ký tự '[' ']' lẫn vào chữ (render thấy
  rõ: "['Tràn khung...', 'Vượt 70 từ...')") — 3 bullets nhồi vào 1 dòng, không
  đếm được "số bullet = 3".
- Sau (v2): 3 textbox bullet riêng, mỗi bullet 1 dòng wrap tối đa 2 dòng, có
  chấm tròn 0.2in — đếm được đúng 3 chấm, đúng 3 dòng.

## Vòng 2 (đang làm): điểm yếu chữ/cỡ — "70 từ trên slide" cần đếm theo slide

Điểm yếu (từ lỗi grader ngày 4): a18 slide 2 (table) 73 từ, a19 slide 3 (bento)
92 từ — vượt ngưỡng 70 **theo toàn slide**, nhưng check_deck trước đây (ngày 2)
chỉ đếm **theo từng khung chữ**, nên không bắt được (slide có 10-15 ô nhỏ, mỗi
ô đều <70 từ, nhưng cộng lại vượt 70).

Sửa (check_deck.py): đổi cách đếm sang **tổng từ của cả slide** (mỗi slide một
số), đúng với cách grader đếm.

Dựng lại: a18 (cắt bảng từ 4 dòng còn 3 dòng + rút ngắn tiêu đề), a19 (bento
4 ô, rút ngắn mô tả 2 ô).

Số đo trước/sau (đã đo bằng script đếm từ theo slide sau khi sửa):
- a18 slide 2 (table): 73 từ → 60 từ (cắt từ 4 dòng bảng còn 3 dòng + rút ngắn
  tiêu đề "Bốn tiêu chí..." còn "Ba tiêu chí...").
- a19 slide 3 (bento): 92 từ → 62 từ (cắt 1 ô bento từ 5 ô còn 4 ô + rút ngắn
  mô tả 2 ô còn 1 dòng ngắn).
- Chạy check_deck.py (v2, đếm theo slide) trên a18, a19 sau khi rebuild: "LỖI
  CỨNG: không, CẢNH BÁO: không" cho cả hai — trùng khớp với kết quả grader
  (không còn báo slide nào vượt 70 từ).

## Tóm tắt 2 vòng (ngày 4)

| Vòng | Điểm yếu | Đã sửa ở | Số đo trước → sau |
|---|---|---|---|
| 1 | bullet "lơ lửng" trong twocol; nhãn barchart che bar; panels nhận sai data | helpers.py (thêm slide_panels, sửa slide_barchart), deck_tools.py, plugin schema | a20 slide 4: 3 bullet nhồi trong 1 dòng có '[' ']' lẫn → 3 bullet riêng, 3 chấm tròn, 0 ký tự '[' ']' trong render |
| 2 | đếm "70 từ" theo từng khung (lỡ đếm, không bắt slide có nhiều ô nhỏ) | check_deck.py: đổi sang đếm tổng từ của cả slide | a18 slide 2: 73 → 60 từ; a19 slide 3: 92 → 62 từ; cả 20 bộ trong deck-lab/ hiện "lỗi cứng: không" |

Kết quả kiểm tra sau 2 vòng (chạy check_deck.py trên toàn bộ deck-lab/): **20 bộ,
0 bộ có lỗi cứng**. Đây là số đo cuối cùng ghi cho ngày 4, dùng làm nền cho vòng tiếp
theo (nếu còn thời gian) và cho tiêu chí E (tổng số bộ, số bộ tiếng Việt, bộ đạt).

## Vòng 3 (regression check): xác nhận 7/7 nhóm lỗi vẫn bị bắt sau refactor

Chạy lại file "cố tình có lỗi" (2 slide: out-of-bounds + chữ 10pt + chữ vàng kém tương
phản + khung rỗng + thiếu transition; slide 2: 80 từ) qua check_deck v2 (đếm từ theo
slide):
- slide 1: bắt "thiếu p:transition", "shape ra ngoài slide", "chữ 10.0pt < 12pt",
  "tương phản #FFD400 < 4.5:1 trên nền FFFFFF", "khung chữ trống" (5 nhóm, đúng như
  vòng 1-2 đã cam kết).
- slide 2: bắt "80 từ trên slide (giới hạn 70, đếm toàn slide)" — chứng nhận refactor
  vòng 2 (đổi cách đếm từ theo slide) không làm mất nhóm "quá 70 từ", chỉ đổi cách
  đếm đúng hơn.
- Nhóm "hai khung chữ chồng nhau" xác nhận bằng thử riêng (2 textbox che nhau 70-80%):
  vẫn báo cảnh báo đúng, không bị vô hiệu hóa bởi refactor.

Kết luận: sau 3 vòng, check_deck bắt được đủ 7/7 nhóm lỗi của tiêu chí C (tràn khung,
chồng khung là 2 nhóm "cảnh báo cần render" theo thiết kế, 5 nhóm còn lại là lỗi
cứng), và không có bộ nào trong 20 bộ deck-lab/ có lỗi cứng.

## Vòng 4 (nâng độ khó, sau khi đạt 2 vòng): điểm yếu "chữ tràn xuống đè chữ dưới" mà
check_deck chưa bắt được — phát hiện bằng render thật của a22 (thi thử MCP)

Bắt đầu: làm a22 (bộ thi thử thứ hai, tạo bằng tool mcp__deck__make_deck thật trong phiên)
và render slide bìa (trang 1) bằng view_image. Thấy ngay: tiêu đề bìa "Thi thử thật: gọi
mcp__deck__make_deck ngay trong phiên" wrap thành 3 dòng (vì dài), dòng 2-3 của tiêu đề đè
vào dòng phụ đề (subtitle) ngay dưới — check_deck v2 (vòng 1-3) KHÔNG báo lỗi này vì
"chồng khung" (điểm 8) trước đây chỉ so hai textbox có chữ, không so textbox chữ này với
textbox chữ khác nằm ngay dưới nó (hai box không giao nhau theo chiều ngang, chỉ theo chiều
dọc/vertically stacked). Đây là loại lỗi "chữ tràn xuống che chữ khác" mà SKILL.md mục 8
ghi là cần render xác nhận — giờ mới phát hiện thật, không phải lý thuyết.

Sửa (helpers.py, vòng 4): 
- `slide_cover`: chuyển tiêu đề từ y=2.2in lên y=1.9in, tăng chiều cao box từ 1.2in còn
  1.6in, hạ subtitle từ y=3.5in xuống y=3.9in (thừa 0.4in giữa hai box) — đủ chỗ cho
  tiêu đề wrap 3 dòng mà không che subtitle. Kiểm tra lại bằng xem coordinate: tiêu đề
  hết ở y≈3.5in (1.9+1.6), subtitle bắt đầu y=3.9in — còn khoảng trống 0.4in, không che
  nữa.
- `slide_closing`: cùng lý do, hạ message chính từ y=2.0 xuống... giữ vị trí nhưng tăng
  chiều cao box 1.6in → 2.2in và hạ phụ đề từ y=4.4 xuống y=4.6 (tránh che khi message
  2 dòng).

Dựng lại: a22_thi_thu.mcp_real (7 slide, rebuild qua make_deck sau khi sửa helpers.py),
render slide 1 lại, view_image xác nhận tiêu đề và subtitle không còn chồng nhau. Số đo
trước/sau (EVOLUTION.md vòng 4, slide 1 a22):
- Trước: tiêu đề wrap 3 dòng, dòng cuối (y≈3.4-3.9in) đè lên subtitle (y≈3.5-4.3in) —
  render cho thấy 2 dòng chữ phủ lên nhau, đọc không rõ.
- Sau: tiêu đề wrap 3 dòng nhưng dừng ở y≈3.5in; subtitle bắt đầu y=3.9in — cách nhau
  0.4in, không che. (còn có thể kiểm thêm bằng cách đo khoảng trống dọc giữa đáy box
  chữ này và đỉnh box chữ khác cùng cột — sẽ làm nếu vòng sau còn thời gian).

Ghi nhận: đây là lần đầu trong 20+ bộ slide mà điểm yếu "tràn chữ xuống che chữ dưới"
được phát hiện bằng render thật (không phải bằng heuristic tràn khung trong check_deck,
vì heuristic đó trước đây chỉ đo chiều cao của MỘT box so với chính nó, không đo box
này chạm box khác). Đây chính là lý do SKILL.md mục 8 ghi 2 điểm "tràn khung" và "chồng
khung" là "cảnh báo, cần render xác nhận" — giờ đã có bằng chứng cụ thể (a22 vòng 4)
chứng minh render là bắt buộc, không phải tùy chọn.

## Vòng 5 (nâng độ khó, cuối ngày 4): tự động hóa loại lỗi "chữ wrap che block dưới"

Bẫy: các vòng 1-4 phát hiện "tiêu đề wrap 3 dòng che phụ đề" (a22 slide bìa) bằng CÁCH NHÌN
RENDER (view_image), không có check nào trong check_deck bắt được kiểu này (điểm 8 "chồng
khung" trước đây chỉ so hai box CHỒNG LÊN NHAU theo diện tích, không so box A có chạm box B
nằm ngay dưới A). Giải pháp: thêm check mới (điểm 9) ước lượng đáy chữ của box A (số dòng
wrap × leading × cỡ chữ) và so với đỉnh box B ngay dưới cùng cột; chỉ cảnh báo khi đáy ước
VƯỢT QUA đỉnh box B ít nhất 0.03in (tránh dương tính giả với các box đặt sát nhau có chủ
đích trong card/bento — chiều rộng được lọc thêm: chỉ so cặp có chiều rộng gần bằng nhau,
tỉ lệ 0.8-1.25).

Kiểm tra đúng/sai (test riêng, không lẫn vào 22 bộ thật):
- File test "tràn thật" (tiêu đề top=2.2in h=1.2in, phụ đề top=3.4in): bắt được
  "tràn xuống che" (đáy ước 3.53in > 3.40in + 0.03in).
- File test "đặt sát có chủ đích" (tiêu đề top=1.9in h=1.6in, phụ đề top=3.9in): không
  cảnh báo (đáy ước 3.23in < 3.9in).

Kết quả áp dụng trên 22 bộ thật: check mới phát hiện 58 cảnh báo "chữ có thể che block
dưới" (không phải lỗi cứng, là cảnh báo cần render xác nhận) — số lượng lớn vì đa số tập
trung ở layout `slide_cards` (ô mô tả cao 1.8in, chữ wrap 3-4 dòng, đáy chạm gần biên card).
Đã sửa slide_cards (ô mô tả: cao 1.8in → 2.0in, đặt sát đáy card) và rebuild a1_102030rule
(1 trong 22 bộ, chưa rebuild hết 22 bộ còn lại vì hết thời gian ngày 4) — cảnh báo của a1
giảm từ "toàn slide" xuống còn 3 (chỉ còn cảnh báo ở các slide cards khác, đúng mức cần
chấp nhận: cảnh báo là hợp lệ, không phải lỗi, vì bản chất các cảnh báo này vẫn cần render
xác nhận như SKILL.md mục 8 ghi — vòng 5 chỉ giúp phát hiện sớm hơn, không xóa bỏ bước
render bằng mắt).

Bài học: check kiểu "tràn xuống che" rất khó chỉnh ngưỡng (cứ siết là dương tính giả 58
bộ, cứ nới là bỏ sót a22) — giữ nguyên ngưỡng 0.03in, coi 58 cảnh báo này là "chưa render
xác nhận" chứ không phải lỗi phải sửa hết ngay, đúng tinh thần SKILL.md mục 8 (2 nhóm
cảnh báo tràn/chồng khung không phải lỗi cứng).

## Vòng 7 (ngày 6): layout mới "kpi" (bố cục số 15) + bẫy dữ liệu delta tự ghi mũi tên

Điểm yếu: layout cards/bento chỉ hiện số + nhãn, không nói được con số đang TĂNG hay GIẢM
so với mốc trước — thiếu một loại slide "đo hiệu quả theo thời gian" chỉ trong một ô.

Sửa: thêm `slide_kpi` (1-4 ô, số lớn 56pt + dòng delta kèm mũi tên ▲/▼ và màu teal/đỏ đất
theo hướng, nhãn 15pt). Bẫy dữ liệu: (a) đầu tiên tôi tự ghi "▼ ..." vào trường delta trong
spec mà không biết helper sẽ thêm mũi tên tự động theo dấu +/- -> render ra "▼ ▼ ..." (mũi
tên kép, lỗi thật, đã bắt được bằng view_image); (b) đã sửa helper để nhận cả hai dạng:
nếu delta đã có ký tự ▲/▼/↑/↓ thì dùng nguyên văn, chỉ thêm mũi tên khi delta là con số
mang dấu +/- hay không dấu (mặc định coi là tăng).

Kết quả: a25_kpi.xuhuong (8 slide, tiếng Việt) tạo qua plugin, check_deck "LỖI CỨNG: không,
CẢNH BÁO: không", render slide 3 (kpi) xác nhận: 3 ô, số 0/15/2, mũi tên ▼49 (đỏ), ▲1
(teal), ▼10 (đỏ) — đúng chuẩn, không mũi tên kép.

Cùng vòng này: patch 8 bộ cover (a3, a5, a9, a14, a16, a17, a18, a19) nâng box tiêu đề từ
1.2in lên 1.6in và hạ phụ đề xuống y=3.9in (bẫy "chữ wrap che block dưới" mà check thứ 9
ngày 5 phát hiện); patch slide_cards (ô số lớn 48pt cao 1.2in→1.6in, ô nhãn 3.3in→3.7in,
chiều cao 0.6in→0.5in) vì chữ 48pt wrap 2 dòng (như "Quá nhiều dòng") thật sự vượt đáy box
cũ. Số đo trước/sau: 24 bộ deck-lab/ hiện 0 lỗi cứng, 0 cảnh báo (trước vòng 7: 0 lỗi cứng
nhưng 10 cảnh báo "tràn xuống che" + 3 cảnh báo "chồng nhau" ở a18).
