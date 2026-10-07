# Kiểm tra nhanh 10 điểm (check_deck thô, ngày 1)

Chạy trên mọi .pptx trong deck-lab/ — phát hiện 8 nhóm lỗi (không gồm tương phản & chữ tràn
khung, cần render):

1. Thiếu `<p:transition>` trên slide
2. Shape vượt ra ngoài biên slide (out-of-bounds, dung sai 1000 EMU)
3. Khung chữ trống (textbox không có fill; hình trang trí có fill không bị đếm)
4. Chữ nhỏ hơn 12pt
5. Slide có hơn 70 từ
6. Hai khung chữ/charts chồng lên nhau (hộp chứa giao nhau > 30% diện tích nhỏ hơn)
7. Ước lượng chữ tràn khung: số dòng × leading > chiều cao frame
8. (chưa tự động, cần render + view_image): tương phản chữ/nền < 4.5:1

Cài đặt tương lai: gói thành plugin `make_deck`/`check_deck` (tiêu chí C) với JSON schema đầy đủ.
