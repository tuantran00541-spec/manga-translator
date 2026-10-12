Làm một dự án Python nhỏ tên `kvstore` ngay trong thư mục hiện tại (đang trống), chỉ dùng thư viện chuẩn. Lập todo gồm đúng 5 việc dưới đây rồi làm lần lượt, xong việc nào đánh dấu việc đó:

1. `kvstore/store.py`: class `Store(clock=time.time)` có `set(key, value, ttl=None)`, `get(key, default=None)`, `delete(key) -> bool` (True nếu có xóa), `keys() -> list` (đã sắp xếp). Giá trị là bất kỳ thứ gì ghi được ra JSON. `from kvstore import Store` phải dùng được.
2. TTL: `ttl` tính bằng giây theo `clock`. Khóa hết hạn thì `get` trả `default`, `keys()` không liệt kê nó, `delete` trả False.
3. Lưu file: `store.save(path)` ghi JSON (kèm thời điểm hết hạn), `Store.load(path, clock=time.time)` đọc lại; khóa đã hết hạn lúc nạp thì bỏ qua.
4. CLI: `python -m kvstore --file data.json set KEY VALUE [--ttl N]`, `get KEY` (in giá trị dạng JSON; không có khóa thì in ra stderr và thoát mã 1), `del KEY`, `keys` (mỗi dòng một khóa). VALUE là JSON hợp lệ thì lưu theo JSON (ví dụ `3` là số, `[1,2]` là list), không thì lưu chuỗi. File chưa có thì coi như kho rỗng.
5. `tests/test_store.py` dùng pytest phủ cả 4 phần trên và chạy pass, cộng một `README.md` ngắn.

Xong hết thì báo cáo ngắn: đã làm gì, kiểm tra thế nào.
