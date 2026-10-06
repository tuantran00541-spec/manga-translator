RED đúng như mong đợi cho `NoteBook`: `ModuleNotFoundError: No module named 'notekeeper.notebook'` — 15 test mới hoàn toàn chưa có code hiện thực.

**Tình trạng tổng thể:**
- Hoàn thành: `pyproject.toml`, `src/notekeeper/models.py` + 8 test xanh, `src/notekeeper/storage.py` + 8 test xanh.
- Vừa viết: `tests/test_notebook.py` (15 test cho `NoteBook`: add gán id tăng dần, từ chối text rỗng, list có lọc tag không phân biệt hoa/thường, find trong text + tags, add_tag idempotent, delete trả True/False, id không tái sử dụng sau khi xóa, hai instance dùng chung file).
- Chưa hiện thực: `src/notekeeper/notebook.py` — vòng RED đã được xác nhận.

**Bước tiếp theo:** hiện thực `NoteBook` tối thiểu để qua 15 test trên (nạp state qua `Storage.load()`, `add` dùng `next_id` tăng dần rồi `save`, các thao tác thuần túy trên list rồi save lại), chạy test xem xanh. Sau đó là vòng TDD cho CLI (`main(argv)` trả exit code), rồi `__init__.py`/`__main__.py`, README, `.gitignore`, và cuối cùng chạy toàn bộ `pytest` đến khi xanh hết + smoke-test CLI end-to-end.