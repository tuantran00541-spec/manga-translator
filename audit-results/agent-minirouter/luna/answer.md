Đã tạo xong dự án Python `notekeeper` từ đầu.

### Thành phần chính

- `src/notekeeper/models.py`
  - Mô hình `Note` với UUID, tiêu đề, nội dung, tag và timestamp.
  - Kiểm tra dữ liệu và serialize/deserialize JSON.

- `src/notekeeper/storage.py`
  - `JsonNoteRepository`.
  - Thêm, liệt kê, tìm kiếm, xóa, gắn/xóa tag.
  - Tìm kiếm không phân biệt hoa thường.
  - Tự tạo file/thư mục lưu trữ.
  - Ghi JSON atomic bằng file tạm và `os.replace`.
  - Xử lý JSON hỏng và schema không hợp lệ.

- `src/notekeeper/cli.py`
  - Các lệnh:
    - `add`
    - `list`
    - `search`
    - `delete`
    - `tag`
  - Hỗ trợ `--file`.
  - Thông báo lỗi thân thiện và mã thoát phù hợp.

- `src/notekeeper/__main__.py`
  - Chạy được bằng `python -m notekeeper`.

- Cấu hình dự án:
  - `pyproject.toml`
  - `pytest.ini`
  - `README.md`

- Tài liệu thiết kế và kế hoạch:
  - `docs/specs/2026-02-14-notekeeper-design.md`
  - `docs/plans/2026-02-14-notekeeper.md`

### Kiểm thử

Đã chạy toàn bộ test:

```text
14 passed in 0.03s
```

Đã chạy smoke test end-to-end với file JSON tạm, bao gồm:

```text
add -> list -> search -> tag -> search -> delete -> list
```

Kết quả:

```text
smoke test passed
```

Lưu ý: `pip install -e .` không chạy được trong sandbox vì môi trường thiếu `setuptools.build_meta` và bị chặn mạng khi tải build dependency. Cấu hình `pyproject.toml` vẫn dùng backend setuptools chuẩn; test và smoke test đã được xác nhận bằng `PYTHONPATH=src`.