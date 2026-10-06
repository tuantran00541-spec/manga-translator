Xong. Dự án **notekeeper** đã được tạo từ số 0, test chạy xanh hết.

## Đã tạo
```
pyproject.toml               # setuptools, src layout, entry point `notekeeper`, cấu hình pytest
README.md                    # hướng dẫn cài đặt, dùng, cấu trúc
src/notekeeper/
    __init__.py
    storage.py               # class Storage: all/add/find/delete, ghi JSON nguyên tử
    cli.py                   # argparse: add / list / find / delete
tests/
    conftest.py              # thêm src vào sys.path
    test_notekeeper.py       # 20 test (storage + CLI)
```

## Cách làm
- Tôi dựng khung (pyproject, README, `__init__.py`) trước, sau đó **chạy 3 agent con song song**: một viết `storage.py`, một viết `cli.py`, một viết test — mỗi agent code đúng theo API công khai đã định sẵn.
- Agent test bị treo nên tôi đóng nó lại và tự viết test thay.

## Kiểm tra
- `pytest -q` → **20 passed**. Có 1 lỗi ban đầu (tag nối thiếu dấu cách `[a,b]` vs `[a, b]`), đã sửa trong `_format_note` của `cli.py` rồi chạy lại xanh.
- Smoke test CLI thật: add 2 ghi chú có tag → list → find theo tag → delete → list, mọi thứ đúng.

## Ghi chú
- File dữ liệu: `notes.json` (đổi bằng biến môi trường `NOTEKEEPER_FILE`).
- Ghi chú dạng `{"id", "content", "tags", "created"}`; ghi JSON bằng `os.replace` (nguyên tử), file hỏng JSON → báo lỗi rõ ràng, exit code 1.