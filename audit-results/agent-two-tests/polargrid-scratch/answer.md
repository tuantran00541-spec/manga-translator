Advisor đúng — pytest 8.3.5 hỗ trợ `pythonpath` natively (đã xác nhận trong `_pytest/python_path.py`). Tôi đã áp dụng cách chuẩn:

- **Thêm** `pythonpath = ["src"]` vào `[tool.pytest.ini_options]` trong `pyproject.toml`
- **Xóa** `tests/conftest.py` (shim sys.path)
- **Chạy lại pytest: 39 passed** ✅

Bây giờ src layout hoạt động theo cách chuẩn, không cần hack. `pyproject.toml` đã được verify thực tế (không chỉ tồn tại).

### Trạng thái cuối
- 39/39 test xanh
- `pythonpath = ["src"]` trong pyproject.toml thay cho conftest shim
- Console script `notekeeper` được khai báo (chạy được khi `pip install -e .` trên máy có mạng; trong sandbox này không cài được vì không có mạng + không có setuptools)
- Cấu trúc: `pyproject.toml`, `README.md`, `DESIGN.md`, `src/notekeeper/{__init__,storage,cli}.py`, `tests/{__init__,test_storage,test_cli}.py`

Dự án hoàn thành.