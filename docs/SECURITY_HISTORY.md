# Security Changelog

Thay đổi bảo mật cho `manga-translator`. Nếu bạn fork từ phiên bản trước commit này, **bắt buộc áp dụng các patch dưới đây** trước khi public repo hoặc deploy lên VPS.

## [security-1] — 2026-07-31

### Vấn đề nghiêm trọng đã vá

| CVE ref (không chính thức) | Mô tả | File |
|---|---|---|
| SSRF | `POST /api/chapter` chấp nhận mọi URL → có thể fetch `file://`, `http://169.254.169.254/`, IP private, scan mạng nội bộ | `app/security.py` (mới), `app/main.py`, `app/downloader/registry.py` |
| Path Traversal | `chapter_id` nối thẳng vào `Path / chapter_id` → `..%2F..%2Fetc%2Fpasswd` đọc file ngoài thư mục dự án | `app/security.py` (mới), `app/main.py` |
| Race condition | 2 request cùng thao tác trên `manifest.json` → request sau ghi đè request trước, mất dữ liệu dịch | `app/pipeline.py` (dùng `filelock`) |
| Thread-unsafe OCR init | 2 thread cùng init PaddleOCR/MangaOCR → leak memory + race | `app/ocr/multi_lang_ocr.py` |
| DoS qua upload | `/api/repaint_mask` không giới hạn size PNG upload → có thể gửi 500MB crash server | `app/main.py` (middleware `RequestSizeLimitMiddleware`) |

### File mới

- `app/security.py` — Module tập trung các hàm validation:
  - `validate_url(url)` — block scheme không hợp lệ + IP private/loopback/link-local/CGNAT/multicast
  - `validate_chapter_id(id)` — regex `^[a-f0-9]{8}$`
  - `MAX_REQUEST_BYTES = 50 MB` — hằng số giới hạn upload

### File sửa đổi

- `app/main.py` — middleware size limit, validate chapter_id/url, safe image endpoints, generic 500 handler
- `app/downloader/registry.py` — defense-in-depth URL checks
- `app/pipeline.py` — filelock per-chapter, atomic manifest writes
- `app/ocr/multi_lang_ocr.py` — thread-safe OCR init

### Cách kiểm tra

```bash
curl "http://127.0.0.1:8000/api/chapter/..%2F..%2Fetc%2Fpasswd"
# Kỳ vọng: 400 Invalid chapter_id

curl -X POST http://127.0.0.1:8000/api/chapter \
  -H "Content-Type: application/json" \
  -d '{"url": "http://169.254.169.254/latest/meta-data/"}'
# Kỳ vọng: 400 blocked IP
```

## [security-2] — 2026-07-31

### Vấn đề P0 còn lại đã vá

| Ref | Mô tả | File |
|---|---|---|
| A3 | `cv2.imdecode` không tuân thủ `PIL.MAX_IMAGE_PIXELS` | `app/pipeline.py` |
| A4 | `RenderRequest.translations` không giới hạn size | `app/main.py` (Pydantic validators) |
| A5 | `manifest.json` ghi trực tiếp → crash giữa chừng hỏng file | atomic write `*.tmp` + `os.replace()` |
| A6 | `uvicorn.run(reload=True)` hardcode | `run.py` đọc env `RELOAD` |
| A7 | Playwright Chromium sandbox | `generic_js.py` |
| A8 | `StaticFiles` mount `/data` expose path traversal | bỏ mount, dùng `/api/image/...` |
| A10 | Stack trace 500 leak | exception handler trả 500 chung |

### Còn lại (P1/P2)

- [x] Dockerfile + docker-compose
- [ ] Auth (nếu public): HTTP Basic Auth qua reverse proxy hoặc JWT

> Dự án này **chỉ chạy CPU** (hardcoded `CPUExecutionProvider`). Không hỗ trợ GPU/CUDA — giữ project nhẹ cho máy cá nhân.

## [quality-1] — 2026-07-31

### Tinh chỉnh code quality

- Logging structured (`loguru`), toast UX, tách editor modules, type hints
- Dependency mới: `loguru`, `filelock`

---

Người vá: security audit  
Ngày: 2026-07-31
