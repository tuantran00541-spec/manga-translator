# Manga & Webtoon Translator

Một ứng dụng web local-first phục vụ quy trình biên tập và dịch thủ công manga / manhwa / webtoon. Công cụ cung cấp pipeline tự động hóa các bước crawl, cắt ảnh, phát hiện khung thoại/chữ, xóa chữ gốc (inpaint), OCR đọc chữ gốc và giao diện web để người dùng tự nhập bản dịch và dàn trang (typesetting).

> [!IMPORTANT]
> **Công cụ KHÔNG tích hợp dịch tự động (NO Automatic AI Translation)**  
> Ứng dụng **không** sử dụng LLM hay API dịch máy để tự động dịch nội dung. OCR chỉ phục vụ đọc và trích xuất chữ gốc từ hình ảnh. Bản dịch (tiếng Việt hoặc ngôn ngữ mục tiêu) **bắt buộc phải do người dùng tự nhập thủ công** vào trình biên tập web.

> [!NOTE]
> **Định hướng công cụ:**  
> Thiết kế tối ưu cho **chạy offline, local-first trên CPU** dành cho cá nhân hoặc nhóm nhỏ. Thuật toán nhận diện và inpaint dựa trên mô hình Computer Vision / AI cục bộ, có sai số nhất định và cần sự can thiệp / tinh chỉnh thủ công của con người.

---

## 🛠️ Luồng hoạt động (Pipeline)

1. **Thu thập & Tải lên (Ingestion):** Tải chapter qua URL (hỗ trợ trang render JavaScript bằng Playwright) hoặc upload trực tiếp ảnh lẻ, file ZIP / CBZ.
2. **Cắt lát Webtoon (Webtoon Slicing):** Tự động phân tích đường cắt thông minh (tránh cắt trúng chữ/vùng nội dung) để chia nhỏ các trang webtoon dài thành nhiều trang chuẩn.
3. **Phát hiện khung thoại & Chữ (Detection):** Sử dụng 2 mô hình YOLOv8 ONNX kết hợp **Unified NMS** để gom nhóm, lọc trùng và xác định chính xác vị trí khung thoại (speech bubbles) cùng các dòng chữ độc lập.
4. **Xóa chữ gốc (Inpainting):** Xóa chữ gốc bằng mô hình LaMa (Large Mask Inpainting) kết hợp kỹ thuật ring-sampling fallback ở viền mask.
5. **Nhận diện chữ gốc (Multi-lang OCR):** Tự động đọc chữ từ vùng phát hiện. Hỗ trợ tiếng Nhật (MangaOCR), tiếng Trung, Hàn, Anh (PaddleOCR).
6. **Biên tập & Chèn chữ thủ công (Web Editor & Typesetting):** Giao diện web cho phép preview, tô sửa bù mask sót, nhập trực tiếp bản dịch tiếng Việt, tùy chỉnh font, kích thước, màu sắc, viền chữ (stroke) và canh lề.
7. **Xuất kết quả (Export):** Xuất toàn bộ trang đã dàn trang hoàn chỉnh ra thư mục `data/output/<chapter_id>/`.

---

## 🚀 Cài đặt & Chạy nhanh

### Cách 1: Sử dụng Docker (Khuyên dùng)

**Yêu cầu:** Docker Desktop & Docker Compose.

```bash
# 1. Clone repository
git clone https://github.com/tuantran00541-spec/manga-translator.git
cd manga-translator

# 2. Tải 3 file ONNX bắt buộc vào thư mục models/ (Xem chi tiết ở mục "Mô hình AI")

# 3. Khởi chạy container
docker compose up --build

# 4. Truy cập giao diện web tại:
# http://127.0.0.1:8000
```

Các thư mục `data/`, `models/`, và `logs/` được bind-mount trực tiếp từ host, đảm bảo dữ liệu không bị mất khi restart container.

### Cách 2: Chạy trực tiếp với Python

**Yêu cầu:** Python 3.10 – 3.12, RAM khuyến nghị >= 8 GB.

```bash
# 1. Clone repository
git clone https://github.com/tuantran00541-spec/manga-translator.git
cd manga-translator

# 2. Tạo và kích hoạt môi trường ảo
python -m venv venv
# Linux / macOS:
source venv/bin/activate
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# 3. Cài đặt thư viện phụ thuộc
pip install -r requirements.txt

# 4. Cài đặt Chromium headless cho Playwright (dùng crawl URL)
playwright install chromium

# 5. Tải 3 file ONNX bắt buộc vào thư mục models/ (Xem chi tiết ở mục "Mô hình AI")

# 6. Khởi chạy server
python run.py
```

Sau khi khởi chạy thành công, mở trình duyệt truy cập `http://127.0.0.1:8000`.

---

## 📦 Yêu cầu Mô hình AI (ONNX Models)

Ứng dụng yêu cầu **3 file mô hình ONNX** đặt chính xác trong thư mục `models/`:

| Tên file ONNX | Mục đích | Nguồn tham chiếu / Chuyển đổi |
|---|---|---|
| `bubble_yolo.onnx` | Detect khung thoại (Speech Bubbles) | [ogkalu/comic-speech-bubble-detector-yolov8m](https://huggingface.co/ogkalu/comic-speech-bubble-detector-yolov8m) |
| `text_segmenter.onnx` | Detect dòng chữ / text block | [ogkalu/comic-text-segmenter-yolov8m](https://huggingface.co/ogkalu/comic-text-segmenter-yolov8m) |
| `lama.onnx` | Inpaint xóa chữ gốc | [Carve/LaMa-ONNX](https://huggingface.co/Carve/LaMa-ONNX) (`lama_fp32.onnx`) |

#### Chuyển đổi weights YOLO (`.pt`) sang ONNX:
Nếu tải file trọng số YOLO dạng `.pt` từ HuggingFace/Ultralytics, hãy chạy script hỗ trợ sẵn:

```bash
pip install ultralytics
python convert_model.py path/to/model.pt
# Đổi tên file .onnx đầu ra thành bubble_yolo.onnx hoặc text_segmenter.onnx rồi copy vào thư mục models/
```

*Ghi chú:* Khi server khởi động, ứng dụng sẽ kiểm tra thư mục `models/`. Nếu thiếu mô hình, hệ thống vẫn khởi chạy nhưng sẽ thông báo cảnh báo rõ ràng qua log và endpoint `/health` (`"models_missing": [...]`).

---

## ⚙️ Cấu hình Biến Môi Trường (Configuration)

Các thông số vận hành được cấu hình qua biến môi trường (Environment Variables) hoặc file cấu hình `app/config.py`:

| Biến môi trường | Mặc định | Mô tả |
|---|---|---|
| `HOST` | `127.0.0.1` | Địa chỉ IP bind server. Đặt `0.0.0.0` nếu muốn truy cập từ mạng LAN. |
| `PORT` | `8000` | Cổng dịch vụ HTTP (TCP Port). |
| `WORKERS` | `1` | Số lượng uvicorn worker process. **Khuyến nghị giữ 1** do các mô hình AI chạy nặng trên CPU và worker Playwright không tối ưu cho đa tiến trình. |
| `ENABLE_TTA` | `0` | Test-Time Augmentation cho YOLO bubble detection. Đặt `1` để bật (tăng độ chính xác detect nhưng thời gian xử lý lâu hơn). |
| `RELOAD` | `0` | Đặt `1` để tự động reload ứng dụng khi thay đổi code (dùng trong Development). |

Ví dụ khởi chạy cho phép truy cập từ LAN:
```bash
HOST=0.0.0.0 PORT=8000 python run.py
```

---

## 🔒 Tính năng Bảo mật (Security & Hardening)

Dù được thiết kế chính cho môi trường local/cá nhân, ứng dụng đã được gia cố sẵn các cơ chế bảo mật tiêu chuẩn:

- **SSRF Protection (`app/security.py`):** Kiểm tra và chặn toàn bộ request tải URL hướng tới IP nội bộ (Private IPs), Loopback (`127.0.0.1`, `localhost`), Link-local / Cloud Metadata (`169.254.169.254`), CGNAT, Multicast và các URL scheme không phải HTTP/HTTPS.
- **Path Traversal Defense:** `chapter_id` được kiểm tra nghiêm ngặt bằng regex (chỉ chấp nhận chuỗi hex 8 ký tự `^[a-f0-9]{8}$`). Bỏ hoàn toàn việc mount tĩnh toàn bộ thư mục `/data`, hình ảnh được phục vụ an toàn qua endpoint kiểm soát danh tính `/api/image/...`.
- **Atomic File Locking & Data Consistency:** Sử dụng `filelock` trên `manifest.json` cho từng chapter và ghi dữ liệu nguyên tử (atomic write qua file tạm `.tmp` rồi `os.replace`), chống race condition khi nhiều request cùng thao tác.
- **Giới hạn Request & Chống DoS bộ nhớ:** Tích hợp `RequestSizeLimitMiddleware` (max 50 MB/request), kiểm tra kích thước danh sách Pydantic schemas, đồng thời tuân thủ giới hạn điểm ảnh `PIL.MAX_IMAGE_PIXELS` khi giải mã ảnh bằng OpenCV.
- **An toàn Tiến trình & Sanitize lỗi:** Khởi tạo OCR thread-safe (tránh leak bộ nhớ khi gọi đồng thời) và giấu stack-trace chi tiết ở môi trường production (trả về lỗi HTTP 500 chuẩn hóa).

---

## 📂 Cấu trúc Thư mục (Directory Structure)

```text
manga-translator/
├── app/
│   ├── detector/             # YOLO Bubble & Text Detector, Unified NMS
│   │   ├── bubble_detector.py
│   │   ├── combined_detector.py
│   │   └── mask_builder.py
│   ├── downloader/           # Crawl URL (Playwright) & Slicer webtoon
│   │   ├── base.py
│   │   ├── generic_js.py
│   │   ├── playwright_worker.py
│   │   ├── registry.py
│   │   └── slicer.py
│   ├── inpaint/              # LaMa Inpainter & Ring-sampling fill
│   │   └── lama_inpainter.py
│   ├── ocr/                  # Multi-language OCR (MangaOCR & PaddleOCR)
│   │   └── multi_lang_ocr.py
│   ├── render/               # Typesetting & Render chữ lên ảnh
│   │   └── text_renderer.py
│   ├── routers/              # API Endpoints (chapters, editor, render, image)
│   │   ├── chapters.py
│   │   ├── editor.py
│   │   ├── image.py
│   │   └── render.py
│   ├── static/               # CSS, JS (Canvas editor UI), Fonts (.ttf)
│   ├── templates/            # HTML templates (index.html)
│   ├── config.py             # File cấu hình trung tâm & biến môi trường
│   ├── main.py               # FastAPI app & Middleware
│   ├── pipeline.py           # Orchestration điều phối pipeline xử lý
│   ├── security.py           # Module kiểm tra bảo mật (SSRF, Path Traversal)
│   └── manifest_utils.py     # Quản lý đọc/ghi manifest.json
├── data/                     # Thư mục chứa dữ liệu runtime (Git-ignored)
│   ├── raw/                  # Ảnh gốc / trang đã slice
│   ├── processed/            # Mask & ảnh sau khi xóa chữ (inpainted)
│   └── output/               # Ảnh kết quả sau khi chèn bản dịch
├── models/                   # Đặt 3 file .onnx (bubble_yolo, text_segmenter, lama)
├── logs/                     # File log ứng dụng
├── convert_model.py          # Script convert PyTorch (.pt) sang ONNX (.onnx)
├── debug_detect.py           # Script debug nhận diện
├── run.py                    # Entry point khởi chạy ứng dụng
├── Dockerfile                # File build Docker image
├── docker-compose.yml        # File cấu hình Docker Compose
└── requirements.txt          # Danh sách thư viện Python
```

---

## 💻 Hướng dẫn Phát triển (Development Instructions)

### Bật chế độ Auto-Reload khi Dev
```bash
RELOAD=1 python run.py
```

### Kiểm tra trạng thái hệ thống (Health Check)
```bash
curl http://127.0.0.1:8000/health
```
**Response mẫu:**
```json
{
  "status": "ok",
  "models_missing": []
}
```

---

## 📜 Giấy phép (License)

Dự án phát hành theo giấy phép **MIT License**.
