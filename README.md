# Manga & Webtoon Translator

**Offline · CPU-only · Local-first**

Tự động dịch manga / manhwa / webtoon: tải chapter từ URL (hoặc upload ảnh/ZIP), cắt lát webtoon dài, detect bong bóng thoại + vùng chữ, xóa chữ thông minh (LaMa + smart fill), OCR đa ngôn ngữ, và chèn bản dịch qua giao diện web local.

> Designed for **local / single-user** use.  
> Có thể self-host cho nhóm nhỏ (xem phần Docker & Security).

---

## ✨ Tính năng chính

- Tải chapter từ URL (hỗ trợ trang JS-rendered qua Playwright)
- Upload hàng loạt ảnh hoặc file ZIP / CBZ
- Tự động cắt lát webtoon dài thành trang vừa vặn
- Detect bong bóng thoại + vùng chữ (YOLO ONNX)
- Smart inpaint (LaMa + ring-sampling fallback)
- OCR: Nhật (MangaOCR), Trung / Hàn / Anh (PaddleOCR)
- Giao diện web: preview → review (tô lỗi) → editor (màu chữ, font) → xuất
- Chạy hoàn toàn trên CPU, không cần GPU

---

## 🚀 Cài đặt nhanh

### Cách 1: Docker (khuyên dùng)

```bash
# 1. Clone
git clone https://github.com/tuantran00541-spec/manga-translator.git
cd manga-translator

# 2. Tải 3 model ONNX vào thư mục models/  (xem mục "Mô hình AI" bên dưới)

# 3. Chạy
docker compose up --build

# 4. Mở trình duyệt
# http://127.0.0.1:8000
```

Dữ liệu chapter và model được mount từ thư mục host → không mất khi restart container.

### Cách 2: Chạy trực tiếp (Python)

**Yêu cầu:** Python 3.10–3.12, ~8 GB RAM khuyến nghị.

```bash
git clone https://github.com/tuantran00541-spec/manga-translator.git
cd manga-translator

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
playwright install chromium       # chỉ cần 1 lần

# Tải 3 model ONNX vào models/ (xem bên dưới)

python run.py
```

Mở http://127.0.0.1:8000

#### Biến môi trường hữu ích

| Biến     | Mặc định    | Mô tả                          |
|----------|-------------|-------------------------------|
| `HOST`   | `127.0.0.1` | Đặt `0.0.0.0` nếu cần truy cập LAN |
| `PORT`   | `8000`      | Cổng lắng nghe                 |
| `RELOAD` | `0`         | `1` = auto-reload khi dev      |

Ví dụ chạy LAN:

```bash
HOST=0.0.0.0 python run.py
```

---

## 📦 Chuẩn bị Mô hình AI

Đặt **3 file ONNX** vào thư mục `models/`:

| File | Nguồn | Ghi chú |
|------|-------|--------|
| `bubble_yolo.onnx` | [ogkalu/comic-speech-bubble-detector-yolov8m](https://huggingface.co/ogkalu/comic-speech-bubble-detector-yolov8m) | Convert từ `.pt` bằng `convert_model.py` |
| `text_segmenter.onnx` | [ogkalu/comic-text-segmenter-yolov8m](https://huggingface.co/ogkalu/comic-text-segmenter-yolov8m) | Convert từ `.pt` bằng `convert_model.py` |
| `lama.onnx` | [Carve/LaMa-ONNX](https://huggingface.co/Carve/LaMa-ONNX) (`lama_fp32.onnx`) | Đổi tên thành `lama.onnx` |

**Convert YOLO → ONNX:**

```bash
pip install ultralytics
# Tải file .pt từ HuggingFace, rồi:
python convert_model.py path/to/comic-speech-bubble-detector.pt
# Đổi tên file .onnx vừa tạo thành bubble_yolo.onnx và copy vào models/
```

Khi server khởi động, nếu thiếu model sẽ hiện cảnh báo rõ ràng (không crash).  
Endpoint `/health` trả về `{"status": "ok"|"degraded", "models_missing": [...]}`.

---

## 🔤 Font chữ

Font mặc định: `app/static/fonts/default.ttf`.  
Có sẵn nhiều font comic trong `app/static/fonts/`.  
Muốn thêm font tiếng Việt đẹp → thả file `.ttf` vào thư mục đó.

---

## 🖥️ Hướng dẫn sử dụng

1. Dán URL chapter **hoặc** kéo thả ảnh / ZIP vào trang.
2. Chọn ngôn ngữ nguồn (ja / ch / korean / en).
3. Preview → bỏ qua trang không có chữ (nếu muốn).
4. **Xử lý các trang đã chọn** → detect + inpaint + OCR.
5. Review: tô vùng còn sót chữ → xử lý lại.
6. Editor: nhập bản dịch, chọn màu / font → **Chèn chữ**.
7. Kết quả nằm trong `data/output/<chapter_id>/` (đúng thứ tự trang).

---

## 🔒 Bảo mật (đọc trước khi mở ra mạng)

Mặc định bind `127.0.0.1` → chỉ máy local truy cập được.

**Đã được vá:**
- Path Traversal (`chapter_id` chỉ chấp nhận 8 ký tự hex)
- SSRF (block private IP, link-local, cloud metadata, scheme nguy hiểm)
- Race condition trên `manifest.json` (filelock)
- Thread-safe OCR init
- Giới hạn kích thước request / ảnh / số file upload
- Không leak stack-trace ra client

**Khi self-host cho nhiều người / LAN:**
- Đặt sau reverse proxy (Caddy / Nginx) + HTTPS + Basic Auth (hoặc OAuth)
- Bật rate-limit ở proxy
- Firewall chỉ cho IP tin cậy
- Không chạy container với quyền root

Xem chi tiết trong `CHANGELOG_SECURITY.md`.

---

## 📂 Cấu trúc thư mục

```text
manga-translator/
├── app/                  # Source code
│   ├── main.py           # FastAPI app
│   ├── pipeline.py       # Orchestration
│   ├── detector/         # YOLO bubble + text
│   ├── inpaint/          # LaMa + smart fill
│   ├── ocr/              # Multi-lang OCR
│   ├── downloader/       # URL crawl + slicer
│   ├── render/           # Text rendering
│   └── static/, templates/
├── data/                 # Runtime data (gitignored)
│   ├── raw/
│   ├── processed/
│   └── output/
├── models/               # Đặt 3 file .onnx vào đây
├── logs/
├── Dockerfile
├── docker-compose.yml
├── run.py
└── requirements.txt
```

---

## 🛠️ Development

```bash
# Dev với auto-reload
RELOAD=1 python run.py

# Kiểm tra health
curl http://127.0.0.1:8000/health
```

---

## License

MIT (hoặc giấy phép bạn chọn khi public repo).

---

## Ghi chú

- Tool này **local-first**. Không được thiết kế sẵn cho multi-tenant SaaS.
- Model ONNX không được phân phối kèm repo (dung lượng lớn + license riêng) → bạn tự tải.
- CPU-only: xử lý chapter dài sẽ mất thời gian, hãy kiên nhẫn hoặc chỉ process vài trang mỗi lần.
