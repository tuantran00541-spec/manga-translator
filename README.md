# Manga & Webtoon Translator

**Local-first manga / manhwa / webtoon editor** — tự động hóa phần nặng, để người dùng tập trung vào **dịch và typeset**.

> **Không có dịch tự động.** OCR chỉ đọc chữ gốc; bản dịch do người dùng tự nhập.
>
> Chạy **local / offline / CPU-only**, phù hợp cho cá nhân hoặc nhóm dịch nhỏ.

---

## ✨ Làm được gì?

| Bước | Chức năng |
|---|---|
| 📥 **Import** | Crawl chapter từ URL hoặc upload ảnh / ZIP / CBZ |
| ✂️ **Slicing** | Chia webtoon dài thành các page an toàn, hạn chế cắt xuyên nội dung |
| 🔎 **Detection** | Phát hiện speech bubble và text block bằng YOLO ONNX |
| 🧹 **Inpaint** | Xóa chữ gốc bằng LaMa trên CPU |
| 🔤 **OCR** | Đọc chữ Nhật, Trung, Hàn, Anh |
| 🖌️ **Manual repair** | Brush / magic wand để cứu các vùng detector hoặc inpaint bỏ sót |
| ✍️ **Translation & Typesetting** | Tự nhập bản dịch, chỉnh font, size, màu, stroke, alignment |
| 📤 **Export** | Xuất chapter đã hoàn thiện ra ảnh |

### Workflow

```text
Chapter
   ↓
Import / Crawl
   ↓
Webtoon Slicing
   ↓
Bubble + Text Detection
   ↓
LaMa Inpaint ──────┐
   ↓               │
OCR ───────────────┤
   ↓               │
Review / Manual Fix│
   ↓               │
Translate + Typeset│
   ↓               │
Export ◄───────────┘
```

Công cụ **không cố tự động làm mọi thứ**. Mục tiêu là tự xử lý phần lớn case thông thường, còn editor có thể nhanh chóng sửa các case khó bằng giao diện web.

---

## 🚀 Quick Start

### Docker — khuyên dùng

```bash
git clone https://github.com/tuantran00541-spec/manga-translator.git
cd manga-translator
```

Đặt 3 model ONNX vào `models/` (xem [Models](#-models)), sau đó:

```bash
docker compose up --build
```

Mở **http://127.0.0.1:8000**.

### Python

**Python 3.10–3.12 · RAM khuyến nghị ≥ 8 GB**

```bash
git clone https://github.com/tuantran00541-spec/manga-translator.git
cd manga-translator

python -m venv venv
# Linux / macOS
source venv/bin/activate
# Windows PowerShell
.\venv\Scripts\Activate.ps1

pip install -r requirements.txt
playwright install chromium
python run.py
```

Sau đó mở **http://127.0.0.1:8000**.

---

## 🤖 Models

Đặt đúng 3 file sau trong `models/`:

| File | Dùng cho | Nguồn |
|---|---|---|
| `bubble_yolo.onnx` | Speech bubble detection | [ogkalu/comic-speech-bubble-detector-yolov8m](https://huggingface.co/ogkalu/comic-speech-bubble-detector-yolov8m) |
| `text_segmenter.onnx` | Text detection | [ogkalu/comic-text-segmenter-yolov8m](https://huggingface.co/ogkalu/comic-text-segmenter-yolov8m) |
| `lama.onnx` | Background reconstruction / text removal | [Carve/LaMa-ONNX](https://huggingface.co/Carve/LaMa-ONNX) (`lama_fp32.onnx`) |

Nếu đang có YOLO weights `.pt`, có thể chuyển sang ONNX bằng:

```bash
pip install ultralytics
python convert_model.py path/to/model.pt
```

Đổi tên output thành `bubble_yolo.onnx` hoặc `text_segmenter.onnx` tương ứng.

> Thiếu model không làm server crash ngay khi startup; trạng thái thiếu model được báo qua log và `/health`.

---

## 🧠 OCR & Translation Philosophy

Dự án **không dùng LLM để tự dịch**.

OCR được dùng để trích xuất chữ gốc nhằm hỗ trợ editor. Các loại text có ý nghĩa như **dialogue, narration, caption, tên chiêu thức / kỹ năng và stylized text** vẫn được xem là nội dung cần OCR; không mặc định coi chữ lớn hoặc stylized là SFX.

SFX thuần túy có thể được bỏ qua tùy workflow của editor.

---

## ⚡ CPU Performance

Pipeline được tối ưu cho môi trường **CPU-only** và xử lý nhiều page đồng thời.

- LaMa dùng ONNX Runtime với CPU execution.
- Default intra-op thread pool được điều chỉnh theo CPU.
- Máy có ≥ 8 logical CPUs dùng **8 LaMa threads** mặc định.
- Pipeline page workers hiện được giới hạn để tránh tạo quá nhiều CPU contention.
- Có thể override số LaMa threads bằng:

```bash
MANGA_ORT_INTRA_OP_THREADS=4 python run.py
```

hoặc:

```bash
MANGA_ORT_INTRA_OP_THREADS=8 python run.py
```

> Benchmark nội bộ cho thấy cấu hình **LaMa 8 threads + 2 page workers** nhanh hơn đáng kể trên workload CPU đã kiểm thử. Đây là default thực dụng, không phải con số tối ưu tuyệt đối cho mọi CPU.

---

## ⚙️ Configuration

Các biến môi trường chính:

| Variable | Default | Mô tả |
|---|---:|---|
| `HOST` | `127.0.0.1` | Địa chỉ bind server |
| `PORT` | `8000` | HTTP port |
| `WORKERS` | `1` | Uvicorn worker processes; nên giữ `1` vì model AI chạy nặng trên CPU |
| `ENABLE_TTA` | `0` | Bật TTA cho bubble detection; chính xác hơn nhưng chậm hơn |
| `RELOAD` | `0` | Auto-reload khi development |
| `MANGA_ORT_INTRA_OP_THREADS` | auto | Override số thread CPU của ONNX Runtime |

Ví dụ:

```bash
HOST=0.0.0.0 PORT=8000 python run.py
```

---

## 🔒 Security & Hardening

Dù chủ yếu chạy local, ứng dụng đã có các lớp hardening cho những điểm nhạy cảm:

- **SSRF protection** khi crawl URL, chặn private / loopback / link-local / metadata addresses và scheme không hợp lệ.
- **Path traversal protection** với chapter IDs và endpoint phục vụ ảnh có kiểm soát.
- **Atomic manifest writes + file locking** để tránh race condition khi cập nhật chapter state.
- **Request / image size limits** nhằm hạn chế memory abuse.
- **Sanitized production errors** và OCR initialization có kiểm soát.

---

## 📁 Project Structure

```text
manga-translator/
├── app/
│   ├── detector/       # Bubble / text detection + mask building
│   ├── downloader/     # URL ingestion + webtoon slicing
│   ├── inpaint/        # LaMa inpainting
│   ├── ocr/            # Multi-language OCR
│   ├── render/         # Translation typesetting / rendering
│   ├── routers/        # API endpoints
│   ├── static/         # Web editor UI
│   ├── templates/      # HTML templates
│   ├── config.py       # Application configuration
│   ├── main.py         # FastAPI application
│   ├── pipeline.py     # Pipeline orchestration
│   ├── security.py     # Security controls
│   └── manifest_utils.py
├── data/               # Runtime data (git-ignored)
│   ├── raw/
│   ├── processed/
│   └── output/
├── models/             # Local ONNX models
├── logs/               # Application logs
├── convert_model.py    # YOLO .pt → .onnx helper
├── run.py              # Application entry point
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 🛠️ Development

Development auto-reload:

```bash
RELOAD=1 python run.py
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{
  "status": "ok",
  "models_missing": []
}
```

---

## 🎯 Project Goal

Đây không phải một demo dịch manga tự động.

Mục tiêu là một **editor tool thực dụng**:

> **Automate the boring parts. Let the editor make the final call.**

Tự động hóa ingestion, slicing, detection, inpainting và OCR; cung cấp manual repair khi automation không hoàn hảo; sau đó để editor tự dịch và typeset với output sạch, ổn định và có thể kiểm soát.

---

## 📜 License

MIT License.
