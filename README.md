# Manga & Webtoon Translator Studio

<p align="center">
  <strong>Local-First AI-Assisted Manga, Manhwa, and Webtoon Typesetting & Translation Studio</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python&logoColor=white" alt="Python Version" />
  <img src="https://img.shields.io/badge/FastAPI-Framework-009688?logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/ONNX%20Runtime-CPU%20Optimized-005CED?logo=onnx&logoColor=white" alt="ONNX Runtime" />
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white" alt="Docker" />
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License" />
</p>

---

<p align="center">
  <a href="#english"><strong>English</strong></a> &bull;
  <a href="#tiếng-việt"><strong>Tiếng Việt</strong></a>
</p>

---

<a name="english"></a>
# English

## Overview

**Manga & Webtoon Translator Studio** is a high-performance, local-first web application designed for manga, manhwa, and webtoon translation teams and solo scanlators. It automates repetitive, labor-intensive tasks—image ingestion, long-strip webtoon slicing, speech bubble detection, text removal (inpainting), and optical character recognition (OCR)—while giving human editors complete creative control over translation, typography, and styling.

> [!NOTE]
> **No Automated Machine Translation**: This tool does not use black-box machine translation or hallucinating LLMs. OCR extracts original source text to assist the editor, who retains full ownership over translation accuracy, tone, and typeset styling.

---

## Key Features

- 📥 **Flexible Ingestion**: Ingest chapters directly from online URLs via Playwright/HTTP scraping, or upload local image folders, ZIP, and CBZ archives.
- ✂️ **Intelligent Webtoon Slicing**: Splits continuous vertical manhwa/webtoon strips into manageable pages using smart seam-carving algorithms that avoid cutting through dialogue bubbles.
- 🎯 **Deep Learning Detection**: Accurately localizes speech bubbles and stylized text regions using YOLOv8 ONNX models (`bubble_yolo.onnx` and `text_segmenter.onnx`).
- 🧹 **CPU-Optimized Inpainting**: Reconstructs pristine background artwork using LaMa (Large Mask Inpainting) on ONNX Runtime, tuned for multi-threaded CPU execution without requiring high-end GPUs.
- 🔤 **Multilingual OCR**: Fast, high-accuracy text extraction for Japanese (vertical & horizontal), Chinese, Korean, and English.
- 🖌️ **Single-Card Review & Manual Repair**: Intuitive canvas workspace with interactive brush tools, inpaint repair, magic-wand exclusion zones, and live mask rendering.
- ✍️ **Professional Typography Studio**: Full vector-like canvas editing for text objects:
  - Drag, resize, and position bounding boxes directly on the canvas.
  - Multi-font selection with instant preview.
  - Auto-fit font sizing or custom point size scaling.
  - Stroke width, stroke color, background highlight pills, corner radiuses, and multi-axis text alignment (horizontal & vertical).
- ⚡ **Concurrency-Safe Architecture**: Atomic manifest persistence with cross-process file locks, debounced async background synchronization, request generation tracking, and local state diff preservation.
- 🔒 **Enterprise-Grade Security Hardening**: SSRF defense against loopback/internal metadata probing, strict path traversal validation, input dimension clamping, and sanitization.

---

## Workflow Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│ 1. INGESTION & PREPARATION                                  │
│    URL Scrape (Playwright) / Upload (Images, ZIP, CBZ)      │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. PREVIEW & WEBTOON SLICING                                │
│    Smart Seam Slicing ──▶ Exclusion Masking ──▶ Skip Pages  │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. AI PIPELINE EXECUTION                                    │
│    YOLO Detection ──▶ LaMa Inpainting ──▶ Multilingual OCR  │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. REVIEW & MANUAL INPAINT REPAIR                           │
│    Single-Card Carousel ──▶ Brush Erase ──▶ Inpaint Re-run  │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. TYPESETTING & EDITING STUDIO                             │
│    Interactive Bounding Boxes ──▶ Typography ──▶ Live Sync  │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. FINAL RENDERING & EXPORT                                 │
│    Pillow Text Layout Engine ──▶ Hi-Res Rendered PNGs       │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Option A: Docker (Recommended)

1. Clone the repository:
   ```bash
   git clone https://github.com/tuantran00541-spec/manga-translator.git
   cd manga-translator
   ```

2. Place required ONNX models into the `models/` directory (see [Models Setup](#models-setup)).

3. Start the containerized service:
   ```bash
   docker compose up --build
   ```

4. Navigate to **`http://127.0.0.1:8000`** in your browser.

---

### Option B: Local Python Environment

**Prerequisites**: Python 3.10–3.12, RAM &ge; 8 GB recommended.

1. Clone the repository and set up a virtual environment:
   ```bash
   git clone https://github.com/tuantran00541-spec/manga-translator.git
   cd manga-translator

   python -m venv venv
   # Linux / macOS:
   source venv/bin/activate
   # Windows PowerShell:
   .\venv\Scripts\Activate.ps1
   ```

2. Install dependencies and browser binaries:
   ```bash
   pip install -r requirements.txt
   playwright install chromium
   ```

3. Place ONNX models in `models/` (see [Models Setup](#models-setup)).

4. Launch the application:
   ```bash
   python run.py
   ```

5. Open **`http://127.0.0.1:8000`**.

---

## Models Setup

Download and place the following 3 ONNX models into the `models/` folder:

| Model File | Function | Recommended Source |
| :--- | :--- | :--- |
| `bubble_yolo.onnx` | Speech bubble detection | [ogkalu/comic-speech-bubble-detector-yolov8m](https://huggingface.co/ogkalu/comic-speech-bubble-detector-yolov8m) |
| `text_segmenter.onnx` | Fine-grained text localization | [ogkalu/comic-text-segmenter-yolov8m](https://huggingface.co/ogkalu/comic-text-segmenter-yolov8m) |
| `lama.onnx` | Background reconstruction & text removal | [Carve/LaMa-ONNX](https://huggingface.co/Carve/LaMa-ONNX) (`lama_fp32.onnx`) |

> [!TIP]
> **Converting YOLO PyTorch Weights (`.pt`) to ONNX**:
> ```bash
> pip install ultralytics
> python convert_model.py path/to/weights.pt
> ```
> Rename generated files to `bubble_yolo.onnx` or `text_segmenter.onnx` accordingly.

---

## Keyboard Shortcuts

| Key / Shortcut | Context | Action |
| :--- | :--- | :--- |
| `PageUp` / `PageDown` | Global (Editor, Preview, Review) | Navigate to Previous / Next page without browser scrolling |
| `ArrowLeft` / `ArrowRight` | Editor (No box selected) | Navigate to Previous / Next page |
| `ArrowUp` / `Down` / `Left` / `Right` | Editor (Box selected) | Nudge active bounding box by 1px (`Shift` + Arrow for 5px) |
| `Escape` | Editor | Deselect currently active text object |
| `Delete` / `Backspace` | Editor (Box selected) | Delete selected text object |
| `Double Click` | Editor (Canvas Overlay) | Focus translation textarea in sidebar |
| `Enter` | Page Jump input | Validate integer and navigate directly to target page |

---

## Configuration & Environment Variables

| Variable | Default | Description |
| :--- | :---:| :--- |
| `HOST` | `127.0.0.1` | Network binding address (`0.0.0.0` for container/network exposure) |
| `PORT` | `8000` | HTTP listening port |
| `WORKERS` | `1` | Uvicorn worker processes (keep `1` to avoid CPU thread oversubscription) |
| `ENABLE_TTA` | `0` | Enable Test-Time Augmentation for bubble detector (higher accuracy, slower inference) |
| `MANGA_ORT_INTRA_OP_THREADS` | auto | Override ONNX Runtime intra-op thread pool (recommended: `4`–`8` on multi-core CPUs) |
| `RELOAD` | `0` | Enable auto-reload for local development |

---

## REST API Reference

### Chapters & Pipeline
- `GET /health` — Check system readiness and model loading status.
- `GET /api/chapters` — Retrieve recent active chapters with progress checkpoints.
- `GET /api/chapter/{chapter_id}` — Retrieve full chapter manifest JSON.
- `POST /api/chapter` — Ingest and slice chapter from URL.
- `POST /api/chapter/upload` — Ingest chapter from uploaded image archive (ZIP/CBZ) or files.
- `POST /api/workflow_checkpoint` — Save current workflow stage (`preview`, `review`, `editor`) and page index.
- `POST /api/process_pages` — Execute batch AI pipeline (detection, inpainting, OCR).
- `POST /api/skip_pages` — Toggle skip status for specific pages.
- `POST /api/save_excluded_regions` — Save non-translatable rectangular zones.

### Review & Manual Inpaint
- `POST /api/manual_mask` — Apply brush stroke inpaint repair on canvas.
- `POST /api/reset_manual_mask` — Reset manual inpaint modifications to clean image.

### Typesetting & Text Objects
- `POST /api/text_object/create` — Create text object (`rectangle` or `ellipse`).
- `POST /api/text_object/update` — Update text object text, geometry, or typography styles.
- `POST /api/text_object/delete` — Delete text object.
- `POST /api/text_object/ocr` — Re-run targeted OCR for a specific text object box.
- `POST /api/render` — Render translated typography onto cleaned page image.
- `GET /api/fonts` — List available bundled and system fonts.

---

## Project Structure

```text
manga-translator/
├── app/
│   ├── detector/          # YOLO Bubble and Text segmentation & mask generators
│   ├── downloader/        # Scrapers, archive extractors, and seam slicer
│   ├── inpaint/           # LaMa ONNX CPU inpainting engine
│   ├── ocr/               # Multi-engine OCR adapters (Manga-OCR, PaddleOCR, RapidOCR)
│   ├── render/            # Text layout, font metrics, and rendering engine
│   ├── routers/           # FastAPI modular route handlers
│   ├── static/            # Frontend assets (CSS stylesheets, JS modules)
│   │   ├── css/           # Modern dark-mode styling
│   │   └── js/            # Canvas manipulators, state machines, API bridges
│   ├── templates/         # Jinja2 HTML layout templates
│   ├── config.py          # Centralized configuration & environment loader
│   ├── dependencies.py    # Shared singleton pipeline lifecycles
│   ├── logging_config.py  # Structured JSON/text logging setup
│   ├── manifest_utils.py  # Concurrency-safe atomic manifest IO with file locks
│   ├── pipeline.py        # Core pipeline coordinator
│   └── security.py        # SSRF filter, path traversal guards, upload limits
├── data/                  # Runtime storage (raw, processed, output)
├── models/                # Local ONNX weights
├── logs/                  # Application logs
├── convert_model.py       # Helper script to export PyTorch weights to ONNX
├── run.py                 # Application launcher
├── Dockerfile             # Production container definition
├── docker-compose.yml     # Container orchestration
└── requirements.txt       # Python dependencies
```

---
---

<a name="tiếng-việt"></a>
# Tiếng Việt

## Tổng quan

**Manga & Webtoon Translator Studio** là ứng dụng web cục bộ (Local-First), chuyên dụng cho các nhóm dịch và cá nhân làm scanlation manga, manhwa, webtoon. Hệ thống tự động hóa toàn bộ các công đoạn xử lý hình ảnh phức tạp—tải chapter, cắt lát webtoon dài, nhận diện bong bóng thoại/vùng chữ, tẩy chữ gốc (Inpaint) và nhận diện chữ (OCR)—đồng thời cung cấp bộ công cụ biên tập trực quan giúp editor toàn quyền kiểm soát bản dịch và kiểu chữ (typeset).

> [!NOTE]
> **Không dịch tự động bằng AI**: Ứng dụng không sử dụng dịch máy hay LLM tự động dịch. OCR chỉ trích xuất chữ gốc hỗ trợ editor; nội dung bản dịch và kiểu dáng chữ do người dùng quyết định hoàn toàn.

---

## Tính năng nổi bật

- 📥 **Nhập liệu đa dạng**: Tải tự động chapter từ đường dẫn URL (qua Playwright/HTTP) hoặc tải lên file ảnh rời, file nén ZIP, CBZ.
- ✂️ **Cắt lát Webtoon thông minh**: Tự động chia các trang truyện dài thành từng lát an toàn, hạn chế tối đa việc cắt ngang bong bóng thoại hay khung hình.
- 🎯 **Nhận diện bằng Deep Learning**: Định vị chính xác bong bóng thoại và vùng chữ cách điệu bằng mô hình YOLOv8 ONNX (`bubble_yolo.onnx` và `text_segmenter.onnx`).
- 🧹 **Tẩy chữ LaMa tối ưu CPU**: Tái tạo phông nền nguyên bản bằng mô hình LaMa Inpainting trên ONNX Runtime đa luồng, vận hành mượt mà trên CPU thông thường mà không cần GPU đắt tiền.
- 🔤 **OCR đa ngôn ngữ chất lượng cao**: Đọc nhanh và chuẩn xác chữ tiếng Nhật (dọc & ngang), tiếng Trung, tiếng Hàn và tiếng Anh.
- 🖌️ **Chế độ kiểm tra & sửa ảnh thủ công (Review)**: Giao diện sửa ảnh đơn lát tập trung, hỗ trợ cọ vẽ tẩy tay (brush), khoanh vùng cấm dịch và xem trước mask trực tiếp.
- ✍️ **Studio Typeset chuyên nghiệp**: Thao tác trực tiếp trên canvas tương tự phần mềm đồ họa vector:
  - Kéo thả, thay đổi kích thước và vị trí ô chữ trực quan trên ảnh.
  - Chọn phông chữ với xem trước tức thì.
  - Tự động co giãn cỡ chữ vừa khung (Auto-fit) hoặc tùy chỉnh kích thước theo ý muốn.
  - Viền chữ (stroke), màu chữ, đổ nền nổi (pill background), bo góc nền và căn lề đa chiều (ngang & dọc).
- ⚡ **Kiến trúc an toàn dữ liệu & đa luồng**: Lưu trữ manifest nguyên tử (atomic write) kèm khóa file (file lock), đồng bộ dữ liệu ngầm chống nghẽn, theo dõi thế hệ request tránh ghi đè dữ liệu cũ.
- 🔒 **Bảo mật chuẩn doanh nghiệp**: Chống tấn công SSRF khi cào dữ liệu, kiểm soát nghiêm ngặt đường dẫn (Path Traversal), giới hạn dung lượng tải lên và kích thước ảnh.

---

## Quy trình làm việc (Workflow)

```text
┌─────────────────────────────────────────────────────────────┐
│ 1. NHẬP LIỆU & CHUẨN BỊ                                     │
│    Tải từ URL (Playwright) / Upload (Ảnh, ZIP, CBZ)         │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. XEM TRƯỚC & CẮT LÁT WEBTOON                              │
│    Cắt lát thông minh ──▶ Đánh dấu vùng cấm ──▶ Bỏ qua lát │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. XỬ LÝ AI TỰ ĐỘNG                                         │
│    Nhận diện YOLO ──▶ Tẩy chữ LaMa Inpaint ──▶ OCR Đa ngữ   │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. KIỂM TRA & TẨY SỬA THỦ CÔNG (REVIEW)                     │
│    Giao diện đơn lát ──▶ Cọ vẽ sửa lỗi ──▶ Tẩy lại LaMa    │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. BIÊN TẬP DỊCH & TYPESET                                  │
│    Ô chữ tương tác ──▶ Kiểu chữ & Màu sắc ──▶ Tự động lưu   │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. XUẤT ẢNH HOÀN THIỆN                                      │
│    Bộ dựng chữ Pillow ──▶ Xuất file ảnh PNG độ nét cao      │
└─────────────────────────────────────────────────────────────┘
```

---

## Hướng dẫn cài đặt

### Cách 1: Sử dụng Docker (Khuyên dùng)

1. Clone mã nguồn về máy:
   ```bash
   git clone https://github.com/tuantran00541-spec/manga-translator.git
   cd manga-translator
   ```

2. Tải và đặt 3 file mô hình ONNX vào thư mục `models/` (xem [Cài đặt Models](#cài-đặt-models)).

3. Khởi chạy ứng dụng:
   ```bash
   docker compose up --build
   ```

4. Truy cập trình duyệt tại **`http://127.0.0.1:8000`**.

---

### Cách 2: Chạy trực tiếp với Python

**Yêu cầu môi trường**: Python 3.10–3.12, RAM &ge; 8 GB.

1. Clone mã nguồn và tạo môi trường ảo:
   ```bash
   git clone https://github.com/tuantran00541-spec/manga-translator.git
   cd manga-translator

   python -m venv venv
   # Linux / macOS:
   source venv/bin/activate
   # Windows PowerShell:
   .\venv\Scripts\Activate.ps1
   ```

2. Cài đặt các thư viện phụ thuộc và trình duyệt Playwright:
   ```bash
   pip install -r requirements.txt
   playwright install chromium
   ```

3. Đặt các file mô hình ONNX vào thư mục `models/` (xem mục kế tiếp).

4. Khởi chạy máy chủ:
   ```bash
   python run.py
   ```

5. Mở trình duyệt tại **`http://127.0.0.1:8000`**.

---

## Cài đặt Models

Đặt 3 file mô hình ONNX sau vào thư mục `models/`:

| Tên file | Chức năng | Nguồn khuyến nghị |
| :--- | :--- | :--- |
| `bubble_yolo.onnx` | Nhận diện bong bóng thoại | [ogkalu/comic-speech-bubble-detector-yolov8m](https://huggingface.co/ogkalu/comic-speech-bubble-detector-yolov8m) |
| `text_segmenter.onnx` | Phân đoạn và định vị chữ | [ogkalu/comic-text-segmenter-yolov8m](https://huggingface.co/ogkalu/comic-text-segmenter-yolov8m) |
| `lama.onnx` | Tái tạo phông nền, tẩy chữ gốc | [Carve/LaMa-ONNX](https://huggingface.co/Carve/LaMa-ONNX) (`lama_fp32.onnx`) |

> [!TIP]
> **Chuyển đổi trọng số YOLO PyTorch (`.pt`) sang ONNX**:
> ```bash
> pip install ultralytics
> python convert_model.py path/to/model.pt
> ```
> Đổi tên file đầu ra thành `bubble_yolo.onnx` hoặc `text_segmenter.onnx` tương ứng.

---

## Bảng phím tắt điều khiển

| Phím tắt | Ngữ cảnh | Thao tác |
| :--- | :--- | :--- |
| `PageUp` / `PageDown` | Toàn hệ thống (Biên tập, Xem trước, Sửa ảnh) | Chuyển sang Trang trước / Trang sau (ngăn cuộn trang mặc định) |
| `ArrowLeft` / `ArrowRight` | Biên tập (Khi không chọn ô chữ) | Chuyển sang Trang trước / Trang sau |
| Phím mũi tên (`↑` `↓` `←` `→`) | Biên tập (Khi đang chọn ô chữ) | Dịch chuyển ô chữ 1px (giữ `Shift` để dịch chuyển 5px) |
| `Escape` | Biên tập | Hủy chọn ô chữ đang kích hoạt |
| `Delete` / `Backspace` | Biên tập (Khi đang chọn ô chữ) | Xóa ô chữ được chọn |
| `Double Click` | Biên tập (Lớp phủ trên ảnh) | Nhảy chuột vào khung nhập bản dịch ở thanh bên |
| `Enter` | Ô nhập nhảy số trang | Xác thực số nguyên và chuyển ngay tới trang chỉ định |

---

## Cấu hình & Biến môi trường

| Biến môi trường | Mặc định | Ý nghĩa |
| :--- | :---:| :--- |
| `HOST` | `127.0.0.1` | Địa chỉ IP máy chủ lắng nghe (`0.0.0.0` để mở mạng ngoài/Docker) |
| `PORT` | `8000` | Cổng HTTP |
| `WORKERS` | `1` | Số tiến trình Uvicorn (nên giữ `1` để tối ưu tải đa luồng CPU cho AI) |
| `ENABLE_TTA` | `0` | Bật Test-Time Augmentation cho phát hiện bong bóng (chính xác hơn, chậm hơn) |
| `MANGA_ORT_INTRA_OP_THREADS` | auto | Số luồng xử lý CPU của ONNX Runtime (khuyến nghị: `4`–`8` trên CPU đa nhân) |
| `RELOAD` | `0` | Tự động tải lại mã nguồn khi lập trình (Development) |

---

## Danh mục API RESTful

### Chapter & Tiến trình Pipeline
- `GET /health` — Kiểm tra trạng thái máy chủ và tình trạng tải models.
- `GET /api/chapters` — Lấy danh sách các chapter đang xử lý kèm mốc tiến độ.
- `GET /api/chapter/{chapter_id}` — Lấy dữ liệu chi tiết manifest của chapter.
- `POST /api/chapter` — Tải và cắt lát chapter từ URL.
- `POST /api/chapter/upload` — Tải lên chapter từ file ảnh hoặc file nén ZIP/CBZ.
- `POST /api/workflow_checkpoint` — Lưu mốc trạng thái làm việc (`preview`, `review`, `editor`) và số trang.
- `POST /api/process_pages` — Chạy tiến trình xử lý AI hàng loạt (phát hiện, tẩy chữ, OCR).
- `POST /api/skip_pages` — Đánh dấu bỏ qua hoặc khôi phục trang chỉ định.
- `POST /api/save_excluded_regions` — Lưu các vùng chữ nhật cấm dịch.

### Sửa ảnh thủ công (Review)
- `POST /api/manual_mask` — Tẩy xóa bằng cọ vẽ thủ công trên canvas.
- `POST /api/reset_manual_mask` — Khôi phục ảnh đã tẩy về trạng thái gốc sạch ban đầu.

### Biên tập & Typeset Ô chữ
- `POST /api/text_object/create` — Tạo ô chữ mới (`rectangle` hoặc `ellipse`).
- `POST /api/text_object/update` — Cập nhật nội dung, vị trí hoặc kiểu dáng ô chữ.
- `POST /api/text_object/delete` — Xóa ô chữ.
- `POST /api/text_object/ocr` — Chạy lại OCR riêng biệt cho một ô chữ cụ thể.
- `POST /api/render` — Dựng chữ bản dịch lên ảnh nền đã tẩy.
- `GET /api/fonts` — Danh sách các phông chữ khả dụng trong hệ thống.

---

## Cấu trúc thư mục dự án

```text
manga-translator/
├── app/
│   ├── detector/          # Mô hình nhận diện bong bóng thoại và phân đoạn chữ
│   ├── downloader/        # Cào dữ liệu webtoon, giải nén và thuật toán cắt lát
│   ├── inpaint/           # Động cơ tẩy chữ LaMa ONNX tối ưu CPU
│   ├── ocr/               # Bộ kết nối các động cơ OCR (Manga-OCR, PaddleOCR, RapidOCR)
│   ├── render/            # Động cơ định dạng văn bản và dựng chữ Pillow
│   ├── routers/           # Bộ điều phối API theo từng nhóm chức năng
│   ├── static/            # Tài nguyên giao diện Web (CSS, JS)
│   │   ├── css/           # Bảng phong cách Dark-mode hiện đại
│   │   └── js/            # Xử lý canvas, máy trạng thái lưu trữ, cầu nối API
│   ├── templates/         # Giao diện HTML Jinja2
│   ├── config.py          # Nạp và quản lý cấu hình tập trung
│   ├── dependencies.py    # Khởi tạo và chia sẻ vòng đời pipeline duy nhất
│   ├── logging_config.py  # Hệ thống ghi nhật ký (logging) có cấu trúc
│   ├── manifest_utils.py  # Đọc/ghi manifest nguyên tử kèm khóa file an toàn
│   ├── pipeline.py        # Điều phối luồng xử lý toàn cục
│   └── security.py        # Bộ lọc chống SSRF, chống Path Traversal, kiểm soát tải lên
├── data/                  # Thư mục lưu trữ dữ liệu thực thi (raw, processed, output)
├── models/                # Thư mục chứa trọng số mô hình ONNX
├── logs/                  # Nhật ký hoạt động của ứng dụng
├── convert_model.py       # Script hỗ trợ chuyển đổi model PyTorch sang ONNX
├── run.py                 # File khởi động chính của ứng dụng
├── Dockerfile             # Cấu hình đóng gói Docker
├── docker-compose.yml     # Cấu hình khởi chạy dịch vụ Docker Compose
└── requirements.txt       # Danh sách thư viện phụ thuộc Python
```

---

## Giấy phép (License)

Dự án được phân phối dưới giấy phép **MIT License**. Chi tiết xem tại file `LICENSE`.
