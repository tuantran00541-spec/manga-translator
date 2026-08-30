# Sơ đồ luồng & Kiến trúc — Manga Translator

> Các sơ đồ dùng [Mermaid](https://mermaid.js.org/) — xem trực tiếp trên GitHub hoặc dán vào editor có hỗ trợ Mermaid preview (VS Code + extension "Markdown Preview Mermaid Support", Typora, ...).

## 1. Tổng quan kiến trúc

```mermaid
flowchart TB
    subgraph Client["Trình duyệt (SPA vanilla JS)"]
        UI["templates/index.html<br/>static/js/*: toast → api → upload → preview → review → editor → main"]
    end

    subgraph Server["FastAPI (app/main.py)"]
        MW["RequestSizeLimitMiddleware"]
        RTR["Routers<br/>chapters · editor · render · image"]
    end

    subgraph Core["Core business"]
        DEP["dependencies.py — singleton"]
        PL["pipeline.py — ChapterPipeline (orchestrator)"]
        OCR["ocr/multi_lang_ocr.py — MultiLangOCR"]
        MAN["manifest_utils.py — manifest.json + file locks"]
        SEC["security.py — SSRF / path traversal / size limits"]
    end

    subgraph AI["AI modules (CPU / ONNX Runtime)"]
        DET["detector/combined_detector.py"]
        YOLO["detector/bubble_detector.py — YoloDetector"]
        MB["detector/mask_builder.py"]
        LAMA["inpaint/lama_inpainter.py — Inpainter"]
    end

    subgraph DL["Downloader"]
        REG["downloader/registry.py"]
        BASE["downloader/base.py — BaseAdapter"]
        JS["downloader/generic_js.py — Playwright"]
        SL["downloader/slicer.py"]
    end

    subgraph RD["Render / Typeset"]
        TR["render/text_renderer.py"]
    end

    subgraph Data["Dữ liệu trên đĩa"]
        RAW["data/raw/{chapter}/sliced/"]
        PROC["data/processed/{chapter}/<br/>manifest.json · clean_*.png · manual_mask_*"]
        OUT["data/output/{chapter}/page_XXX.png"]
        MOD["models/ — 3 file ONNX"]
    end

    UI -->|HTTP /api/*| MW --> RTR
    RTR --> DEP
    DEP --> PL
    DEP --> OCR
    PL --> DET --> YOLO
    DET --> MB
    PL --> LAMA --> MB
    PL --> REG --> BASE
    REG --> JS
    PL --> SL
    RTR --> TR
    PL <--> MAN
    MAN --> PROC
    RTR --> PROC
    REG --> RAW
    SL --> RAW
    PL --> PROC
    TR --> OUT
    RTR --> OUT
    YOLO --> MOD
    LAMA --> MOD
```

## 2. Khởi động (startup)

```mermaid
flowchart TD
    A["run.py"] --> B["config.ensure_directories() — tạo data/, models/, logs/"]
    A --> C["config.check_models() — cảnh báo nếu thiếu 3 model ONNX"]
    A --> D["uvicorn.run('app.main:app')"]
    D --> E["main.py: lifespan"]
    E --> F["check_models() → log trạng thái"]
    E --> G["mount /static → app/static/"]
    E --> H["include_router × 4: chapters · editor · render · image"]
    E --> I["thêm RequestSizeLimitMiddleware"]
    E --> J["GET /health → {status, models_missing}"]
    E --> K["GET / → app/templates/index.html"]
```

## 3. Tạo chapter từ URL (Import / Crawl)

```mermaid
sequenceDiagram
    participant UI as Frontend — api.js loadChapter()
    participant RT as POST /api/chapter (chapters.py)
    participant SEC as security.validate_url
    participant PL as pipeline.download_chapter()
    participant REG as registry.download_chapter()
    participant ST as GenericStaticAdapter (bs4)
    participant JS as GenericJsAdapter (Playwright)
    participant SL as slicer.slice_image()
    participant MAN as manifest_utils (locks)

    UI->>RT: {url, workers}
    RT->>SEC: chặn SSRF: scheme http(s), private/loopback/link-local IP
    RT->>PL: download_chapter(url, chapter_id = os.urandom(4).hex())
    PL->>REG: tải ảnh → data/raw/{id}/
    alt static parse được ảnh
        REG->>ST: đọc <img> (data-src/src) → tải từng ảnh
    else static thất bại / rỗng
        REG->>JS: Playwright: mở trang, cuộn, lọc ảnh ≥ 400px → tải
    end
    PL->>SL: slice_image từng ảnh → data/raw/{id}/sliced/ (chia webtoon dài)
    PL->>MAN: get_manifest_lock → save_manifest_raw(manifest khởi tạo)
    RT-->>UI: manifest (urlify_manifest: đường ảnh → /api/image/...)
    UI->>UI: renderPreview()
```

## 4. Tạo chapter từ upload (ảnh / ZIP / CBZ)

```mermaid
flowchart TD
    A["POST /api/chapter/upload — multipart"] --> B["RequestSizeLimitMiddleware: tổng ≤ 500MB"]
    B --> C["≤ 300 file, mỗi file ≤ 100MB"]
    C --> D{"File có phải ZIP/CBZ?"}
    D -->|"có"| E["Giải nén, tự nhiên-sort tên, bỏ __MACOSX/, file ẩn"]
    D -->|"không"| F["Giữ nguyên"]
    E --> G["validate_upload_image: PIL verify + format PNG/JPEG/WEBP/BMP"]
    F --> G
    G --> H["Ghi ảnh ra data/raw/{id}/000.ext ... (rename theo thứ tự)"]
    H --> I["_build_chapter_from_raw_paths: slice + tạo manifest"]
    I --> J["urlify_manifest → trả về cho UI"]
```

## 5. Xử lý trang — process_pages (luồng chính)

```mermaid
sequenceDiagram
    participant UI as Frontend — processSelectedPages()
    participant RT as POST /api/process_pages
    participant PL as pipeline.process_pages()
    participant LK as manifest_utils (lock + snapshot)
    participant PP as _process_page() (worker thread ×2–8)
    participant DET as CombinedTextDetector
    participant LAMA as Inpainter (LaMa)
    participant MAN as save_manifest_raw

    UI->>RT: {chapter_id, page_indices, workers}
    RT->>RT: validate_chapter_id + kiểm tra index hợp lệ
    RT->>PL: process_pages()
    PL->>LK: get_manifest_lock → load manifest
    PL->>LK: capture_processing_state() — snapshot từng page (original, excluded, boxes, manual_mask)
    PL->>PP: ThreadPoolExecutor(max 2–8)
    PP->>PP: read_image (chống ảnh quá lớn)
    PP->>DET: detect(image) → list[BubbleBox]
    PP->>PP: lọc bỏ box nằm trong excluded_regions
    PP->>LAMA: inpaint(image, boxes)
    PP->>LAMA: nếu có manual_mask → inpaint_mask() bổ sung
    PP-->>PL: {tmp_clean, boxes (kèm mask base64), manual_mask}
    PL->>LK: is_processing_state_current() — chống ghi đè stale
    alt state không đổi
        PL->>MAN: os.replace(tmp → clean_{name}.png), merge boxes mới + manual boxes
        PL->>MAN: invalidate_page_render (rendered = false) → save manifest
        PL->>PL: _sync_output_dir → copy ảnh sạch ra data/output
    else state đã đổi trong lúc xử lý
        PL-->>PL: hủy output cũ (discard), giữ nguyên manifest
    end
    RT-->>UI: manifest mới → renderReview()
```

## 6. Detection chi tiết

```mermaid
flowchart TD
    A["Ảnh page (BGR)"] --> B["CombinedTextDetector.detect()"]
    B --> C["YoloDetector(bubble_yolo.onnx).detect()"]
    B --> D["YoloDetector(text_segmenter.onnx).detect()"]
    C --> E{"Text box có nằm trong bubble?<br/>_is_inside (tâm hoặc ≥50% diện tích)"}
    D --> E
    E -->|"có"| F["_split_cluster_by_lines: gom theo dòng nếu >3 box hoặc cao"]
    F --> G["_merge_masks: hợp mask các text box → 1 BubbleBox"]
    E -->|"không"| H["Bubble không chứa text → crop mask lề (3%)"]
    D --> I["Free text còn lại → _cluster_free_text_boxes<br/>(gom box gần nhau + giới hạn diện tích 35%)"]
    I --> J["_refine_and_split_tall_boxes: tách box cao >45px theo dòng"]
    G --> J
    H --> J
    J --> K["_apply_final_nms (IoU 0.35)"]
    K --> L["list[BubbleBox] — kèm mask từng box"]
    C -->|"đường khác"| M["YoloDetector nội bộ: preprocess 1024×1024 → NMS →<br/>decode mask từ prototype (YOLO seg)"]
    D --> M
```

## 7. Inpaint chi tiết

```mermaid
flowchart TD
    A["Inpainter.inpaint(image, boxes)"] --> B["_cluster_boxes: gom box gần nhau<br/>(padding 35px, giới hạn cluster ≤600px)"]
    B --> C["_compute_crop_region: padding 35, vuông hóa crop"]
    C --> D["mask_builder.build_mask: mask học được từng box,<br/>rect fallback nếu thiếu, adaptive_dilate (kernel 7–9)"]
    D --> E["_smart_paint_region"]
    E --> F{"Crop đơn sắc?"}
    F -->|"trắng >70%"| G["fill màu median của vùng trắng"]
    F -->|"đen >70%"| H["fill màu median của vùng đen"]
    F -->|"std < 12"| I["fill màu median tổng"]
    F -->|"phức tạp"| J["_lama_fill → ONNX LaMa 512×512<br/>(tiled + feather cho vùng tô tay lớn)"]
    G --> K["Ảnh sạch (clean)"]
    H --> K
    I --> K
    J --> K
```

## 8. OCR

```mermaid
flowchart TD
    A["POST /api/ocr_box hoặc chapter OCR job"] --> B["OCRService: snapshot source/file/box geometry<br/>+ kiểm machine cache"]
    B -->|"cache hợp lệ"| C["trả ocr_text + confidence/model/orientation/<br/>region_count/quality metadata"]
    B -->|"cache miss"| D["ocr_crop_from_box: tight theo segmentation mask + 12px<br/>fallback bbox + 20px"]
    D --> E["MultiLangOCR.read_detailed(image, lang)"]
    E --> F{"lang?"}
    F -->|"ja"| G["MangaOCR primary<br/>lazy load + lock"]
    F -->|"en / ch / zh"| H["PaddleOCR 3.x<br/>PP-OCRv6 small/medium det + rec"]
    F -->|"ko / korean"| I["PP-OCRv6 detector +<br/>korean_PP-OCRv5_mobile_rec"]
    H --> J["reconstruct_reading_order"]
    I --> J
    J --> K["centered target selection mặc định<br/>rollback: MANGA_OCR_TARGET_SELECTION=all"]
    G --> L["classify_ocr_quality"]
    K --> L
    L --> M["OCRService commit dưới manifest lock:<br/>kiểm lại source/file/geometry stale"]
    M --> N["stamp machine cache + metadata<br/>sync auto text object + translation ownership"]
    N --> O["invalidate render → save manifest → sync output"]
```

Production routing và knobs:

- Japanese luôn đi MangaOCR primary; không dùng confidence để fallback sang MangaOCR.
- English/Chinese dùng PP-OCRv6; Korean dùng recognizer Korean PP-OCRv5 mobile với detector PP-OCRv6.
- `MANGA_PPOCRV6_TIER=small|medium`, mặc định `small`.
- `MANGA_PPOCRV6_TEXTLINE_ORIENTATION=1` bật model orientation bổ sung; mặc định tắt để giảm cold-start/CPU/memory.
- `MANGA_OCR_TARGET_SELECTION=centered|all`, mặc định `centered`; `all` là rollback nhanh nếu crop đặc biệt cần toàn bộ region.
- OCR `review` vẫn được phép dịch; chỉ kết quả machine OCR `reject` chưa bị người dùng sửa mới bị chặn translation.
- Paddle detector polygon chỉ phục vụ localization/reading order OCR, không trở thành erase/inpaint mask.

## 9. Render / Typeset

```mermaid
sequenceDiagram
    participant UI as Frontend — renderTranslations()
    participant RT as POST /api/render (render.py)
    participant LK as manifest lock
    participant TR as text_renderer.render_text_in_box()
    participant OUT as data/output/{id}/page_{n}.png

    UI->>RT: translations + styles (per text_object hoặc per box + drafts)
    RT->>LK: snapshot: boxes, text_objects, clean, drafts
    RT->>RT: mở ảnh base (clean → original nếu skip)
    alt page có text_objects
        RT->>TR: _render_text_objects — shape (rect/ellipse), align, style từng object
    else legacy (chỉ có boxes)
        RT->>TR: _render_boxes_legacy — translations + drafts theo box index
    end
    TR->>TR: auto font-size (binary search 6–48px), màu tự động theo nền,
    TR->>TR: stroke, bg, corner radius, wrap text, align
    RT->>OUT: lưu tmp → kiểm tra state → os.replace → rendered = true
    RT->>RT: ghi lại drafts / translations vào manifest
    RT-->>UI: {output: "/api/image/{id}/{page}/rendered"}
```

## 10. Sửa tay (Manual repair) — brush / magic wand

```mermaid
sequenceDiagram
    participant UI as review.js — setupBrush / floodFillSelect
    participant RT as POST /api/repaint_mask
    participant PL as pipeline.repaint_mask()
    participant LK as page_lock + manifest_lock
    participant LAMA as Inpainter.inpaint_mask()

    UI->>UI: người dùng tô vùng cần xóa trên canvas
    UI->>RT: form-data {chapter_id, page_index, mask PNG}
    RT->>RT: decode mask (kênh alpha / gray), resize đúng kích thước page
    RT->>PL: repaint_mask()
    PL->>LK: page lock (chống 2 người sửa cùng trang) + manifest lock
    PL->>PL: gộp mask mới + manual_mask cũ (accumulate, resize nếu lệch)
    PL->>LAMA: connectedComponents → dilate từng cụm (kernel 9–15)
    PL->>LAMA: crop quanh cụm (padding 72) → LaMa → blend feather
    PL->>LK: lưu manual_mask_{name}.png, cập nhật clean, invalidate render
    RT-->>UI: manifest mới (ảnh clean mới)
```

## 11. Frontend — luồng màn hình

```mermaid
flowchart LR
    A["index.html — nạp CSS + JS theo thứ tự:<br/>toast → api → upload → preview → review<br/>→ review-workspace → editor → editor-box-transform → main"] --> B["main.js bootstrap:<br/>gán nút Tải chapter, initUpload, loadRecentChapters, loadFonts"]
    B --> C["api.js — lớp gọi /api/*"]
    C --> D["preview.js — renderPreview():<br/>xem các trang sau slice, chọn trang để xử lý"]
    D --> E["review.js — renderReview():<br/>kiểm tra/điều chỉnh box + brush/magic wand"]
    E --> F["editor.js — renderEditor():<br/>dịch + chỉnh font/size/màu/stroke/align"]
    F --> G["editor-box-transform.js —<br/>kéo/thả, resize text object"]
    G -->|"flush → /api/text_object/*"| C
    C -->|"/api/process_pages"| D
    C -->|"/api/repaint_mask / reset_manual_mask"| E
    C -->|"/api/render"| F
    F --> H["preview.js — showRenderResult():<br/>hiển thị ảnh đã chèn chữ"]
    C -->|"GET /api/chapters — hồi phục chapter dở"| B
```

## 12. Concurrency & an toàn dữ liệu

```mermaid
flowchart TD
    A["Nhiều request cùng chapter/page"] --> B["get_manifest_lock()<br/>filelock: manifest.lock (timeout 30s)"]
    A --> C["get_page_lock()<br/>filelock: page_XXX.lock (timeout 60s)"]
    B --> D["save_manifest_raw: ghi tmp + os.replace (atomic)"]
    C --> D
    A --> E["capture_processing_state → snapshot canonical inputs<br/>(original, skipped, excluded_regions, boxes, manual_mask)"]
    E --> F["is_processing_state_current: sau khi xử lý xong<br/>so lại snapshot → discard nếu stale"]
```

## Ghi chú phát hiện khi rà soát

- `app/static/js/box-item.js`, `editor-properties.js`, `editor-workspace.js` **không được nạp từ `index.html`** — có vẻ là module legacy; hai file sau còn định nghĩa lại `window.renderEditor` (trùng tên với `editor.js` đang dùng).
- `app/downloader/playwright_worker.py` là script subprocess không được `registry.py` gọi — bản dự định thay thế `generic_js.py`.
- Luồng ảnh: `data/raw/{id}` (gốc + sliced) → `data/processed/{id}` (clean, manifest) → `data/output/{id}` (render cuối). Tất cả đều git-ignored.