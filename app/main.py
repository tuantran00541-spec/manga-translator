import json
import os
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, field_validator
from starlette.middleware.base import BaseHTTPMiddleware

from contextlib import asynccontextmanager, contextmanager
from filelock import FileLock, Timeout

from app.config import (
    PROCESSED_DIR,
    OUTPUT_DIR,
    BASE_DIR,
    ensure_directories,
    check_models,
)
from app.logging_config import logger
from app.ocr.multi_lang_ocr import MultiLangOCR
from app.pipeline import ChapterPipeline
from app.render.text_renderer import list_available_fonts, render_text_in_box
from app.security import (
    MAX_RENDER_TEXT_LEN,
    MAX_RENDER_TRANSLATIONS,
    MAX_REQUEST_BYTES,
    MAX_UPLOAD_FILES,
    MAX_UPLOAD_TOTAL_BYTES,
    validate_chapter_id,
    validate_image_size,
    validate_upload_image,
    validate_url,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_directories()
    missing = check_models()
    if missing:
        logger.warning(
            "Models missing: %s — /api/process_pages will fail until models are placed in models/",
            ", ".join(missing),
        )
    else:
        logger.info("ONNX models OK")
    yield


app = FastAPI(lifespan=lifespan, title="Manga Translator", version="1.0.0")
pipeline = ChapterPipeline()
ocr = MultiLangOCR()

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "app" / "static")), name="static")


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        limit = MAX_UPLOAD_TOTAL_BYTES if request.url.path == "/api/chapter/upload" else MAX_REQUEST_BYTES
        cl = request.headers.get("content-length")
        if cl:
            try:
                size = int(cl)
            except (TypeError, ValueError):
                return JSONResponse({"detail": "Invalid Content-Length"}, status_code=400)
            if size > limit:
                return JSONResponse({"detail": "Request too large"}, status_code=413)
        return await call_next(request)


app.add_middleware(RequestSizeLimitMiddleware)

# Per-chapter locks: prevent concurrent requests (e.g. many /api/ocr_box calls
# firing at once for the same chapter) from racing on manifest.json — both the
# Windows "Access is denied" file-replace race and the silent lost-update race
# where one request's saved changes overwrite another's.



_MANIFEST_LOCK_TIMEOUT = 30


@contextmanager
def _get_manifest_lock(chapter_id: str):
    lock_path = PROCESSED_DIR / chapter_id / "manifest.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(lock_path)
    try:
        with lock.acquire(timeout=_MANIFEST_LOCK_TIMEOUT):
            yield
    except Timeout:
        raise RuntimeError(f"Manifest lock timeout for chapter {chapter_id}")


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
    logger.exception(f"Unhandled exception on {request.method} {request.url.path}")
    return JSONResponse({"detail": "Internal server error"}, status_code=500)


def _load_manifest_raw(chapter_id: str) -> dict:
    validate_chapter_id(chapter_id)
    manifest_path = PROCESSED_DIR / chapter_id / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(404, f"Chapter {chapter_id} not found")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _save_manifest_raw(chapter_id: str, manifest: dict) -> None:
    processed_dir = PROCESSED_DIR / chapter_id
    processed_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = processed_dir / f"manifest.json.{uuid.uuid4().hex}.tmp"
    final_path = processed_dir / "manifest.json"
    tmp_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(tmp_path, final_path)


def urlify_manifest(manifest: dict) -> dict:
    result = json.loads(json.dumps(manifest))
    chapter_id = result["chapter_id"]
    for i, page in enumerate(result["pages"]):
        page["original"] = f"/api/image/{chapter_id}/{i}/original"
        if page.get("clean"):
            page["clean"] = f"/api/image/{chapter_id}/{i}/clean"
        for box in page.get("boxes", []):
            box.pop("mask", None)
    return result


class ChapterRequest(BaseModel):
    url: str
    workers: int = 2


class OcrBoxRequest(BaseModel):
    chapter_id: str
    page_index: int
    box_index: int
    lang: str


class RenderRequest(BaseModel):
    chapter_id: str
    page_index: int
    translations: dict[int, str]
    colors: dict[str, str] = {}
    fonts: dict[str, str] = {}
    font_sizes: dict[str, int | str] = {}
    bolds: dict[str, bool] = {}
    stroke_widths: dict[str, int | str] = {}
    stroke_colors: dict[str, str] = {}
    bg_colors: dict[str, str] = {}
    corner_radii: dict[str, int] = {}

    @field_validator("translations")
    @classmethod
    def _check_translations(cls, v: dict[int, str]) -> dict[int, str]:
        if len(v) > MAX_RENDER_TRANSLATIONS:
            raise ValueError(f"Too many translations: {len(v)} > {MAX_RENDER_TRANSLATIONS}")
        for k, val in v.items():
            if len(val) > MAX_RENDER_TEXT_LEN:
                raise ValueError(f"Translation {k} too long: {len(val)} chars")
        return v


class SaveDraftRequest(BaseModel):
    chapter_id: str
    drafts: dict[str, dict] = {}


class ProcessPagesRequest(BaseModel):
    chapter_id: str
    page_indices: list[int]
    workers: int = 2


class SkipPagesRequest(BaseModel):
    chapter_id: str
    page_indices: list[int]
    skipped: bool


class AddBoxRequest(BaseModel):
    chapter_id: str
    page_index: int
    x1: int
    y1: int
    x2: int
    y2: int


class RemoveBoxRequest(BaseModel):
    chapter_id: str
    page_index: int
    box_index: int


@app.get("/api/fonts")
def get_available_fonts() -> list[dict[str, str]]:
    return list_available_fonts()


@app.get("/api/chapters")
def list_chapters() -> list[dict]:
    chapters = []
    if not PROCESSED_DIR.exists():
        return chapters
    for d in sorted(PROCESSED_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        manifest_path = d / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not any(p.get("clean") for p in manifest["pages"]):
            continue
        chapters.append({
            "chapter_id": manifest["chapter_id"],
            "source_url": manifest.get("source_url", ""),
            "total_pages": len(manifest["pages"]),
            "updated_at": manifest_path.stat().st_mtime,
        })
    return chapters


def _clamp_workers(n: int | None) -> int:
    try:
        v = int(n) if n is not None else 2
    except (TypeError, ValueError):
        v = 2
    return max(1, min(v, 8))


@app.post("/api/chapter")
def create_chapter(req: ChapterRequest) -> dict:
    validate_url(req.url)
    chapter_id = uuid.uuid4().hex[:8]
    workers = _clamp_workers(req.workers)
    logger.info(f"Creating chapter {chapter_id} from {req.url} (workers={workers})")
    manifest = pipeline.download_chapter(req.url, chapter_id, workers=workers)
    return urlify_manifest(manifest)


@app.post("/api/chapter/upload")
async def create_chapter_from_upload(
    files: list[UploadFile] = File(...),
    workers: int = Form(2),
) -> dict:
    if not files:
        raise HTTPException(400, "No files uploaded")
    if len(files) > MAX_UPLOAD_FILES:
        raise HTTPException(400, f"Too many files: max {MAX_UPLOAD_FILES} per upload")

    uploads = []
    total_bytes = 0
    for f in files:
        data = await f.read()
        total_bytes += len(data)
        if total_bytes > MAX_UPLOAD_TOTAL_BYTES:
            raise HTTPException(413, f"Total upload size exceeds {MAX_UPLOAD_TOTAL_BYTES // (1024*1024)}MB")
        uploads.append((f.filename or "unnamed", data))

    chapter_id = uuid.uuid4().hex[:8]
    workers_n = _clamp_workers(workers)
    logger.info(f"Creating chapter {chapter_id} from {len(uploads)} uploaded files (workers={workers_n})")
    manifest = pipeline.create_chapter_from_uploads(chapter_id, uploads, workers=workers_n)
    return urlify_manifest(manifest)


@app.post("/api/process_pages")
def process_pages(req: ProcessPagesRequest) -> dict:
    validate_chapter_id(req.chapter_id)
    workers = _clamp_workers(req.workers)
    logger.info(
        f"Chapter {req.chapter_id}: processing pages {req.page_indices} (workers={workers})"
    )
    manifest = pipeline.process_pages(req.chapter_id, req.page_indices, workers=workers)
    return urlify_manifest(manifest)


@app.post("/api/skip_pages")
def skip_pages(req: SkipPagesRequest) -> dict:
    validate_chapter_id(req.chapter_id)
    manifest = pipeline.mark_skipped(req.chapter_id, req.page_indices, req.skipped)
    return urlify_manifest(manifest)


@app.post("/api/add_box")
def add_box(req: AddBoxRequest) -> dict:
    validate_chapter_id(req.chapter_id)
    manifest = pipeline.add_manual_box(
        req.chapter_id, req.page_index, req.x1, req.y1, req.x2, req.y2
    )
    return urlify_manifest(manifest)


@app.post("/api/remove_box")
def remove_box(req: RemoveBoxRequest) -> dict:
    validate_chapter_id(req.chapter_id)
    manifest = pipeline.remove_box(req.chapter_id, req.page_index, req.box_index)
    return urlify_manifest(manifest)


@app.post("/api/repaint_mask")
async def repaint_mask(
    chapter_id: str = Form(...),
    page_index: int = Form(...),
    mask: UploadFile = File(...),
) -> dict:
    validate_chapter_id(chapter_id)
    mask_bytes = await mask.read()
    logger.info(f"Chapter {chapter_id} page {page_index}: repaint mask ({len(mask_bytes)} bytes)")
    manifest = pipeline.repaint_mask(chapter_id, page_index, mask_bytes)
    return urlify_manifest(manifest)


@app.get("/api/chapter/{chapter_id}")
def get_chapter(chapter_id: str) -> dict:
    return urlify_manifest(_load_manifest_raw(chapter_id))


@app.get("/api/image/{chapter_id}/{page_index}/{kind}")
def get_image(chapter_id: str, page_index: int, kind: str):
    validate_chapter_id(chapter_id)
    if page_index < 0:
        raise HTTPException(400, "page_index must be >= 0")
    if kind not in ("original", "clean", "rendered"):
        raise HTTPException(400, f"Invalid kind: {kind}")

    if kind == "rendered":
        path = OUTPUT_DIR / chapter_id / f"page_{page_index:03d}.png"
    else:
        manifest = _load_manifest_raw(chapter_id)
        if page_index >= len(manifest["pages"]):
            raise HTTPException(404, "Page not found")
        page = manifest["pages"][page_index]
        field = page.get(kind) if kind != "original" else page.get("original")
        if not field:
            raise HTTPException(404, f"{kind} not available for this page")
        path = Path(field)

    if not path.exists():
        raise HTTPException(404, "Image file not found")
    validate_image_size(path)
    return FileResponse(path)


@app.post("/api/ocr_box")
def ocr_box(req: OcrBoxRequest) -> dict:
    validate_chapter_id(req.chapter_id)
    manifest = _load_manifest_raw(req.chapter_id)
    page = manifest["pages"][req.page_index]
    box = page["boxes"][req.box_index]

    data = np.fromfile(page["original"], dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    h, w = image.shape[:2]

    pad = 20
    y1 = max(0, box["y1"] - pad)
    y2 = min(h, box["y2"] + pad)
    x1 = max(0, box["x1"] - pad)
    x2 = min(w, box["x2"] + pad)

    crop = image[y1:y2, x1:x2]
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    text = ocr.read(crop_rgb, req.lang)

    with _get_manifest_lock(req.chapter_id):
        manifest = _load_manifest_raw(req.chapter_id)
        manifest["pages"][req.page_index]["boxes"][req.box_index]["ocr_text"] = text
        _save_manifest_raw(req.chapter_id, manifest)

    return {"text": text}


@app.post("/api/save_draft")
def save_draft(req: SaveDraftRequest) -> dict:
    validate_chapter_id(req.chapter_id)
    with _get_manifest_lock(req.chapter_id):
        manifest = _load_manifest_raw(req.chapter_id)
        if "drafts" not in manifest:
            manifest["drafts"] = {}
        manifest["drafts"].update(req.drafts)
        _save_manifest_raw(req.chapter_id, manifest)
    return {"ok": True}


def _style_get(d: dict, idx: int, default=None):
    """Lookup style value by box index — keys may be int or str after JSON/Pydantic."""
    if not d:
        return default
    if idx in d:
        return d[idx]
    s = str(idx)
    if s in d:
        return d[s]
    return default


@app.post("/api/render")
def render_page(req: RenderRequest) -> dict:
    validate_chapter_id(req.chapter_id)
    manifest = _load_manifest_raw(req.chapter_id)
    if req.page_index < 0 or req.page_index >= len(manifest["pages"]):
        raise HTTPException(400, f"Invalid page_index: {req.page_index}")
    page = manifest["pages"][req.page_index]
    logger.info(
        f"Chapter {req.chapter_id} page {req.page_index}: "
        f"rendering {len(req.translations)} translations"
    )

    base_image_path = page.get("clean") or page.get("original")
    if not base_image_path:
        raise HTTPException(400, "Page has no image to render onto")
    base_path = Path(base_image_path)
    if not base_path.is_file():
        raise HTTPException(
            404,
            f"Base image not found: {base_path.name}. "
            "Hãy chạy xử lý trang (process) trước khi chèn chữ.",
        )

    try:
        image = Image.open(base_path).convert("RGB")
    except Exception as e:
        logger.exception("Failed to open base image %s", base_path)
        raise HTTPException(500, f"Cannot open base image: {e}") from e

    colors_dict = req.colors or {}
    fonts_dict = req.fonts or {}
    font_sizes_dict = req.font_sizes or {}
    bolds_dict = req.bolds or {}
    stroke_w_dict = req.stroke_widths or {}
    stroke_c_dict = req.stroke_colors or {}
    bg_colors_dict = req.bg_colors or {}
    radii_dict = req.corner_radii or {}

    rendered_count = 0
    for box_key, translation in req.translations.items():
        try:
            box_idx = int(box_key)
        except (TypeError, ValueError):
            continue
        if box_idx < 0 or box_idx >= len(page["boxes"]):
            continue
        box = page["boxes"][box_idx]
        if box.get("removed"):
            continue
        if not (translation or "").strip():
            continue

        coords = (
            int(box["x1"]),
            int(box["y1"]),
            int(box["x2"]),
            int(box["y2"]),
        )
        box_color = _style_get(colors_dict, box_idx, "auto")
        box_font = _style_get(fonts_dict, box_idx, "default")
        box_size = _style_get(font_sizes_dict, box_idx, "auto")
        bold_val = _style_get(bolds_dict, box_idx, False)
        box_bold = bool(bold_val) if bold_val is not None else False
        stroke_w = _style_get(stroke_w_dict, box_idx, "auto")
        stroke_c = _style_get(stroke_c_dict, box_idx, "auto")
        bg_c = _style_get(bg_colors_dict, box_idx, "transparent")
        r_raw = _style_get(radii_dict, box_idx, 0)
        try:
            r_val = int(r_raw or 0)
        except (TypeError, ValueError):
            r_val = 0

        try:
            image = render_text_in_box(
                image,
                translation.strip(),
                coords,
                fill=box_color,
                font_size=box_size,
                is_bold=box_bold,
                font_name=box_font or "default",
                stroke_width=stroke_w,
                stroke_color=stroke_c,
                bg_color=bg_c,
                corner_radius=r_val,
            )
            rendered_count += 1
        except Exception as e:
            logger.exception(
                "Render failed for chapter %s page %s box %s",
                req.chapter_id,
                req.page_index,
                box_idx,
            )
            raise HTTPException(
                500,
                f"Chèn chữ thất bại (ô {box_idx}): {e}",
            ) from e

    out_dir = OUTPUT_DIR / req.chapter_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"page_{req.page_index:03d}.png"
    try:
        image.save(out_path)
    except Exception as e:
        logger.exception("Failed to save rendered page %s", out_path)
        raise HTTPException(500, f"Cannot save rendered image: {e}") from e

    page["rendered"] = True
    _save_manifest_raw(req.chapter_id, manifest)

    logger.info(
        "Chapter %s page %s: rendered %s box(es) -> %s",
        req.chapter_id,
        req.page_index,
        rendered_count,
        out_path.name,
    )
    return {"output": f"/api/image/{req.chapter_id}/{req.page_index}/rendered"}


@app.get("/api/download/{chapter_id}/{page_index}")
def download_page(chapter_id: str, page_index: int):
    validate_chapter_id(chapter_id)
    if page_index < 0:
        raise HTTPException(400, "page_index must be >= 0")
    path = OUTPUT_DIR / chapter_id / f"page_{page_index:03d}.png"
    if not path.exists():
        raise HTTPException(404, "Output image not found")
    return FileResponse(path)


@app.get("/health")
def health():
    from app.config import check_models
    missing = check_models()
    return {
        "status": "ok" if not missing else "degraded",
        "models_missing": missing,
    }


@app.get("/")
def index():
    return FileResponse("app/templates/index.html")
