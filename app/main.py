from contextlib import asynccontextmanager
import os
from pathlib import Path
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import (
    BASE_DIR,
    LAMA_DYNAMIC_MODEL,
    LAMA_MODEL,
    OUTPUT_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    check_models,
    ensure_directories,
)
from app.dependencies import ocr as ocr_runtime, pipeline
from app.logging_config import logger
from app.manifest_utils import cleanup_stale_temp_artifacts
from app.parameters import USE_DYNAMIC_LAMA
from app.routers import automation, chapters, editor, export, image, ocr, render, render_commit, translation, visual_qc
from app.security import MAX_REQUEST_BYTES, MAX_UPLOAD_TOTAL_BYTES


def _cleanup_extra_stale_artifacts(max_age_seconds: float = 3600.0) -> int:
    """Clean crash leftovers not covered by per-chapter manifest temp cleanup."""
    cutoff = time.time() - max(0.0, float(max_age_seconds))
    candidates: set[Path] = set()

    if RAW_DIR.exists():
        candidates.update(RAW_DIR.rglob("*.part"))

    if OUTPUT_DIR.exists():
        candidates.update(OUTPUT_DIR.rglob("*.export.*.tmp"))
        candidates.update(OUTPUT_DIR.rglob("page_*.export.*.png"))

    removed = 0
    for path in candidates:
        try:
            if path.is_symlink() or not path.is_file():
                continue
            if path.stat().st_mtime > cutoff:
                continue
            path.unlink()
            removed += 1
        except OSError:
            continue
    return removed


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_directories()
    removed_temp_artifacts = _cleanup_extra_stale_artifacts()
    for root in (PROCESSED_DIR, OUTPUT_DIR):
        if not root.exists():
            continue
        for chapter_dir in root.iterdir():
            if chapter_dir.is_dir():
                removed_temp_artifacts += cleanup_stale_temp_artifacts(chapter_dir)
    if removed_temp_artifacts:
        logger.info("Removed {} stale temporary artifact(s) from previous runs", removed_temp_artifacts)

    missing = check_models()
    if missing:
        logger.warning(
            "Models missing: %s — /api/process_pages will fail until models are placed in models/",
            ", ".join(missing),
        )
    else:
        logger.info("ONNX models OK")
    yield


app = FastAPI(lifespan=lifespan, title="Manga Translator", version="0.2.0")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "app" / "static")), name="static")


class RequestSizeLimitMiddleware:
    """Enforce request limits on bytes actually received, not only headers."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        path = str(scope.get("path") or "")
        limit = (
            MAX_UPLOAD_TOTAL_BYTES
            if path == "/api/chapter/upload"
            else MAX_REQUEST_BYTES
        )
        headers = {
            key.lower(): value
            for key, value in scope.get("headers", [])
        }
        content_length = headers.get(b"content-length")
        if content_length:
            try:
                declared_size = int(content_length.decode("ascii"))
            except (UnicodeDecodeError, ValueError):
                response = JSONResponse(
                    {"detail": "Invalid Content-Length"}, status_code=400
                )
                await response(scope, receive, send)
                return
            if declared_size < 0:
                response = JSONResponse(
                    {"detail": "Invalid Content-Length"}, status_code=400
                )
                await response(scope, receive, send)
                return
            if declared_size > limit:
                response = JSONResponse(
                    {"detail": "Request too large"}, status_code=413
                )
                await response(scope, receive, send)
                return

        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message.get("type") == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise HTTPException(status_code=413, detail="Request too large")
            return message

        await self.app(scope, limited_receive, send)


app.add_middleware(RequestSizeLimitMiddleware)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
    logger.exception(f"Unhandled exception on {request.method} {request.url.path}")
    return JSONResponse({"detail": "Internal server error"}, status_code=500)


app.include_router(chapters.router)
app.include_router(automation.router)
app.include_router(translation.router)
app.include_router(ocr.router)
app.include_router(editor.router)
app.include_router(render_commit.router)
app.include_router(render.router)
app.include_router(image.router)
app.include_router(export.router)
app.include_router(visual_qc.router)


def _current_rss_bytes() -> int | None:
    """Return current process RSS without adding a runtime dependency."""
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(counters)
            process = ctypes.windll.kernel32.GetCurrentProcess()
            ok = ctypes.windll.psapi.GetProcessMemoryInfo(
                process,
                ctypes.byref(counters),
                counters.cb,
            )
            return int(counters.WorkingSetSize) if ok else None
        except Exception:
            return None

    try:
        statm = Path("/proc/self/statm").read_text(encoding="ascii").split()
        if len(statm) >= 2:
            return int(statm[1]) * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError, AttributeError):
        pass
    return None


def _runtime_state() -> dict:
    detector = getattr(pipeline, "_detector", None)
    inpainter = getattr(pipeline, "_inpainter", None)
    paddle = getattr(ocr_runtime, "_paddle", None)
    paddle_pipelines = getattr(paddle, "_pipelines", {}) if paddle is not None else {}

    inpaint_session_loaded = bool(
        inpainter is not None and getattr(inpainter, "session_loaded", False)
    )
    active_inpaint_model = None
    inpaint_dynamic = None
    if inpaint_session_loaded:
        model_path = getattr(inpainter, "lama_model_path", None)
        active_inpaint_model = Path(model_path).name if model_path else None
        inpaint_dynamic = bool(getattr(inpainter, "dynamic_lama", False))

    preferred_inpaint_model = (
        LAMA_DYNAMIC_MODEL.name
        if USE_DYNAMIC_LAMA and LAMA_DYNAMIC_MODEL.is_file()
        else LAMA_MODEL.name
    )

    return {
        "rss_bytes": _current_rss_bytes(),
        "models": {
            "detector": {
                "resident": detector is not None,
                "bubble_session_loaded": bool(
                    detector is not None
                    and getattr(getattr(detector, "bubble_detector", None), "session", None) is not None
                ),
                "text_session_loaded": bool(
                    detector is not None
                    and getattr(getattr(detector, "text_detector", None), "session", None) is not None
                ),
            },
            "inpaint": {
                "object_created": inpainter is not None,
                "session_loaded": inpaint_session_loaded,
                "preferred_model": preferred_inpaint_model,
                "active_model": active_inpaint_model,
                "dynamic": inpaint_dynamic,
                "fixed_available": LAMA_MODEL.is_file(),
                "dynamic_available": LAMA_DYNAMIC_MODEL.is_file(),
            },
            "ocr": {
                "manga_ocr_loaded": getattr(ocr_runtime, "_manga_ocr", None) is not None,
                "paddle_loaded": sorted(str(key) for key in paddle_pipelines.keys()),
            },
        },
    }


@app.get("/health")
def health():
    missing = check_models()
    return {
        "status": "ok" if not missing else "degraded",
        "models_missing": missing,
        "version": app.version,
        "runtime": _runtime_state(),
    }


@app.get("/")
def index():
    return FileResponse("app/templates/index.html")
