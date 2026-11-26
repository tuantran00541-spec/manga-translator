from contextlib import asynccontextmanager
import mimetypes
import os
from pathlib import Path
import time
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import (
    BASE_DIR,
    HOST,
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
from app.manifest_utils import (
    cleanup_stale_temp_artifacts,
    recover_page_artifact_transactions,
)
from app.media_downloader import cleanup_expired_media_jobs
from app.parameters import (
    FIXED_LAMA_CONCURRENT_INFERENCE,
    INPAINT_PRELOAD_ENABLED,
    STALE_TEMP_MAX_AGE_SECONDS,
    USE_DYNAMIC_LAMA,
)
from app.routers import account, agent, ai_mode, api_proxy, automation, chapters, editor, export, extension, image, media, ocr, render, render_commit, studio, translation, visual_qc
from app.security import MAX_REQUEST_BYTES, MAX_UPLOAD_TOTAL_BYTES

MAX_STUDIO_UPLOAD_BYTES = 4 * 1024 * 1024 * 1024  # 4 GB max for long-form local movie uploads


def _cleanup_extra_stale_artifacts(
    max_age_seconds: float = STALE_TEMP_MAX_AGE_SECONDS,
) -> int:
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
    recovered_transactions = 0
    if PROCESSED_DIR.exists():
        for chapter_dir in PROCESSED_DIR.iterdir():
            if chapter_dir.is_dir():
                recovered_transactions += recover_page_artifact_transactions(
                    chapter_dir
                )
    if recovered_transactions:
        logger.warning(
            "Recovered {} interrupted page artifact transaction(s)",
            recovered_transactions,
        )
    removed_temp_artifacts = _cleanup_extra_stale_artifacts()
    removed_temp_artifacts += cleanup_expired_media_jobs()
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
            "Models missing: {} — /api/process_pages will fail until models are placed in models/",
            ", ".join(missing),
        )
    else:
        logger.info("ONNX models OK")
        if INPAINT_PRELOAD_ENABLED:
            try:
                pipeline.inpainter.preload()
            except Exception:
                logger.exception(
                    "Startup inpaint model preparation failed; "
                    "page processing will report the same model error"
                )
    yield


def register_javascript_mime_type() -> None:
    mimetypes.add_type("text/javascript", ".js")


register_javascript_mime_type()
app = FastAPI(lifespan=lifespan, title="Manga Translator", version="0.3.2")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "app" / "static")), name="static")


class RequestSizeLimitMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        path = str(scope.get("path") or "")
        if path == "/api/studio/projects/upload":
            limit = MAX_STUDIO_UPLOAD_BYTES
        elif path == "/api/chapter/upload":
            limit = MAX_UPLOAD_TOTAL_BYTES
        else:
            limit = MAX_REQUEST_BYTES
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


def allowed_hosts() -> set[str]:
    hosts = {"127.0.0.1", "localhost", "::1"}
    if HOST not in ("0.0.0.0", "::", ""):
        hosts.add(HOST.lower())
    hosts.update(h.strip().lower() for h in os.getenv("MANGA_ALLOWED_HOSTS", "").split(",") if h.strip())
    return hosts


def _host_name(value: str) -> str:
    value = value.strip().lower()
    if value.startswith("["):
        return value[1:].split("]", 1)[0]
    return value.rsplit(":", 1)[0] if value.count(":") == 1 else value


class LocalHostOnlyMiddleware:

    def __init__(self, app, hosts: set[str] | None = None):
        self.app = app
        self.hosts = hosts if hosts is not None else allowed_hosts()

    async def __call__(self, scope, receive, send):
        if scope.get("type") in ("http", "websocket"):
            host = next((v.decode("latin-1") for k, v in scope.get("headers", []) if k.lower() == b"host"), "")
            if _host_name(host) not in self.hosts:
                if scope["type"] == "websocket":
                    await send({"type": "websocket.close", "code": 1008})
                    return
                await JSONResponse({"detail": "Host not allowed"}, status_code=403)(scope, receive, send)
                return
        await self.app(scope, receive, send)


_STATE_CHANGING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
_EXTENSION_UPLOAD_PATHS = frozenset({"/api/extension/hello", "/api/chapter/upload"})
_DISALLOWED_SIMPLE_CONTENT_TYPES = frozenset({"text/plain", "application/x-www-form-urlencoded"})
_SECURITY_HEADERS: tuple[tuple[bytes, bytes], ...] = (
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
    (b"referrer-policy", b"no-referrer"),
)


class CSRFAndSecurityHeadersMiddleware:

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        method = str(scope.get("method") or "GET").upper()
        path = str(scope.get("path") or "")
        headers = {k.lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}

        if method in _STATE_CHANGING_METHODS and path.startswith("/api/"):
            req_host = (headers.get(b"host") or "").strip().lower()
            origin = (headers.get(b"origin") or "").strip()
            sec_fetch = (headers.get(b"sec-fetch-site") or "").strip().lower()
            is_ext_origin = origin.lower().startswith("chrome-extension://") and path in _EXTENSION_UPLOAD_PATHS

            if sec_fetch and sec_fetch not in ("same-origin", "none") and not is_ext_origin:
                await JSONResponse({"detail": "Cross-site requests are forbidden"}, status_code=403)(scope, receive, send)
                return

            if origin and not is_ext_origin:
                parsed = urlsplit(origin)
                if parsed.scheme not in ("http", "https") or not parsed.netloc or (req_host and parsed.netloc.lower() != req_host):
                    await JSONResponse({"detail": "Cross-origin requests are forbidden"}, status_code=403)(scope, receive, send)
                    return
            elif not origin:
                referer = (headers.get(b"referer") or "").strip()
                if referer:
                    parsed_ref = urlsplit(referer)
                    if parsed_ref.netloc and req_host and parsed_ref.netloc.lower() != req_host:
                        await JSONResponse({"detail": "Cross-origin requests are forbidden"}, status_code=403)(scope, receive, send)
                        return

            ctype = (headers.get(b"content-type") or "").split(";", 1)[0].strip().lower()
            if ctype in _DISALLOWED_SIMPLE_CONTENT_TYPES:
                await JSONResponse({"detail": "Unsupported Content-Type"}, status_code=415)(scope, receive, send)
                return

        async def send_with_headers(message):
            if message.get("type") == "http.response.start":
                raw_headers = list(message.get("headers") or [])
                existing = {k.lower() for k, _ in raw_headers}
                for hk, hv in _SECURITY_HEADERS:
                    if hk not in existing:
                        raw_headers.append((hk, hv))
                message = {**message, "headers": raw_headers}
            await send(message)

        await self.app(scope, receive, send_with_headers)


app.add_middleware(RequestSizeLimitMiddleware)
app.add_middleware(CSRFAndSecurityHeadersMiddleware)
app.add_middleware(LocalHostOnlyMiddleware)


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
app.include_router(ai_mode.router)
app.include_router(account.router)
app.include_router(agent.router)
app.include_router(api_proxy.router)
app.include_router(extension.router)
app.include_router(media.router)
app.include_router(studio.router)


def _current_rss_bytes() -> int | None:
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
    inpaint_serialized = None
    inpaint_load_status = {
        "state": "idle",
        "load_ms": None,
        "failed": False,
    }
    if inpainter is not None:
        status_provider = getattr(inpainter, "session_load_status", None)
        if callable(status_provider):
            inpaint_load_status = status_provider()
    if inpaint_session_loaded:
        model_path = getattr(inpainter, "lama_model_path", None)
        active_inpaint_model = Path(model_path).name if model_path else None
        inpaint_dynamic = bool(getattr(inpainter, "dynamic_lama", False))
        inpaint_serialized = bool(
            getattr(inpainter, "serialized_inference", not inpaint_dynamic)
        )

    prefers_dynamic_inpaint = USE_DYNAMIC_LAMA and LAMA_DYNAMIC_MODEL.is_file()
    preferred_inpaint_model = (
        LAMA_DYNAMIC_MODEL.name
        if prefers_dynamic_inpaint
        else LAMA_MODEL.name
    )

    return {
        "rss_bytes": _current_rss_bytes(),
        "models": {
            "detector": {
                "resident": detector is not None,
                "session_loaded": bool(detector is not None and getattr(detector, "session", None) is not None),
            },
            "inpaint": {
                "object_created": inpainter is not None,
                "session_loaded": inpaint_session_loaded,
                "load_state": inpaint_load_status["state"],
                "load_ms": inpaint_load_status["load_ms"],
                "load_failed": inpaint_load_status["failed"],
                "preload_enabled": INPAINT_PRELOAD_ENABLED,
                "preferred_model": preferred_inpaint_model,
                "active_model": active_inpaint_model,
                "dynamic": inpaint_dynamic,
                "serialized_inference": inpaint_serialized,
                "serialization_scope": (
                    "global" if inpaint_serialized else None
                ),
                "preferred_serialized_inference": bool(
                    not prefers_dynamic_inpaint
                    and not FIXED_LAMA_CONCURRENT_INFERENCE
                ),
                "fixed_available": LAMA_MODEL.is_file(),
                "dynamic_available": LAMA_DYNAMIC_MODEL.is_file(),
            },
            "ocr": {
                "paddle_loaded": sorted(str(key) for key in paddle_pipelines.keys()),
            },
        },
    }


@app.get("/health")
def health():
    missing = check_models()
    runtime = _runtime_state()
    inpaint_failed = bool(
        runtime.get("models", {}).get("inpaint", {}).get("load_failed", False)
    )
    return {
        "status": "ok" if not missing and not inpaint_failed else "degraded",
        "models_missing": missing,
        "version": app.version,
        "runtime": runtime,
    }


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "app" / "templates" / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse(BASE_DIR / "app" / "static" / "favicon.ico", media_type="image/x-icon")
