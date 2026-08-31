import os
from pathlib import Path

from app.parameters import (
    BUBBLE_DESTRUCTIVE_CONF_THRESHOLD,
    BUBBLE_IOU_THRESHOLD,
    DETECTOR_TTA_ENABLED,
    INPAINT_SIZE,
    MASK_DILATE_KERNEL_SIZE,
    MAX_FONT_SIZE,
    MIN_FONT_SIZE,
    SLICE_MAX_HEIGHT,
    SLICE_MIN_HEIGHT,
    SLICE_SEARCH_WINDOW,
    SLICE_TARGET_HEIGHT,
    SMART_FILL_CLEAN_RING_MARGIN,
    TEXT_CONF_THRESHOLD,
)

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "data" / "output"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

BUBBLE_DETECTOR_MODEL = MODELS_DIR / "bubble_yolo.onnx"
TEXT_SEGMENTER_MODEL = MODELS_DIR / "text_segmenter.onnx"
LAMA_MODEL = MODELS_DIR / "lama.onnx"
LAMA_DYNAMIC_MODEL = MODELS_DIR / "lama-manga-dynamic.onnx"

REQUIRED_MODELS = [
    BUBBLE_DETECTOR_MODEL,
    TEXT_SEGMENTER_MODEL,
]

# Compatibility aliases for modules/third-party code that historically imported
# tuning from app.config. New runtime code should import from app.parameters.
BUBBLE_CONF_THRESHOLD = BUBBLE_DESTRUCTIVE_CONF_THRESHOLD
ENABLE_TTA = DETECTOR_TTA_ENABLED

DEFAULT_FONT = BASE_DIR / "app" / "static" / "fonts" / "default.ttf"

SUPPORTED_OCR_LANGS = ["ja", "ch", "korean", "en"]

HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))
RELOAD = os.getenv("RELOAD", "0") == "1"
WORKERS = int(os.getenv("WORKERS", "1"))


def ensure_directories() -> None:
    for d in (RAW_DIR, PROCESSED_DIR, OUTPUT_DIR, MODELS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def check_models() -> list[str]:
    missing = []
    for path in REQUIRED_MODELS:
        if not path.is_file():
            missing.append(path.name)
    if not LAMA_DYNAMIC_MODEL.is_file() and not LAMA_MODEL.is_file():
        # Preserve the historical filename in the public response while accepting
        # the preferred dynamic model as a complete inpaint backend on its own.
        missing.append(LAMA_MODEL.name)
    return missing
