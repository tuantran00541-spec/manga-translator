import os
from pathlib import Path

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
    LAMA_MODEL,
]

BUBBLE_CONF_THRESHOLD = 0.4
BUBBLE_IOU_THRESHOLD = 0.3
TEXT_CONF_THRESHOLD = 0.20

ENABLE_TTA = os.getenv("ENABLE_TTA", "0").lower() in ("1", "true", "yes")

MASK_DILATE_KERNEL_SIZE = 7
SMART_FILL_CLEAN_RING_MARGIN = 6

INPAINT_SIZE = 512

SLICE_TARGET_HEIGHT = 1400
SLICE_SEARCH_WINDOW = 180
SLICE_MIN_HEIGHT = 500
SLICE_MAX_HEIGHT = 1536

DEFAULT_FONT = BASE_DIR / "app" / "static" / "fonts" / "default.ttf"
MIN_FONT_SIZE = 6
MAX_FONT_SIZE = 48

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
    return missing
