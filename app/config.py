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
]

DEFAULT_FONT = BASE_DIR / "app" / "static" / "fonts" / "default.ttf"

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
        missing.append(LAMA_MODEL.name)
    return missing
