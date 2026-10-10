import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "data" / "output"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

KIUYHA_TEXT_MODEL = MODELS_DIR / "kiuyha_text_1280.onnx"
LAMA_MODEL = MODELS_DIR / "lama.onnx"
LAMA_DYNAMIC_MODEL = MODELS_DIR / "lama-manga-dynamic.onnx"
CTD_MODEL = MODELS_DIR / "ctd_seg.onnx"

REQUIRED_MODELS = [KIUYHA_TEXT_MODEL, CTD_MODEL]

# H7: SHA256 pins for the ONNX model files. Fill these in from a trusted download; a model whose
# hash is pinned is verified on every load, and a model with no pin logs a warning instead of
# loading blind. (Keys are file names under MODELS_DIR.)
MODEL_SHA256: dict[str, str] = {
    "kiuyha_text_1280.onnx": "030961ef6041b17d43103e17c8b4dbd8840067e89d8898a4a12081bfe702b7cc",  # yolo26s.onnx from Kiuyha/Manga-Bubble-YOLO
    "lama-manga-dynamic.onnx": "de31ffa5ba26916b8ea35319f6c12151ff9654d4261bccf0583a69bb095315f9",  # from ogkalu/lama-manga-onnx-dynamic
    "ctd_seg.onnx": "dd7c2555ed56a123163c2144ebf9bf07352d412d7574d55e967b1168476d916d",  # seg-only extract from comictextdetector.pt.onnx (manga-image-translator beta-0.3)
    # "lama.onnx": "<sha256>",
}

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
