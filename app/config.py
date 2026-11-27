import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "data" / "output"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

KIUYHA_TEXT_MODELS = (
    "kiuyha-tuned-1023.onnx",
    "kiuyha-tuned-transient-1024.onnx",
)
_kiuyha_override = os.getenv("MANGA_KIUYHA_MODEL", "").strip()
KIUYHA_TEXT_MODEL = (
    (Path(_kiuyha_override) if os.path.isabs(_kiuyha_override) else MODELS_DIR / _kiuyha_override)
    if _kiuyha_override
    else MODELS_DIR / KIUYHA_TEXT_MODELS[0]
)
LAMA_MODEL = MODELS_DIR / "lama.onnx"
LAMA_DYNAMIC_MODEL = MODELS_DIR / "lama-manga-dynamic.onnx"
MIGAN_MODEL = MODELS_DIR / "migan_pipeline_v2.onnx"
MIGAN_MODEL_URL = "https://huggingface.co/andraniksargsyan/migan/resolve/main/migan_pipeline_v2.onnx"
CTD_MODEL = MODELS_DIR / "ctd_seg.onnx"

REQUIRED_MODELS = [KIUYHA_TEXT_MODEL, CTD_MODEL]

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
