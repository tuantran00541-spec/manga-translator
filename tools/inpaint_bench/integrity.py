from __future__ import annotations
import hashlib
from pathlib import Path

# Production baseline SHA-256 integrity map
# These files must remain completely unchanged throughout Phase 0/0.1/0.2.
PRODUCTION_BASELINE_HASHES = {
    "app/inpaint/lama_inpainter.py": "",
    "app/ort_utils.py": "",
}


def compute_file_sha256(path: Path | str) -> str:
    p = Path(path)
    if not p.is_file():
        return ""
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def get_production_hashes() -> dict[str, str]:
    return {
        "app/inpaint/lama_inpainter.py": compute_file_sha256("app/inpaint/lama_inpainter.py"),
        "app/ort_utils.py": compute_file_sha256("app/ort_utils.py"),
    }
