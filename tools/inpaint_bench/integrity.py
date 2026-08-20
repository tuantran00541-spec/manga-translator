from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Any

# ==============================================================================
# IMMUTABLE PRODUCTION BASELINE SHA-256 HASHES
# These hashes are the exact byte-for-byte baseline fingerprints established
# during Phase 0. Under NO circumstances should production files be modified.
# ==============================================================================

LAMA_INPAINTER_BASELINE_SHA256 = (
    "1d6046e7fbb64f2db163a8301fa3839aa6400dbdc270fe17fa008fe37ba42a42"
)
ORT_UTILS_BASELINE_SHA256 = (
    "9d5b066d7cefa089d81d2ef39d22be3f5ea27b949bc54b66dfa891e4f4841f39"
)

PRODUCTION_BASELINE_HASHES = {
    "app/inpaint/lama_inpainter.py": LAMA_INPAINTER_BASELINE_SHA256,
    "app/ort_utils.py": ORT_UTILS_BASELINE_SHA256,
}


def compute_file_sha256(path: Path | str) -> str:
    p = Path(path)
    if not p.is_file():
        return ""
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def verify_production_integrity(base_dir: Path | str = ".") -> tuple[bool, dict[str, Any]]:
    base_path = Path(base_dir)
    results = {}
    all_valid = True

    for rel_path, expected_hash in PRODUCTION_BASELINE_HASHES.items():
        file_path = base_path / rel_path
        if not file_path.is_file():
            results[rel_path] = {
                "exists": False,
                "expected_hash": expected_hash,
                "actual_hash": "",
                "valid": False,
                "error": f"File not found: {file_path}",
            }
            all_valid = False
            continue

        actual_hash = compute_file_sha256(file_path)
        is_match = (actual_hash.lower() == expected_hash.lower())
        results[rel_path] = {
            "exists": True,
            "expected_hash": expected_hash,
            "actual_hash": actual_hash,
            "valid": is_match,
            "error": "" if is_match else f"SHA-256 mismatch: expected {expected_hash}, got {actual_hash}",
        }
        if not is_match:
            all_valid = False

    return all_valid, results
