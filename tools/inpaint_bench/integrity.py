from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Any

# ==============================================================================
# IMMUTABLE PRODUCTION & MODEL BASELINE SHA-256 HASHES
# These hashes are the exact byte-for-byte baseline fingerprints established
# during Phase 0. Under NO circumstances should production files or model be modified.
# ==============================================================================

LAMA_INPAINTER_BASELINE_SHA256 = (
    "1d6046e7fbb64f2db163a8301fa3839aa6400dbdc270fe17fa008fe37ba42a42"
)
ORT_UTILS_BASELINE_SHA256 = (
    "9d5b066d7cefa089d81d2ef39d22be3f5ea27b949bc54b66dfa891e4f4841f39"
)
LAMA_MODEL_BASELINE_SHA256 = (
    "e4b3e648c668b556942ad7096e23616a2ef74092b1be753d0c9c7f66a2e48fae"
)

PRODUCTION_BASELINE_HASHES = {
    "app/inpaint/lama_inpainter.py": LAMA_INPAINTER_BASELINE_SHA256,
    "app/ort_utils.py": ORT_UTILS_BASELINE_SHA256,
    "models/lama.onnx": LAMA_MODEL_BASELINE_SHA256,
}


def compute_file_sha256(path: Path | str) -> str:
    p = Path(path)
    if not p.is_file():
        return ""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_trusted_baseline_manifest(manifest_path: Path | str | None = None) -> dict[str, Any]:
    import json
    p = Path(manifest_path) if manifest_path else Path(__file__).parent / "baseline_manifest.json"
    if not p.is_file():
        raise FileNotFoundError(f"Baseline manifest not found at: {p}")
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    prod_hashes = data.get("production_hashes", {})
    if prod_hashes.get("app/inpaint/lama_inpainter.py") != LAMA_INPAINTER_BASELINE_SHA256:
        raise ValueError("Baseline manifest contains tampered lama_inpainter hash")
    if prod_hashes.get("app/ort_utils.py") != ORT_UTILS_BASELINE_SHA256:
        raise ValueError("Baseline manifest contains tampered ort_utils hash")
    if data.get("model_hash") != LAMA_MODEL_BASELINE_SHA256:
        raise ValueError("Baseline manifest contains tampered model hash")

    return data


def verify_production_integrity(
    base_dir: Path | str = ".",
    actual_model_path: Path | str | None = None,
) -> tuple[bool, dict[str, Any]]:
    base_path = Path(base_dir)
    results = {}
    all_valid = True

    for rel_path, expected_hash in PRODUCTION_BASELINE_HASHES.items():
        if rel_path == "models/lama.onnx" and actual_model_path:
            file_path = Path(actual_model_path)
        else:
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
            "error": "" if is_match else f"SHA-256 mismatch for {file_path}: expected {expected_hash}, got {actual_hash}",
        }
        if not is_match:
            all_valid = False

    return all_valid, results
