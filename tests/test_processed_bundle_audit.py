from __future__ import annotations

import json
from pathlib import Path
import zipfile

import cv2
import numpy as np

from scripts.audit_processed_bundle import audit_bundle


def _png(image: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    return buffer.tobytes()


def test_bundle_audit_counts_human_repaint_and_enforces_mask_authority(tmp_path: Path):
    bundle = tmp_path / "chapter.zip"
    automatic = np.full((20, 30, 3), 200, np.uint8)
    final = automatic.copy()
    final[5:10, 7:13] = 240
    mask = np.zeros((20, 30), np.uint8)
    mask[5:10, 7:13] = 255
    manifest = {
        "chapter_id": "c0ffee12",
        "pages": [{
            "original": "data/raw/c0ffee12/000.png",
            "clean": "data/processed/c0ffee12/clean_000.png",
            "manual_mask": "data/processed/c0ffee12/manual_mask_000.png",
            "boxes": [{"safe_to_inpaint": True}],
            "skipped": False,
        }],
    }
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("chapter/manifest.json", json.dumps(manifest))
        archive.writestr("chapter/auto_clean_000.png", _png(automatic))
        archive.writestr("chapter/clean_000.png", _png(final))
        archive.writestr("chapter/manual_mask_000.png", _png(mask))

    report = audit_bundle(bundle)
    assert report["status"] == "pass"
    assert report["source_sha256"]
    assert report["counts"]["manual_correction_slices"] == 1
    assert report["manual_evidence"]["paired_auto_final_slices"] == 1
    assert report["manual_evidence"]["outside_manual_changed_pixels"] == 0
    assert report["metrics"]["automatic_completion_proxy"] == 0.0
