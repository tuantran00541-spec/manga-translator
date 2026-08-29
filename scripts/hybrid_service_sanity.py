#!/usr/bin/env python3
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import threading

import cv2
import numpy as np

import app.manifest_utils as manifest_utils
import app.ocr.service as service_module
from app.ocr.hybrid_service import HybridOCRService
from app.ocr.paddle_v6 import OCRReadResult


class _FakePipeline:
    def __init__(self) -> None:
        self.sync_calls = 0

    def _sync_output_dir(self, _chapter_id: str, _manifest: dict, _pages: list[int]) -> None:
        self.sync_calls += 1


class _ConcurrentDetailedEngine:
    def __init__(self) -> None:
        self.calls = 0
        self._calls_lock = threading.Lock()
        self._barrier = threading.Barrier(2)

    def read_detailed(self, image: np.ndarray, _lang: str) -> OCRReadResult:
        with self._calls_lock:
            self.calls += 1
        self._barrier.wait(timeout=5)
        mean = float(image.mean())
        if mean < 120.0:
            return OCRReadResult(
                text="DARK",
                confidence=0.91,
                model="fake-dark",
                orientation="horizontal",
                region_count=1,
                quality="good",
                quality_reason=None,
            )
        return OCRReadResult(
            text="LIGHT",
            confidence=0.97,
            model="fake-light",
            orientation="horizontal",
            region_count=1,
            quality="good",
            quality_reason=None,
        )


def _install_manifest_fakes(manifest: dict, lock: threading.RLock) -> None:
    def get_lock(_chapter_id: str):
        return lock

    def load(_chapter_id: str) -> dict:
        return manifest

    def save(_chapter_id: str, _manifest: dict) -> None:
        assert _manifest is manifest

    def invalidate(_manifest: dict, _page_index: int) -> None:
        assert _manifest is manifest

    service_module.get_manifest_lock = get_lock
    service_module.load_manifest_raw = load
    manifest_utils.get_manifest_lock = get_lock
    manifest_utils.load_manifest_raw = load
    manifest_utils.save_manifest_raw = save
    manifest_utils.invalidate_page_render = invalidate


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="hybrid-service-sanity-") as temp_dir:
        source = Path(temp_dir) / "source.png"
        image = np.full((100, 220, 3), 230, dtype=np.uint8)
        image[:, :105] = 20
        ok, encoded = cv2.imencode(".png", image)
        assert ok
        encoded.tofile(source)

        manifest = {
            "pages": [
                {
                    "original": str(source),
                    "source_revision": 3,
                    "boxes": [
                        {
                            "id": "dark",
                            "x1": 10,
                            "y1": 20,
                            "x2": 70,
                            "y2": 80,
                            "ocr_eligible": True,
                        },
                        {
                            "id": "light",
                            "x1": 145,
                            "y1": 20,
                            "x2": 205,
                            "y2": 80,
                            "ocr_eligible": True,
                        },
                    ],
                }
            ]
        }
        manifest_lock = threading.RLock()
        _install_manifest_fakes(manifest, manifest_lock)

        engine = _ConcurrentDetailedEngine()
        pipeline = _FakePipeline()
        service = HybridOCRService(engine, pipeline)

        with ThreadPoolExecutor(max_workers=2) as executor:
            dark_future = executor.submit(
                service.inspect_box_id, "sanity", 0, "dark", "en"
            )
            light_future = executor.submit(
                service.inspect_box_id, "sanity", 0, "light", "en"
            )
            dark = dark_future.result(timeout=10)
            light = light_future.result(timeout=10)

        assert dark["text"] == "DARK", dark
        assert dark["model"] == "fake-dark", dark
        assert dark["confidence"] == 0.91, dark
        assert light["text"] == "LIGHT", light
        assert light["model"] == "fake-light", light
        assert light["confidence"] == 0.97, light
        assert not dark["cached"] and not light["cached"]
        assert engine.calls == 2
        assert pipeline.sync_calls == 2

        boxes = {box["id"]: box for box in manifest["pages"][0]["boxes"]}
        assert boxes["dark"]["ocr_text"] == "DARK"
        assert boxes["dark"]["ocr_model"] == "fake-dark"
        assert boxes["dark"]["ocr_confidence"] == 0.91
        assert boxes["dark"]["ocr_quality"] == "good"
        assert boxes["light"]["ocr_text"] == "LIGHT"
        assert boxes["light"]["ocr_model"] == "fake-light"
        assert boxes["light"]["ocr_confidence"] == 0.97
        assert boxes["light"]["ocr_quality"] == "good"

        # Second pass must use the persisted machine cache and must not invoke
        # the engine or another manifest sync.
        dark_cached = service.inspect_box_id("sanity", 0, "dark", "en")
        light_cached = service.inspect_box_id("sanity", 0, "light", "en")
        assert dark_cached["cached"] and light_cached["cached"]
        assert dark_cached["model"] == "fake-dark"
        assert light_cached["model"] == "fake-light"
        assert engine.calls == 2
        assert pipeline.sync_calls == 2

    print("hybrid service cache/concurrency invariants: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
