#!/usr/bin/env python3
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading

import cv2
import numpy as np

import app.ocr.service as service_module
from app.ocr.service import OCRService

CHAPTER_ID = "deadbeef"


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

    def read_detailed(self, image: np.ndarray, _lang: str):
        with self._calls_lock:
            self.calls += 1
        self._barrier.wait(timeout=5)
        mean = float(image.mean())
        if mean < 120.0:
            return SimpleNamespace(
                text="DARK",
                confidence=0.91,
                model="fake-dark",
                orientation="horizontal",
                region_count=1,
                quality="good",
                quality_reason=None,
            )
        return SimpleNamespace(
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
    service_module.save_manifest_raw = save
    service_module.invalidate_page_render = invalidate


def _assert_grouped_translation_ownership() -> None:
    obj = {
        "ocr_text": "OLD GROUP SOURCE",
        "translation": "OLD GROUP TRANSLATION",
        "translation_source": "deepseek",
        "translation_model": "deepseek-chat",
        "translation_input_text": "OLD GROUP SOURCE",
        "auto_translation": "OLD GROUP TRANSLATION",
    }
    OCRService._stamp_group_object(
        obj,
        source_box_ids=["box_a", "box_b"],
        combined="NEW GROUP SOURCE",
        lang="en",
        engine="test-engine",
        source_revision=4,
        original_revision=(100, 200, 300),
        region={"x1": 1, "y1": 2, "x2": 50, "y2": 60},
    )
    assert obj["ocr_text"] == "NEW GROUP SOURCE"
    assert obj["translation"] == ""
    assert "translation_source" not in obj
    assert "translation_model" not in obj
    assert "translation_input_text" not in obj
    assert "auto_translation" not in obj
    assert obj["ocr_quality"] == "review"
    assert obj["ocr_quality_reason"] == "grouped-machine-ocr"

    obj.update(
        {
            "translation": "MANUAL GROUP TRANSLATION",
            "translation_source": "deepseek",
            "translation_model": "deepseek-chat",
            "translation_input_text": "NEW GROUP SOURCE",
            "auto_translation": "AUTO GROUP TRANSLATION",
        }
    )
    OCRService._stamp_group_object(
        obj,
        source_box_ids=["box_a", "box_b"],
        combined="THIRD GROUP SOURCE",
        lang="en",
        engine="test-engine",
        source_revision=5,
        original_revision=(100, 200, 300),
        region={"x1": 1, "y1": 2, "x2": 50, "y2": 60},
    )
    assert obj["ocr_text"] == "THIRD GROUP SOURCE"
    assert obj["translation"] == "MANUAL GROUP TRANSLATION"
    assert obj["auto_translation"] == "AUTO GROUP TRANSLATION"
    assert obj["ocr_quality"] == "review"


def _assert_legacy_reader_metadata() -> None:
    class LegacyEngine:
        def read(self, _image: np.ndarray, _lang: str) -> str:
            return "HELLO"

    service = OCRService(LegacyEngine(), _FakePipeline())
    service._cached_source_image = lambda _path: np.full((32, 64, 3), 255, dtype=np.uint8)
    text = service._read_box_text(
        Path("unused.png"),
        {"x1": 0, "y1": 0, "x2": 64, "y2": 32},
        "en",
    )
    metadata = service._result_local.metadata
    assert text == "HELLO"
    assert metadata["model"] == "legacy-reader"
    assert metadata["region_count"] == 1
    assert metadata["quality"] == "good"


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="ocr-service-fold-sanity-") as temp_dir:
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
                    "text_objects": [
                        {
                            "id": "text_dark",
                            "shape": "rectangle",
                            "region": {"x1": 10, "y1": 20, "x2": 70, "y2": 80},
                            "source_boxes": ["dark"],
                            "ocr_text": "OLD DARK",
                            "auto_ocr_text": "OLD DARK",
                            "auto_geometry": {"x1": 10, "y1": 20, "x2": 70, "y2": 80},
                            "translation": "BAN DICH CU",
                            "translation_source": "deepseek",
                            "translation_model": "deepseek-chat",
                            "translation_input_text": "OLD DARK",
                            "auto_translation": "BAN DICH CU",
                            "auto_generated": True,
                        }
                    ],
                }
            ]
        }
        manifest_lock = threading.RLock()
        _install_manifest_fakes(manifest, manifest_lock)

        engine = _ConcurrentDetailedEngine()
        pipeline = _FakePipeline()
        service = OCRService(engine, pipeline)

        with ThreadPoolExecutor(max_workers=2) as executor:
            dark_future = executor.submit(
                service.inspect_box_id, CHAPTER_ID, 0, "dark", "en"
            )
            light_future = executor.submit(
                service.inspect_box_id, CHAPTER_ID, 0, "light", "en"
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

        text_obj = manifest["pages"][0]["text_objects"][0]
        assert text_obj["ocr_text"] == "DARK"
        assert text_obj["auto_ocr_text"] == "DARK"
        assert text_obj["ocr_model"] == "fake-dark"
        assert text_obj["ocr_quality"] == "good"
        assert text_obj["translation"] == ""
        assert "translation_source" not in text_obj
        assert "translation_model" not in text_obj
        assert "translation_input_text" not in text_obj
        assert "auto_translation" not in text_obj

        dark_cached = service.inspect_box_id(CHAPTER_ID, 0, "dark", "en")
        light_cached = service.inspect_box_id(CHAPTER_ID, 0, "light", "en")
        assert dark_cached["cached"] and light_cached["cached"]
        assert dark_cached["model"] == "fake-dark"
        assert light_cached["model"] == "fake-light"
        assert engine.calls == 2
        assert pipeline.sync_calls == 2

    _assert_grouped_translation_ownership()
    _assert_legacy_reader_metadata()
    print("OCRService fold cache/concurrency/propagation/grouped invariants: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
