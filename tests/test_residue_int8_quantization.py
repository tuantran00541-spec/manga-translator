from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "quantize_text_segmenter_int8.py"
WORKFLOW = ROOT / ".github" / "workflows" / "residue-640-int8-ab.yml"


def _load_quantizer():
    assert SCRIPT.is_file(), "INT8 quantization build tool is missing"
    spec = importlib.util.spec_from_file_location("quantize_text_segmenter_int8", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_calibration_preprocessing_matches_production_letterbox():
    from app.detector.bubble_detector import YoloDetector

    quantizer = _load_quantizer()
    image = np.arange(117 * 263 * 3, dtype=np.uint8).reshape(117, 263, 3)

    detector = YoloDetector.__new__(YoloDetector)
    detector.input_size = 640
    production_blob, _ = detector._preprocess(image)

    calibration_blob = quantizer.preprocess_calibration_image(image, 640)

    assert production_blob is not None
    assert calibration_blob.dtype == np.float32
    assert np.array_equal(calibration_blob, production_blob)


def test_calibration_window_is_deterministic_and_disjoint(tmp_path):
    quantizer = _load_quantizer()
    for name in ("005.png", "001.webp", "004.jpg", "002.png", "003.jpeg", "006.bmp"):
        (tmp_path / name).write_bytes(b"x")

    paths = quantizer.select_calibration_images(
        tmp_path,
        start_index=2,
        subset_size=3,
    )

    assert [path.name for path in paths] == ["003.jpeg", "004.jpg", "005.png"]


def test_quantizer_refuses_to_overwrite_source_model():
    quantizer = _load_quantizer()

    with pytest.raises(ValueError, match="overwrite"):
        quantizer.validate_model_paths(Path("model.onnx"), Path("model.onnx"))


def test_int8_workflow_is_retry_only_and_has_hard_parity_gate():
    assert WORKFLOW.is_file(), "INT8 A/B workflow is missing"
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "text_segmenter_640_int8.onnx" in text
    assert "MANGA_DETECTOR_RETRY_SEGMENTER_ENABLED" in text
    assert "authority masks changed" in text
    assert "outside authority" in text.lower()
    assert "models/bubble_yolo.onnx" not in text
    assert "cp models/text_segmenter_640_int8.onnx models/text_segmenter.onnx" not in text


def test_int8_artifact_name_does_not_escape_github_expression():
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "\\${{ github.run_id }}" not in text
    assert "name: residue-640-int8-${{ github.run_id }}" in text
