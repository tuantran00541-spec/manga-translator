from app import ort_utils


def test_openvino_takes_the_text_detector_and_leaves_the_varying_inputs(monkeypatch):
    monkeypatch.setenv("MANGA_ORT_PROVIDER", "openvino")
    monkeypatch.setenv("MANGA_ORT_OPENVINO_SCOPE", "detectors")
    assert ort_utils._openvino_selected("models/kiuyha-tuned-1023.onnx")
    assert not ort_utils._openvino_selected("models/ctd_seg.onnx")
    assert not ort_utils._openvino_selected("models/lama-manga-dynamic.onnx")
