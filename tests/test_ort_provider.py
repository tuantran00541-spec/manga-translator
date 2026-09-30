from app import ort_utils


def test_openvino_takes_the_text_detector_and_leaves_the_varying_inputs(monkeypatch):
    # The scope named models removed long ago, so the text detector never ran on OpenVINO.
    monkeypatch.setenv("MANGA_ORT_PROVIDER", "openvino")
    monkeypatch.setenv("MANGA_ORT_OPENVINO_SCOPE", "detectors")
    assert ort_utils._openvino_selected("models/kiuyha_text_1280.onnx")
    assert not ort_utils._openvino_selected("models/ctd_seg.onnx")
    assert not ort_utils._openvino_selected("models/lama-manga-dynamic.onnx")
