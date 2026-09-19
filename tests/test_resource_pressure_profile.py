from __future__ import annotations

import sys

import pytest


def test_retry_provider_override_is_model_local(monkeypatch):
    import app.ort_utils as ort_utils

    monkeypatch.setenv("MANGA_ORT_PROVIDER", "openvino")
    monkeypatch.setenv("MANGA_ORT_OPENVINO_SCOPE", "detectors")
    monkeypatch.setattr(
        ort_utils.ort,
        "get_available_providers",
        lambda: ["OpenVINOExecutionProvider", "CPUExecutionProvider"],
    )

    inherited, inherited_openvino = ort_utils._provider_stack(
        "models/text_segmenter_640.onnx"
    )
    cpu_only, cpu_openvino = ort_utils._provider_stack(
        "models/text_segmenter_640.onnx",
        provider_override="cpu",
    )

    assert inherited_openvino is True
    assert inherited[0][0] == "OpenVINOExecutionProvider"
    assert cpu_openvino is False
    assert cpu_only == ["CPUExecutionProvider"]


def test_profile_processing_accepts_controlled_window_arguments(monkeypatch, tmp_path):
    import scripts.profile_processing as profile_processing

    captured = {}

    def fake_prepare(args):
        captured["prepare_all"] = args.prepare_all
        captured["max_pages"] = args.max_pages

    monkeypatch.setattr(profile_processing, "prepare", fake_prepare)
    monkeypatch.setattr(profile_processing, "run", lambda args: pytest.fail("run should not execute"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "profile_processing.py",
            "--prepare",
            "--prepare-all",
            "--max-pages",
            "32",
            "--raw-dir",
            str(tmp_path),
        ],
    )

    profile_processing.main()

    assert captured == {"prepare_all": True, "max_pages": 32}
