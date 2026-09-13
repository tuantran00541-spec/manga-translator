import threading
import time
from unittest.mock import patch

import pytest

import app.inpaint.lama_inpainter as lama_module
import app.main as main_module
from app.inpaint.lama_inpainter import Inpainter


def _configure_fake_session(inpainter):
    def configure(session, model_path, *, expected_dynamic):
        inpainter.session = session
        inpainter.lama_model_path = model_path
        inpainter.dynamic_lama = bool(expected_dynamic)

    return configure


def test_preload_builds_one_shared_session_for_two_page_workers():
    inpainter = Inpainter()
    inpainter._prefer_dynamic = False
    session = object()
    calls = []
    start = threading.Barrier(3)

    def make_fake_session(*_args, **_kwargs):
        calls.append(1)
        time.sleep(0.05)
        return session

    def worker():
        start.wait()
        inpainter.preload()

    with patch.object(lama_module, "make_session", side_effect=make_fake_session), patch.object(
        inpainter,
        "_configure_loaded_session",
        side_effect=_configure_fake_session(inpainter),
    ):
        threads = [threading.Thread(target=worker) for _ in range(2)]
        for thread in threads:
            thread.start()
        start.wait()
        for thread in threads:
            thread.join(timeout=2)

    assert calls == [1]
    assert inpainter.session is session
    status = inpainter.session_load_status()
    assert status["state"] == "ready"
    assert status["failed"] is False
    assert status["load_ms"] >= 50


def test_load_status_does_not_wait_for_session_construction():
    inpainter = Inpainter()
    inpainter._prefer_dynamic = False
    session = object()
    constructing = threading.Event()
    release = threading.Event()

    def make_fake_session(*_args, **_kwargs):
        constructing.set()
        assert release.wait(timeout=2)
        return session

    with patch.object(lama_module, "make_session", side_effect=make_fake_session), patch.object(
        inpainter,
        "_configure_loaded_session",
        side_effect=_configure_fake_session(inpainter),
    ):
        thread = threading.Thread(target=inpainter.preload)
        thread.start()
        assert constructing.wait(timeout=2)
        started = time.perf_counter()
        status = inpainter.session_load_status()
        elapsed = time.perf_counter() - started
        release.set()
        thread.join(timeout=2)

    assert status["state"] == "loading"
    assert elapsed < 0.1
    assert inpainter.session is session


def test_preload_failure_is_visible_and_can_be_retried():
    inpainter = Inpainter()
    inpainter._prefer_dynamic = False

    with patch.object(lama_module, "make_session", side_effect=RuntimeError("broken model")):
        with pytest.raises(RuntimeError, match="broken model"):
            inpainter.preload()

    failed = inpainter.session_load_status()
    assert failed["state"] == "failed"
    assert failed["failed"] is True
    assert failed["load_ms"] is not None

    session = object()
    with patch.object(lama_module, "make_session", return_value=session), patch.object(
        inpainter,
        "_configure_loaded_session",
        side_effect=_configure_fake_session(inpainter),
    ):
        inpainter.preload()

    ready = inpainter.session_load_status()
    assert ready["state"] == "ready"
    assert ready["failed"] is False
    assert inpainter.session is session


def test_health_is_degraded_when_inpaint_preload_failed():
    runtime = {
        "models": {
            "inpaint": {
                "load_failed": True,
            }
        }
    }
    with patch.object(main_module, "check_models", return_value=[]), patch.object(
        main_module,
        "_runtime_state",
        return_value=runtime,
    ):
        response = main_module.health()

    assert response["status"] == "degraded"
    assert response["models_missing"] == []
    assert response["runtime"] is runtime
