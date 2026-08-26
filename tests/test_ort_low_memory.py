from __future__ import annotations

import threading
import time

import app.ort_utils as ort_utils


class _FakeSession:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def run(self, *args, **kwargs):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.03)
        with self._lock:
            self.active -= 1
        return [1]

    def get_inputs(self):
        return []


def test_low_memory_defaults_disable_ort_retention(monkeypatch):
    created = []

    def fake_ctor(*args, **kwargs):
        session = _FakeSession(*args, **kwargs)
        created.append(session)
        return session

    monkeypatch.delenv("MANGA_ORT_CPU_MEM_ARENA", raising=False)
    monkeypatch.delenv("MANGA_ORT_MEM_PATTERN", raising=False)
    monkeypatch.delenv("MANGA_ORT_SERIALIZE_INFERENCE", raising=False)
    monkeypatch.setattr(ort_utils.ort, "InferenceSession", fake_ctor)

    session = ort_utils.make_session("dummy.onnx", intra_op_threads=1)
    opts = created[0].kwargs["sess_options"]

    assert opts.enable_cpu_mem_arena is False
    assert opts.enable_mem_pattern is False
    assert isinstance(session, _FakeSession)


def test_environment_can_restore_ort_allocator_features(monkeypatch):
    created = []

    def fake_ctor(*args, **kwargs):
        session = _FakeSession(*args, **kwargs)
        created.append(session)
        return session

    monkeypatch.setenv("MANGA_ORT_CPU_MEM_ARENA", "1")
    monkeypatch.setenv("MANGA_ORT_MEM_PATTERN", "true")
    monkeypatch.setenv("MANGA_ORT_SERIALIZE_INFERENCE", "0")
    monkeypatch.setattr(ort_utils.ort, "InferenceSession", fake_ctor)

    session = ort_utils.make_session("dummy.onnx", intra_op_threads=1)
    opts = created[0].kwargs["sess_options"]

    assert opts.enable_cpu_mem_arena is True
    assert opts.enable_mem_pattern is True
    assert isinstance(session, _FakeSession)


def test_opt_in_serialization_prevents_overlapping_high_memory_runs(monkeypatch):
    created = []
    active = 0
    max_active = 0
    active_lock = threading.Lock()

    class FakeConcurrentSession(_FakeSession):
        def run(self, *args, **kwargs):
            nonlocal active, max_active
            with active_lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.05)
            with active_lock:
                active -= 1
            return [1]

    def fake_ctor(*args, **kwargs):
        session = FakeConcurrentSession(*args, **kwargs)
        created.append(session)
        return session

    monkeypatch.setattr(ort_utils.ort, "InferenceSession", fake_ctor)

    a = ort_utils.make_session("lama-a.onnx", intra_op_threads=1, serialize_inference=True)
    b = ort_utils.make_session("lama-b.onnx", intra_op_threads=1, serialize_inference=True)

    t1 = threading.Thread(target=a.run, args=(None, {}))
    t2 = threading.Thread(target=b.run, args=(None, {}))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert max_active == 1


def test_default_sessions_can_run_concurrently(monkeypatch):
    created = []
    active = 0
    max_active = 0
    active_lock = threading.Lock()

    class FakeConcurrentSession(_FakeSession):
        def run(self, *args, **kwargs):
            nonlocal active, max_active
            with active_lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.05)
            with active_lock:
                active -= 1
            return [1]

    def fake_ctor(*args, **kwargs):
        session = FakeConcurrentSession(*args, **kwargs)
        created.append(session)
        return session

    monkeypatch.delenv("MANGA_ORT_SERIALIZE_INFERENCE", raising=False)
    monkeypatch.setattr(ort_utils.ort, "InferenceSession", fake_ctor)

    a = ort_utils.make_session("det-a.onnx", intra_op_threads=1)
    b = ort_utils.make_session("det-b.onnx", intra_op_threads=1)

    t1 = threading.Thread(target=a.run, args=(None, {}))
    t2 = threading.Thread(target=b.run, args=(None, {}))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert max_active >= 2


def test_fixed_lama_recycles_session_before_pathological_run_count(monkeypatch):
    import numpy as np
    import app.inpaint.lama_inpainter as lama_mod

    created = []

    class _Input:
        def __init__(self, name, shape):
            self.name = name
            self.shape = shape

    class FakeLamaSession:
        def __init__(self):
            self.runs = 0

        def get_inputs(self):
            return [
                _Input("image", [1, 3, 512, 512]),
                _Input("mask", [1, 1, 512, 512]),
            ]

        def run(self, *_args, **_kwargs):
            self.runs += 1
            return [np.zeros((1, 3, 512, 512), dtype=np.float32)]

    def fake_make_session(*_args, **_kwargs):
        session = FakeLamaSession()
        created.append(session)
        return session

    monkeypatch.setenv("MANGA_USE_DYNAMIC_LAMA", "0")
    monkeypatch.setattr(lama_mod, "make_session", fake_make_session)

    inpainter = lama_mod.Inpainter()
    canvas = np.zeros((512, 512, 3), dtype=np.uint8)
    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[100:200, 100:200] = 255

    for _ in range(lama_mod.FIXED_LAMA_SESSION_MAX_RUNS + 1):
        inpainter._run_lama(canvas, mask)

    assert len(created) == 2
    assert created[0].runs == lama_mod.FIXED_LAMA_SESSION_MAX_RUNS
    assert created[1].runs == 1


def test_cpu_count_respects_cgroup_quota(monkeypatch):
    monkeypatch.setattr(ort_utils.os, "cpu_count", lambda: 64)
    real_open = open

    class _FakeCpuMax:
        def __enter__(self):
            return self
        def __exit__(self, *_args):
            return False
        def read(self):
            return "400000 100000\n"

    def fake_open(path, *args, **kwargs):
        if str(path) == "/sys/fs/cgroup/cpu.max":
            return _FakeCpuMax()
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", fake_open)
    assert ort_utils._cpu_count() == 4
    assert ort_utils._default_intra_op_threads() == 2
