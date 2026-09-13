from __future__ import annotations

import numpy as np

from app.inpaint.runtime_tuned_inpainter import (
    LamaTensorPool,
    _runtime_session_options,
)


class _FakeBinding:
    def __init__(self):
        self.inputs = {}
        self.output = None

    def bind_cpu_input(self, name, value):
        self.inputs[name] = value

    def bind_output(self, name, **kwargs):
        self.output = (name, kwargs)


class _FakeSession:
    def __init__(self):
        self.bindings = []

    def io_binding(self):
        binding = _FakeBinding()
        self.bindings.append(binding)
        return binding


def test_runtime_session_options_enable_lossless_cpu_knobs():
    opts = _runtime_session_options(
        threads=3,
        arena=True,
        mem_pattern=True,
        dynamic_block_base=4,
    )
    assert opts.enable_cpu_mem_arena is True
    assert opts.enable_mem_pattern is True
    assert opts.intra_op_num_threads == 3
    assert opts.get_session_config_entry("session.dynamic_block_base") == "4"


def test_tensor_pool_reuses_exact_shapes_and_evicts_lru():
    session = _FakeSession()
    pool = LamaTensorPool(capacity=2)

    a1, hit1 = pool.get(session, "image", "mask", "out", 16, 24)
    a2, hit2 = pool.get(session, "image", "mask", "out", 16, 24)
    assert hit1 is False
    assert hit2 is True
    assert a1 is a2

    pool.get(session, "image", "mask", "out", 24, 24)
    pool.get(session, "image", "mask", "out", 32, 24)
    _, hit_after_eviction = pool.get(session, "image", "mask", "out", 16, 24)
    assert hit_after_eviction is False


def test_tensor_pool_fill_matches_base_bgr_to_rgb_contract():
    session = _FakeSession()
    pool = LamaTensorPool(capacity=1)
    entry, _ = pool.get(session, "image", "mask", "out", 2, 2)
    canvas = np.array(
        [
            [[0, 64, 255], [255, 128, 0]],
            [[30, 60, 90], [12, 24, 48]],
        ],
        dtype=np.uint8,
    )
    mask = np.array([[0, 127], [128, 255]], dtype=np.uint8)

    pool.fill(entry, canvas, mask)

    expected = canvas[:, :, ::-1].astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    np.testing.assert_array_equal(entry.image, expected)
    np.testing.assert_array_equal(
        entry.mask,
        np.array([[[[0.0, 0.0], [1.0, 1.0]]]], dtype=np.float32),
    )
