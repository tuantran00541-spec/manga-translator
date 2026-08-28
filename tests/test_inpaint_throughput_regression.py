from __future__ import annotations

import app.ort_utils as ort_utils


def test_high_cpu_default_caps_each_session_at_four_threads(monkeypatch):
    monkeypatch.setattr(ort_utils, "_cpu_count", lambda: 16)
    assert ort_utils._default_intra_op_threads() == 4
