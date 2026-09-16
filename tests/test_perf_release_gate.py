from __future__ import annotations

from scripts.perf_release_gate import evaluate


def _phase():
    return {"status": "pass", "promotion_eligible": True, "gates": {"all": True}}


def test_release_gate_requires_every_phase_and_real_e2e():
    reports = {name: _phase() for name in ("E5", "E6", "E7", "E8", "E9", "E10")}
    result = evaluate(
        reports,
        {
            "speedup_pct": 16.0,
            "safety": "pass",
            "memory_gate": "pass",
        },
    )
    assert result["status"] == "pass"

    result = evaluate(reports, None)
    assert result["status"] == "blocked"
    assert any("E2E" in blocker for blocker in result["blockers"])


def test_release_gate_blocks_non_promotable_synthetic_phase():
    reports = {name: _phase() for name in ("E5", "E6", "E7", "E8", "E9", "E10")}
    reports["E5"]["promotion_eligible"] = False
    result = evaluate(
        reports,
        {"speedup_pct": 20, "safety": "pass", "memory_gate": "pass"},
    )
    assert result["status"] == "blocked"
    assert any("E5" in blocker for blocker in result["blockers"])
