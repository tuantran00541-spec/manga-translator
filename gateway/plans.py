from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Plan:
    id: str
    label: str
    chapters_per_month: int
    features: frozenset[str]
    # Most one chapter may spend; a long webtoon chapter costs about $0.25, so this only stops scripted use.
    job_cost_guard_usd: float


PLANS: dict[str, Plan] = {
    "free": Plan("free", "Free", 3, frozenset(), 0.75),
    "plus": Plan("plus", "Plus", 30, frozenset({"visual_qc"}), 2.0),
    "pro": Plan("pro", "Pro", 100, frozenset({"visual_qc", "byok", "custom_providers"}), 2.0),
}


def get_plan(plan_id: str) -> Plan:
    plan = PLANS.get(plan_id)
    if plan is None:
        raise ValueError(f"Unknown plan: {plan_id}")
    return plan
