from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Plan:
    id: str
    label: str
    chapters_per_month: int
    features: frozenset[str]


PLANS: dict[str, Plan] = {
    "free": Plan("free", "Free", 3, frozenset()),
    "plus": Plan("plus", "Plus", 30, frozenset({"visual_qc"})),
    "pro": Plan("pro", "Pro", 100, frozenset({"visual_qc", "byok", "custom_providers"})),
}


def get_plan(plan_id: str) -> Plan:
    plan = PLANS.get(plan_id)
    if plan is None:
        raise ValueError(f"Unknown plan: {plan_id}")
    return plan
