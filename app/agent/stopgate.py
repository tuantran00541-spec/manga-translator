"""The stop gate: when the agent stops, a small trained judge reads how the run ended and says whether the grader is likely to pass it."""
from __future__ import annotations

import json
import urllib.request

# The question the judge was trained on (.github/laya/build.py); it must not change without retraining.
QUESTION = {"verified": {"type": "noul", "instructions": (
    "The coding agent has stopped and says it is finished. Is the task really done and shown to work, "
    "so that the task's own grader will pass it?")}}


def tail(text, n):
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else "…" + text[-n:]


def state_of(rows: list[dict]) -> tuple[str, str]:
    """The final reply first, then the last steps, then the task, so truncation drops the least useful part."""
    task = next((r.get("text") for r in rows if r.get("type") == "user"), "")
    final = next(r for r in reversed(rows) if r.get("type") == "assistant")
    calls = {c.get("id"): c for r in rows if r.get("type") == "assistant" for c in r.get("calls") or []}
    steps = []
    for r in [r for r in rows if r.get("type") == "tool"][-3:]:
        args = (calls.get(r.get("id")) or {}).get("args") or {}
        what = args.get("command") or args.get("path") or args.get("name") or ""
        steps.append(f"{r.get('name')} {tail(what, 160)} -> {'ok' if r.get('ok', True) else 'failed'}: {tail(r.get('output'), 350)}")
    return (f"Final reply: {tail(final.get('text'), 900)}\n\nLast steps:\n" + "\n".join(steps) +
            f"\n\nTask: {tail(task, 600)}"), final.get("text") or ""


def score(url: str, state: str, timeout: float = 60) -> float:
    """The judge's probability that the run passes its grader."""
    request = urllib.request.Request(url, data=json.dumps({"state": state}).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return float(json.loads(response.read())["verified"])
