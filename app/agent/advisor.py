"""A second model that reads the agent's recent steps and speaks up only when it sees a concrete risk (from omp's advisor)."""
from __future__ import annotations

import json
import re

PROMPT = """You shadow a coding agent for the user, who is not watching. You read its latest steps and advise it only when it matters.
Your lane: correctness, missed edge cases, a wrong code path, guessing what a quick check would show, thin verification, a premature "done", churn without progress, drift from what the user asked.
Stay silent when the agent is on track. Never restate what it already knows (errors it saw, failing tests it read), never repeat advice already given, never ask it to confirm scope or seek clarification, never police the size of a change the user asked for, never raise backwards compatibility unless the user required it.
Cite only what the transcript shows; arguments or outputs you cannot see are unknown. Steps marked in progress may be partial: only a blocker for damage happening now justifies interrupting them.
Severities: nit (fold in at the next step), concern (the agent may be heading wrong; it decides), blocker (stop and reconsider: data loss, the user's explicit instruction broken, a claim of done that the transcript contradicts).
Reply with one line of JSON: {"advice": [{"severity": "nit" or "concern" or "blocker", "text": "one or two terse sentences addressed to the agent"}]} with at most 3 items, or {"advice": []}."""
SEVERITIES = ("blocker", "concern", "nit")


def _key(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()[:200]


def advise(complete, provider, key: str, model: str, asked: list[str], steps: str, given: list[str], finishing: bool) -> tuple[list[tuple[str, str]], dict]:
    """([(severity, text)], usage) for the steps shown; nothing on any failure, since silence is always safe."""
    request = "\n".join(f"- {m[:1500]}" for m in asked[-6:]) or "- (none)"
    earlier = "\n".join(f"- {g}" for g in given[-12:]) or "- (none)"
    state = "The agent says it is finished; check the claim against the transcript." if finishing else "[in progress - more steps follow]"
    messages = [{"role": "system", "content": PROMPT},
                {"role": "user", "content": f"The user asked:\n{request}\n\nAdvice already given:\n{earlier}\n\nLatest steps {state}:\n{steps[-60000:]}"}]
    try:
        reply = complete(provider, key, model, messages, tools=None)
        text = reply["text"]
        start, end = text.find("{"), text.rfind("}")
        items = json.loads(text[start:end + 1]).get("advice") or []
    except Exception:
        return [], {}
    seen = {_key(g.split(": ", 1)[-1]) for g in given}
    notes = []
    for item in items[:3] if isinstance(items, list) else []:
        severity = str(item.get("severity") or "").lower() if isinstance(item, dict) else ""
        body = str(item.get("text") or "").strip()[:600] if isinstance(item, dict) else ""
        if severity in SEVERITIES and body and _key(body) not in seen:
            seen.add(_key(body))
            notes.append((severity, body))
    return notes, reply.get("usage") or {}
