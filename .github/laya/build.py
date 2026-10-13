"""Turn Terminal-Bench trial logs into Laya training rows: the moment the agent stops, labelled by whether the grader passed it."""
import gzip
import hashlib
import json
from pathlib import Path
import sys

QUESTION = {"verified": {"type": "noul", "instructions": (
    "The coding agent has stopped and says it is finished. Is the task really done and shown to work, "
    "so that the task's own grader will pass it?")}}


def events(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        return [json.loads(line) for line in f if line.startswith("{")]


def tail(text, n):
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else "…" + text[-n:]


def state_of(rows):
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


def main(root, out):
    out.mkdir(parents=True, exist_ok=True)
    split = {"train": [], "eval": []}
    skipped = {"no_result": 0, "no_stop": 0}
    for result in sorted(root.rglob("result.json")):
        log = next(iter(sorted(result.parent.glob("events.jsonl*"))), None)
        try:
            reward = json.loads(result.read_text())["verifier_result"]["rewards"]["reward"]
        except (KeyError, TypeError, ValueError):
            reward = None
        if log is None or reward is None:
            skipped["no_result"] += 1
            continue
        rows = events(log)
        last = next((r for r in reversed(rows) if r.get("type") == "assistant"), None)
        # Only a real stop counts: the last reply carries text and no tool calls.
        if not last or last.get("calls") or not str(last.get("text") or "").strip():
            skipped["no_stop"] += 1
            continue
        state, final = state_of(rows)
        task = result.parent.name.rsplit("__", 1)[0]
        part = "eval" if int(hashlib.md5(task.encode()).hexdigest(), 16) % 5 == 0 else "train"
        split[part].append({"state": state, "questions": QUESTION, "expected": {"verified": reward >= 1.0},
                            "meta": {"task": task, "trial": str(result.parent.relative_to(root)), "final": final[-2000:]}})
    for part, rows in split.items():
        with open(out / f"{part}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    stats = {part: {"rows": len(rows), "passed": sum(r["expected"]["verified"] for r in rows),
                    "tasks": len({r["meta"]["task"] for r in rows})} for part, rows in split.items()}
    stats["skipped"] = skipped
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
