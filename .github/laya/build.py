"""Turn Terminal-Bench trial logs into Laya training rows: the moment the agent stops, labelled by whether the grader passed it."""
import gzip
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.agent.stopgate import QUESTION, state_of  # noqa: E402


def events(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        return [json.loads(line) for line in f if line.startswith("{")]


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
