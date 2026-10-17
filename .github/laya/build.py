"""Turn Terminal-Bench trial logs into Laya training rows: the moment the agent stops, labelled by whether the grader passed it."""
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.agent.stopgate import QUESTION, state_of  # noqa: E402

# A reply mid-run that sounds finished; in a run the grader failed, it is a stop that should have been sent back.
DONE_CLAIM = re.compile(r"\b(?:done|complete[d]?|finished|fixed|implemented|works|pass(?:es|ed)?|success(?:fully)?)\b|✅", re.I)
FAKE_PER_RUN = 2


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
        task = result.parent.name.rsplit("__", 1)[0]
        part = "eval" if int(hashlib.md5(task.encode()).hexdigest(), 16) % 5 == 0 else "train"
        trial = str(result.parent.relative_to(root))

        def add(upto, kind, passed, suffix=""):
            state, final = state_of(rows[:upto])
            split[part].append({"state": state, "questions": QUESTION, "expected": {"verified": passed},
                                "meta": {"task": task, "trial": trial + suffix, "kind": kind, "final": final[-2000:]}})

        said = [i for i, r in enumerate(rows) if r.get("type") == "assistant"]
        last = rows[said[-1]] if said else None
        # Only a real stop counts for eval: the last reply carries text and no tool calls.
        stopped = last and not last.get("calls") and str(last.get("text") or "").strip()
        if stopped:
            add(len(rows), "stop", reward >= 1.0)
        else:
            skipped["no_stop"] += 1
        if part == "eval" or not said:
            continue
        if not stopped:
            add(said[-1] + 1, "cut", reward >= 1.0)
        if reward < 1.0:
            fakes = [i for i in said[:-1] if DONE_CLAIM.search(str(rows[i].get("text") or ""))][-FAKE_PER_RUN:]
            for i in fakes:
                add(i + 1, "fake", False, f"#{i}")
    for part, rows in split.items():
        with open(out / f"{part}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    stats = {part: {"rows": len(rows), "passed": sum(r["expected"]["verified"] for r in rows),
                    "tasks": len({r["meta"]["task"] for r in rows}),
                    "kinds": {k: sum(r["meta"]["kind"] == k for r in rows) for k in ("stop", "cut", "fake")}} for part, rows in split.items()}
    stats["skipped"] = skipped
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
