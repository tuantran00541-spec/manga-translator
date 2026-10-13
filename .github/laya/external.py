"""Turn public SWE agent trajectories with grader verdicts into Laya stop rows, in the same state format as our own runs."""
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.agent.stopgate import QUESTION, state_of  # noqa: E402

SUBMIT_NAMES = {"submit", "finish", "task_complete", "attempt_completion"}
ISSUE = [re.compile(p, re.S) for p in (r"<issue_description>(.*?)</issue_description>", r"<pr_description>(.*?)</pr_description>",
                                        r"--- BEGIN ISSUE ---(.*?)--- END ISSUE ---")]
EXIT = re.compile(r"<returncode>(-?\d+)</returncode>|\"returncode\":\s*(-?\d+)|exit code (-?\d+)", re.I)
BLOCK = re.compile(r"```[a-z_]*\n(.*?)```", re.S)


def _task(text):
    for p in ISSUE:
        m = p.search(text or "")
        if m:
            return m.group(1).strip()
    return text or ""


def _ok(output):
    m = EXIT.search(output or "")
    return m is None or int(next(g for g in m.groups() if g is not None)) == 0


def _calls(msg, n):
    calls = []
    for i, c in enumerate(msg.get("tool_calls") or []):
        fn = c.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except ValueError:
            args = {"command": fn.get("arguments")}
        calls.append({"id": c.get("id") or f"c{n}-{i}", "name": fn.get("name") or "tool", "args": args if isinstance(args, dict) else {}})
    if not calls:
        m = BLOCK.search(msg.get("content") or "")
        if m:
            calls.append({"id": f"c{n}", "name": "bash", "args": {"command": m.group(1).strip()}})
    return calls


def _submits(call):
    return call["name"] in SUBMIT_NAMES or "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in str(call["args"].get("command") or "")


def to_rows(messages):
    """Our event rows up to the agent's submit, which becomes the final reply; None when the run never submitted."""
    rows, last, seen_task = [], [], False
    for n, msg in enumerate(messages):
        role, content = msg.get("role"), msg.get("content") or ""
        if role == "system":
            continue
        if role == "user" and not seen_task:
            rows.append({"type": "user", "text": _task(content)})
            seen_task = True
        elif role == "assistant":
            calls = _calls(msg, n)
            done = [c for c in calls if _submits(c)]
            if done:
                note = str(done[0]["args"].get("message") or done[0]["args"].get("result") or "")
                text = BLOCK.sub("", content).removeprefix("THOUGHT:").strip()
                rows.append({"type": "assistant", "text": (text + "\n" + note).strip(), "calls": []})
                return rows if rows[-1]["text"] else None
            rows.append({"type": "assistant", "text": content, "calls": calls})
            last = calls
        elif role in ("tool", "user"):
            call = next((c for c in last if c["id"] == msg.get("tool_call_id")), last[0] if last else {"id": "", "name": "tool", "args": {}})
            out = content.removeprefix("OBSERVATION:").strip()
            rows.append({"type": "tool", "id": call["id"], "name": call["name"], "ok": _ok(out), "output": out})
    return None


def row(source, task, messages, passed):
    rows = to_rows(messages)
    if not rows or not any(r["type"] == "tool" for r in rows):
        return None
    state, final = state_of(rows)
    return {"state": state, "questions": QUESTION, "expected": {"verified": bool(passed)},
            "meta": {"task": f"{source}:{task}", "source": source, "final": final[-2000:]}}


def _rank(item):
    return hashlib.md5(json.dumps(item["meta"]["task"] + item["state"][:200]).encode()).hexdigest()


def pick(items, quota, per_task):
    """Half passed and half failed where possible, at most per_task rows per task, in a fixed pseudo-random order."""
    items.sort(key=_rank)
    out, count = [], {}
    for want in (True, False):
        start = len(out)
        for i in (i for i in items if i["expected"]["verified"] is want):
            if len(out) - start >= quota // 2:
                break
            if count.get(i["meta"]["task"], 0) < per_task:
                count[i["meta"]["task"]] = count.get(i["meta"]["task"], 0) + 1
                out.append(i)
    return out


def tarsur(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for r in data if isinstance(data, list) else data.get("data", []):
        yield row("tarsur", r["task_id"], r["messages"], (r.get("reward") or 0) >= 1)


def parquet(path, source, messages, task, label):
    import pyarrow.parquet as pq
    for batch in pq.ParquetFile(path).iter_batches(batch_size=64, columns=[task, messages, label] + (["exit_status"] if source == "nebius" else [])):
        for r in batch.to_pylist():
            if r[label] not in (0, 1) or (source == "nebius" and r.get("exit_status") != "submit"):
                continue
            yield row(source, r[task], r[messages], r[label] == 1)


def main(out, shards, quota, *files):
    """files are source=path; writes shard-<k>.jsonl, each mixing every source."""
    picked = []
    for spec in files:
        source, path = spec.split("=", 1)
        if source == "tarsur":
            items = [i for p in path.split(",") for i in tarsur(p) if i]
        elif source == "nebius":
            items = [i for i in parquet(path, "nebius", "trajectory", "instance_id", "resolved") if i]
        else:
            items = [i for p in path.split(",") for i in parquet(p, source, "messages", "instance_id", "resolved") if i]
        share = int(quota[source]) if source in quota else int(quota["*"])
        chosen = pick(items, share, 12 if source == "tarsur" else 3)
        print(json.dumps({"source": source, "stops": len(items), "passed": sum(i["expected"]["verified"] for i in items),
                          "picked": len(chosen)}), flush=True)
        picked += chosen
    picked.sort(key=_rank)
    out.mkdir(parents=True, exist_ok=True)
    for k in range(shards):
        with open(out / f"shard-{k}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in picked[k::shards])


if __name__ == "__main__":
    quota = dict(q.split(":") for q in sys.argv[3].split(","))
    main(Path(sys.argv[1]), int(sys.argv[2]), quota, *sys.argv[4:])
