"""Run our agent on the four chain-eval tasks for each arm as the agent user, check each workspace and summarise how the tools were used."""
import collections
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARMS = {"chain": {}, "shell": {"shell_tool": True, "min_calls": 1}}
WORK = Path("/opt/ce-work")


def run_one(arm, task, base, model, out, minutes):
    """One agent session in a fresh copy of the task, then the hidden check."""
    workspace, home = WORK / arm / task / "ws", WORK / arm / task / "home"
    shutil.copytree(HERE / "tasks" / task / "start", workspace)
    (home / ".manga-agent").mkdir(parents=True)
    profile = {"token_budget": 10 ** 9, "max_steps": 300, "max_output_tokens": 16000, **ARMS[arm]}
    (home / ".manga-agent" / "profile.json").write_text(json.dumps(profile))
    subprocess.run(["sudo", "chown", "-R", "agent", str(WORK / arm / task)], check=True)
    prompt = (HERE / "tasks" / task / "prompt.txt").read_text()
    events = out / f"{arm}-{task}.jsonl"
    started = time.time()
    with open(events, "w") as sink, open(out / f"{arm}-{task}.stderr", "w") as err:
        subprocess.run(["sudo", "-u", "agent", "env", f"HOME={home}", "PYTHONPATH=/opt/chaos", "AGENT_API_KEY=relay",
                        "/opt/chaos-venv/bin/python", "-m", "app.agent.cli", "exec", prompt, "--base", base, "--model", model,
                        "--provider", "bench", "--allow-private-base", "--workspace", str(workspace), "--mode", "bypass",
                        "--sandbox", "full-access", "--timeout-min", str(minutes), "--json"],
                       stdout=sink, stderr=err, stdin=subprocess.DEVNULL, timeout=minutes * 60 + 300, cwd=workspace)
    seconds = time.time() - started
    subprocess.run(["sudo", "chown", "-R", os.environ["USER"], str(workspace)], check=True)
    check = subprocess.run([sys.executable, str(HERE / "tasks" / task / "check.py"), str(workspace)], capture_output=True, text=True, timeout=300)
    return {"arm": arm, "task": task, "passed": check.returncode == 0, "check": (check.stdout + check.stderr)[-400:],
            "seconds": round(seconds), **usage(events)}


def usage(path):
    """Replies, calls per reply, tool names, refusals and the tool calls written into create code, from the event stream."""
    tools, in_code, per_reply, refused, chained, result = collections.Counter(), collections.Counter(), [], 0, 0, {}
    for line in open(path, encoding="utf-8", errors="replace"):
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("type") == "assistant" and row.get("calls"):
            per_reply.append(len(row["calls"]))
            # Calls made from create code never show as events, so they are counted from the code itself.
            for call in row["calls"]:
                if call.get("name") == "create":
                    inner = re.findall(r"\btools\.(\w+)\s*\(", str((call.get("args") or {}).get("code") or ""))
                    chained += len(inner)
                    in_code.update(inner)
        elif row.get("type") == "tool":
            tools[row.get("name")] += 1
            refused += str(row.get("output", "")).startswith("Refused: every reply")
        elif row.get("type") == "result":
            result = row
    return {"replies_with_calls": len(per_reply), "calls_per_reply": round(sum(per_reply) / max(1, len(per_reply)), 2),
            "tools": dict(tools.most_common()), "refused": refused, "chained_calls": chained,
            "tools_in_create_code": dict(in_code.most_common(8)),
            "status": result.get("status"), "usage": result.get("usage"), "stats": result.get("stats")}


def main(base, model, out, minutes="15", workers="2"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tasks = sorted(p.name for p in (HERE / "tasks").iterdir() if (p / "start").is_dir())
    jobs = [(arm, task) for task in tasks for arm in ARMS]
    with ThreadPoolExecutor(int(workers)) as pool:
        rows = list(pool.map(lambda job: run_one(*job, base, model, out, int(minutes)), jobs))
    summary = {arm: {"passed": sum(r["passed"] for r in rows if r["arm"] == arm), "of": len(tasks),
                     "refused": sum(r["refused"] for r in rows if r["arm"] == arm),
                     "chained_calls": sum(r["chained_calls"] for r in rows if r["arm"] == arm)} for arm in ARMS}
    (out / "summary.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
