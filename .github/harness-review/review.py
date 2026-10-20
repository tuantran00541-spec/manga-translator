"""One-off: one long-context model reads the whole Chaos harness and proposes optimizations."""
import glob, json, os, sys, time

import requests

model, effort, out = sys.argv[1], sys.argv[2], sys.argv[3]
files = sorted(f for f in glob.glob("app/agent/**/*.py", recursive=True) if "/tests/" not in f)
files += [".github/terminal-bench/run_agent.py", "docs/chaos-open-design.md"]
code = "\n\n".join(f"===== {f} =====\n{open(f, encoding='utf-8').read()}" for f in files)
notes = open(".github/harness-review/notes.md", encoding="utf-8").read()
prompt = f"""You are reviewing "Chaos", an open-source coding-agent harness (Python), whose full source follows.
The owner wants the harness to stay thin and open (plugins, MCP, model-written tools), and wants it to score higher on
Terminal-Bench 2.1 across many models. Measured runs on this exact harness:
{notes}

Task: find the changes most likely to raise task success, ranked by expected gain per line of code.
For each proposal give: the problem with evidence (file and function, quote the line), why it costs tasks,
the concrete change (small diff or pseudo-diff), the risk, and how to measure it. Also list anything that is
dead, duplicated or overbuilt and could be removed. Be specific to this code; no generic agent advice.
Answer in English, in Markdown.

{code}"""
body = {"model": model, "max_tokens": 64000, "reasoning_effort": effort, "messages": [{"role": "user", "content": prompt}]}
started = time.time()
r = requests.post("https://api.commandcode.ai/provider/v1/chat/completions", json=body, timeout=(10, 3600),
                  headers={"Authorization": "Bearer " + os.environ["COMMANDCODE"]})
os.makedirs(out, exist_ok=True)
js = r.json() if r.ok else {}
msg = (js.get("choices") or [{}])[0].get("message") or {}
open(f"{out}/review.md", "w", encoding="utf-8").write(msg.get("content") or f"HTTP {r.status_code}\n{r.text[:2000]}")
json.dump({"model": model, "effort": effort, "status": r.status_code, "seconds": round(time.time() - started),
           "files": files, "prompt_chars": len(prompt), "usage": js.get("usage")}, open(f"{out}/meta.json", "w"), indent=1)
print(r.status_code, round(time.time() - started), js.get("usage"))
