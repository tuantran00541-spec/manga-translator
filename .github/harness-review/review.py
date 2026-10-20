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
body = {"model": model, "max_tokens": 64000, "reasoning_effort": effort, "stream": True,
        "messages": [{"role": "user", "content": prompt}]}
started = time.time()
text, usage, status, error = "", None, 0, ""
# Streaming keeps the connection alive through a long think, which a plain request loses to a 524.
for attempt in range(3):
    try:
        with requests.post("https://api.commandcode.ai/provider/v1/chat/completions", json=body, stream=True, timeout=(10, 600),
                           headers={"Authorization": "Bearer " + os.environ["COMMANDCODE"]}) as r:
            status = r.status_code
            if not r.ok:
                error = r.text[:2000]
                continue
            for line in r.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:") or line.strip() == "data: [DONE]":
                    continue
                chunk = json.loads(line[5:])
                usage = chunk.get("usage") or usage
                for choice in chunk.get("choices") or []:
                    text += (choice.get("delta") or {}).get("content") or ""
        if text:
            break
    except (requests.RequestException, ValueError) as exc:
        error = str(exc)[:2000]
os.makedirs(out, exist_ok=True)
open(f"{out}/review.md", "w", encoding="utf-8").write(text or f"HTTP {status}\n{error}")
json.dump({"model": model, "effort": effort, "status": status, "seconds": round(time.time() - started),
           "files": files, "prompt_chars": len(prompt), "usage": usage, "error": error[:500]}, open(f"{out}/meta.json", "w"), indent=1)
print(status, round(time.time() - started), usage, len(text))
