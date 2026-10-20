"""One-off: AIME 2025 and 2026 through one CommandCode model, several samples per problem."""
import json, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor

import requests
from datasets import load_dataset

model, out, samples, effort = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
problems = []
for year in ("2025", "2026"):
    try:
        for row in load_dataset(f"MathArena/aime_{year}", split="train"):
            problems.append({"id": f"{year}-{row.get('problem_idx')}", "problem": row["problem"], "answer": str(row["answer"]).strip()})
    except Exception as exc:  # a year that is not published is skipped
        print(f"skip {year}: {exc}", flush=True)
print(len(problems), "problems", flush=True)


def boxed(text):
    found = re.findall(r"\\boxed\{([^{}]*)\}", text or "")
    return re.sub(r"[^0-9]", "", found[-1]).lstrip("0") or "0" if found else None


def solve(job):
    p, k = job
    body = {"model": model, "max_tokens": 64000, "reasoning_effort": effort, "messages": [{"role": "user", "content":
            p["problem"] + "\n\nSolve the problem. Put only the final integer answer in \\boxed{}."}]}
    started = time.time()
    for attempt in range(3):
        try:
            r = requests.post("https://api.commandcode.ai/provider/v1/chat/completions", json=body, timeout=(10, 1800),
                              headers={"Authorization": "Bearer " + os.environ["COMMANDCODE"]})
            if r.status_code == 200:
                js = r.json()
                text = (js["choices"][0]["message"].get("content") or "")
                got = boxed(text)
                return {"id": p["id"], "sample": k, "answer": p["answer"].lstrip("0") or "0", "got": got,
                        "correct": got == (p["answer"].lstrip("0") or "0"), "seconds": round(time.time() - started),
                        "usage": js.get("usage"), "tail": text[-300:]}
            error = f"HTTP {r.status_code} {r.text[:200]}"
        except requests.RequestException as exc:
            error = str(exc)[:200]
        time.sleep(20 * (attempt + 1))
    return {"id": p["id"], "sample": k, "answer": p["answer"], "got": None, "correct": False, "error": error,
            "seconds": round(time.time() - started)}


rows = []
os.makedirs(out, exist_ok=True)
log = open(f"{out}/rows.jsonl", "w")
with ThreadPoolExecutor(8) as pool:
    for row in pool.map(solve, [(p, k) for p in problems for k in range(samples)]):
        rows.append(row)
        log.write(json.dumps(row) + "\n"); log.flush()
        print(row["id"], row["sample"], row["got"], row["answer"], row["correct"], row["seconds"], row.get("error", ""), flush=True)
os.makedirs(out, exist_ok=True)
acc = sum(r["correct"] for r in rows) / len(rows) if rows else 0
json.dump({"model": model, "effort": effort, "problems": len(problems), "samples": samples, "accuracy": round(acc, 4),
           "errors": sum(1 for r in rows if r.get("error")), "rows": rows}, open(f"{out}/results.json", "w"), indent=1)
print(f"ACCURACY {model}: {acc:.3f}", flush=True)
