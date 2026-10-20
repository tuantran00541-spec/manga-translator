"""One-off: Beam reads one area of the manga app, reports bugs with evidence and writes pytest tests for the fragile parts."""
import ast, glob, json, os, sys, time

import requests

area, model, effort, out = sys.argv[1:5]
spec = json.load(open(".github/code-review/areas.json", encoding="utf-8"))[area]
files = sorted({f for pattern in spec["files"] for f in glob.glob(pattern, recursive=True) if "/tests/" not in f})
code = "\n\n".join(f"===== {f} =====\n{open(f, encoding='utf-8').read()}" for f in files)
# Existing test names only, so the new tests do not repeat them.
known = []
for f in sorted(glob.glob("tests/test_*.py")):
    names = [n.name for n in ast.walk(ast.parse(open(f, encoding="utf-8").read())) if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")]
    known.append(f"{f}: {', '.join(names)}")
prompt = f"""You are reviewing part of "Manga Translator", a FastAPI app that downloads manga/manhwa chapters, cleans the text off
the images and letters a Vietnamese translation back in. This part is {spec['what']}. Its full source follows.

Do two things, in English Markdown.

## Part 1: bugs
Find real defects: wrong results, crashes, data loss, races, off-by-one at slice seams, wrong coordinates, silent failures.
For each: file and function, the exact line(s) quoted, a concrete input or state that triggers it, what goes wrong, and the
smallest fix. Mark each "certain" (provable from the code alone) or "suspected" (depends on data or behaviour you cannot see).
Rank by impact. Do not report style, naming or missing docs. Skip anything you cannot tie to a quoted line.

## Part 2: tests
Write pytest tests for the most fragile logic in this code (geometry, merging, seams, fitting, state transitions, parsing),
including one test per "certain" bug above that fails on the current code. Rules: import only from the code shown; no network,
no model files (.onnx), no GPU; build inputs with numpy/PIL in the test; each test runs in under 2 seconds; mark a test
that should fail today with a comment "# fails today: <bug>". Give each file as one ```python block headed by its path,
tests/test_beam_<topic>.py. Do not repeat these existing tests:
{chr(10).join(known)}

{code}"""
body = {"model": model, "max_tokens": 64000, "reasoning_effort": effort, "stream": True, "messages": [{"role": "user", "content": prompt}]}
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
json.dump({"area": area, "model": model, "effort": effort, "status": status, "seconds": round(time.time() - started),
           "files": files, "prompt_chars": len(prompt), "usage": usage, "error": error[:500]}, open(f"{out}/meta.json", "w"), indent=1)
print(status, round(time.time() - started), usage, len(text), len(prompt))
