"""One-off: measure the Agnes free tier with tiny requests in three phases (burst, 10 RPM for an hour, 20 RPM)."""
import json
import os
from pathlib import Path
import sys
import time

import requests

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
BASE, MODEL, KEY = os.environ["AGENT_BASE"], os.environ["AGENT_MODEL"], os.environ["AGENT_API_KEY"]
PHASES = [("A_burst", 40, 0.0), ("B_10rpm", 600, 6.0), ("C_20rpm", 200, 3.0)]
log = open(OUT / "requests.jsonl", "w", encoding="utf-8")


def one(phase: str, index: int) -> dict:
    body = {"model": MODEL, "max_tokens": 5, "messages": [{"role": "user", "content": "Reply with: ok"}]}
    start = time.time()
    try:
        reply = requests.post(f"{BASE}/chat/completions", json=body, headers={"Authorization": f"Bearer {KEY}"}, timeout=60)
        status, text = reply.status_code, reply.text[:300]
        headers = {k: v for k, v in reply.headers.items() if any(w in k.lower() for w in ("limit", "retry", "remaining", "reset"))}
        usage = reply.json().get("usage") if reply.ok else None
    except Exception as exc:
        status, text, headers, usage = 0, f"{type(exc).__name__}: {exc}"[:300], {}, None
    row = {"phase": phase, "i": index, "t": round(start, 2), "status": status, "latency_s": round(time.time() - start, 2),
           "headers": headers, "usage": usage, "error": "" if status == 200 else text}
    log.write(json.dumps(row, ensure_ascii=False) + "\n")
    log.flush()
    return row


summary = {}
for phase, count, gap in PHASES:
    rows, began = [], time.time()
    for index in range(count):
        tick = time.time()
        rows.append(one(phase, index))
        time.sleep(max(0.0, gap - (time.time() - tick)))
    minutes = (time.time() - began) / 60
    codes = {}
    for row in rows:
        codes[str(row["status"])] = codes.get(str(row["status"]), 0) + 1
    ok = [r for r in rows if r["status"] == 200]
    first_429 = next((r["i"] for r in rows if r["status"] == 429), None)
    summary[phase] = {"requests": count, "minutes": round(minutes, 1), "sent_rpm": round(count / minutes, 1), "ok": len(ok),
                      "ok_rpm": round(len(ok) / minutes, 1), "codes": codes, "first_429_at": first_429,
                      "median_latency_s": sorted(r["latency_s"] for r in rows)[len(rows) // 2],
                      "headers_seen": sorted({k for r in rows for k in r["headers"]}),
                      "errors": sorted({r["error"][:160] for r in rows if r["error"]})[:5]}
    print(phase, json.dumps(summary[phase], ensure_ascii=False), flush=True)
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    time.sleep(90)  # let any per-minute window clear before the next phase
