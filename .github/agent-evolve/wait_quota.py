"""One-off: wait until Agnes takes two requests in a row again, the sign its daily quota is back (when throttled it allows one a minute)."""
import os
import sys
import time

import requests

BASE, MODEL, KEY = os.environ["AGENT_BASE"], os.environ["AGENT_MODEL"], os.environ["AGENT_API_KEY"]
LIMIT_S = float(sys.argv[1]) * 60 if len(sys.argv) > 1 else 4 * 3600


def ask() -> tuple[int, str]:
    body = {"model": MODEL, "max_tokens": 5, "messages": [{"role": "user", "content": "Reply with: ok"}]}
    try:
        reply = requests.post(f"{BASE}/chat/completions", json=body, headers={"Authorization": f"Bearer {KEY}"}, timeout=60)
        return reply.status_code, reply.text[:200]
    except requests.RequestException as exc:
        return 0, type(exc).__name__


start = time.time()
while True:
    first, second = ask(), ask()
    print(time.strftime("%H:%M:%S"), first[0], second[0], second[1][:160] if second[0] != 200 else "", flush=True)
    if first[0] == 200 and second[0] == 200:
        print("quota is back", flush=True)
        break
    if time.time() - start > LIMIT_S:
        print("quota still throttled; sitting the exam anyway at the slow pace", flush=True)
        break
    time.sleep(600)
