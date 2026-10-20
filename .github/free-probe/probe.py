"""One-off: can CommandCode's free models answer, read an image and call a tool."""
import base64, json, os, sys, time

import cv2, numpy as np, requests

BASE = "https://api.commandcode.ai/provider/v1/chat/completions"
KEY = os.environ["COMMANDCODE"]
image = np.full((120, 360, 3), 255, np.uint8)
cv2.putText(image, "GOBLIN 47", (14, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 4)
pic = "data:image/jpeg;base64," + base64.b64encode(cv2.imencode(".jpg", image)[1].tobytes()).decode()
TOOL = [{"type": "function", "function": {"name": "run_command", "description": "Run a shell command.",
         "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]
TESTS = {
    "text": {"messages": [{"role": "user", "content": "Dịch sang tiếng Việt tự nhiên, chỉ trả lời câu dịch: \"Screw it. Whatever. It says I'm going to die in four days.\""}]},
    "image": {"messages": [{"role": "user", "content": [{"type": "text", "text": "What exactly is written in this image? Answer with the text only."},
                                                      {"type": "image_url", "image_url": {"url": pic}}]}]},
    "tool": {"messages": [{"role": "user", "content": "List the files in /tmp using the tool."}], "tools": TOOL},
}
rows = []
for model in sys.argv[2].split():
    for name, body in TESTS.items():
        started = time.time()
        try:
            r = requests.post(BASE, headers={"Authorization": "Bearer " + KEY},
                              json={"model": model, "max_tokens": 4000, **body}, timeout=(10, 300))
            js = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
            msg = ((js.get("choices") or [{}])[0].get("message") or {})
            row = {"status": r.status_code, "answer": (msg.get("content") or "")[:400],
                   "tool_calls": [c.get("function") for c in msg.get("tool_calls") or []],
                   "usage": js.get("usage"), "error": "" if r.ok else r.text[:400]}
        except requests.RequestException as exc:
            row = {"status": 0, "error": str(exc)[:300]}
        row |= {"model": model, "test": name, "seconds": round(time.time() - started, 1)}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
os.makedirs(sys.argv[1], exist_ok=True)
json.dump(rows, open(os.path.join(sys.argv[1], "results.json"), "w"), ensure_ascii=False, indent=1)
