"""Counts what the harness features a probe run used, from its events, and checks the result."""
import collections
import json
import re
import sys

events = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.startswith("{")]
calls = [c for e in events if e.get("type") == "assistant" for c in e.get("calls") or []]
names = collections.Counter(c["name"] for c in calls)
commands = [str((c.get("args") or {}).get("command") or "") for c in calls if c["name"] == "run_command"]
codes = [str((c.get("args") or {}).get("code") or "") for c in calls if c["name"] == "create"]
bridge = collections.Counter(m for cmd in commands for m in re.findall(r"agent-tool\s+(\w+)", cmd))
python_bridge = collections.Counter(m for text in commands + codes for m in re.findall(r"tools\.(\w+)\(", text))
notices = [e.get("text", "") for e in events if e.get("type") == "notice"]
result = next((e for e in events if e.get("type") == "result"), {})
print(json.dumps({
    "tool_calls": dict(names.most_common()),
    "agent_tool_from_shell": dict(bridge),
    "tools_dot_calls": dict(python_bridge),
    "skills_loaded": [(c.get("args") or {}).get("name") for c in calls if c["name"] == "skill"],
    "background_commands": sum(1 for c in calls if c["name"] == "run_command" and (c.get("args") or {}).get("background")),
    "check_nudges": sum("without a passing check" in n for n in notices),
    "notices": notices[:40],
    "status": result.get("status"), "usage": result.get("usage"),
    "final_text": next((e.get("text", "") for e in reversed(events) if e.get("type") == "assistant" and e.get("text")), "")[-3000:],
}, ensure_ascii=False, indent=1))
