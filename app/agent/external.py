"""Other coding agents as helpers: Codex, Claude Code or any command that takes a prompt, run as separate processes."""
from __future__ import annotations

import shutil
import subprocess
import time

from app.agent.tools import ToolError, clip

TIMEOUT = 900
# What is tried when the command is found on this machine; a profile's external_agents add or replace entries.
DEFAULTS = {"codex": ["codex", "exec", "--skip-git-repo-check", "{prompt}"], "claude": ["claude", "-p", "{prompt}"]}


def available(configured: dict) -> dict[str, list[str]]:
    found = {n: c for n, c in DEFAULTS.items() if shutil.which(c[0])}
    for name, row in configured.items():
        command = row.get("command") if isinstance(row, dict) else None
        if isinstance(command, list) and command and all(isinstance(p, str) for p in command) and shutil.which(command[0]):
            found[str(name)] = command
    return found


def spec(names: list[str]) -> dict:
    return {"name": "delegate", "description": "Give a standalone job to another coding agent running on this machine (it starts cold, "
                                               f"with only your text). Agents: {', '.join(names)}. Always asks the user first.",
            "parameters": {"type": "object", "required": ["agent", "prompt"], "properties": {
                "agent": {"type": "string"}, "prompt": {"type": "string"}}}}


def run(session, args: dict, commands: dict[str, list[str]]) -> str:
    command = commands.get(str(args.get("agent") or ""))
    prompt = str(args.get("prompt") or "").strip()
    if command is None:
        raise ToolError(f"No external agent named {args.get('agent')!r}; known: {', '.join(commands)}")
    if not prompt:
        raise ToolError("delegate needs a prompt")
    argv = [p.replace("{prompt}", prompt) for p in command]
    proc = subprocess.Popen(argv, cwd=session.workspace.root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace",
                            stdin=subprocess.DEVNULL)
    end = time.time() + TIMEOUT
    # Read in short steps (so a big output never blocks it), and the user's Stop ends the other agent too.
    while True:
        try:
            out, err = proc.communicate(timeout=0.5)
            break
        except subprocess.TimeoutExpired:
            if time.time() < end and not getattr(session, "_stop", False):
                continue
            proc.kill()
            proc.communicate()
            raise ToolError(f"{command[0]} was stopped" if getattr(session, "_stop", False) else f"{command[0]} did not finish in {TIMEOUT} s")
    return clip(f"{out.strip()}\n{err.strip()[-2000:]}\n[exit code {proc.returncode}]".strip())
