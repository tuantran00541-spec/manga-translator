"""Runs the Chaos harness on one instruction without the page, for Terminal-Bench, and writes its events as JSON lines."""
import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

from app.agent import client, sandbox
from app.agent.session import AgentSessionManager
from app.agent.tools import Workspace
from app.ai_providers import PROVIDERS

NOBODY = "Nobody can answer during this run; choose what fits the task best and go on."


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("instruction")
    parser.add_argument("--base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--minutes", type=float, default=60)
    args = parser.parse_args()
    # The relay sits on the Docker host's private address, which the app's guard against private hosts would refuse.
    client.validate_url = lambda url: url
    provider = dataclasses.replace(PROVIDERS["openai"], id="bench", label="bench", api_base=args.base.rstrip("/"), builtin=False, tracks_cost=False)
    manager = AgentSessionManager(None, home=Path.home())
    # The task container is the sandbox, so the harness runs with full access and nobody approves calls.
    session = manager.create(provider, "relay", args.model, Workspace(args.workspace, sandbox.Policy("full-access", True)), "bypass")
    session.set_deadline(args.minutes * 60)
    session.send(args.instruction)
    seen, end = 0, time.time() + args.minutes * 60 + 120
    while time.time() < end:
        time.sleep(0.5)
        for event in session.events[seen:]:
            print(json.dumps(event, ensure_ascii=False, default=str), flush=True)
        seen = len(session.events)
        if session.status == "waiting" and session.pending:
            # A question gets the answer of a person away from the keyboard; anything else that still asks is refused.
            session.decide("allow" if session.pending.get("name") == "ask_user" else "deny", NOBODY)
        if session.status == "idle" and any(e["type"] == "done" for e in session.events[-3:]):
            break
    status = "done" if session.status == "idle" else "timeout"
    if status == "timeout":
        session.stop()
    print(json.dumps({"type": "result", "status": status, "usage": session.usage, "stats": session.stats}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
