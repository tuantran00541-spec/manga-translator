#!/usr/bin/env python3
"""Start one AI-mode chapter job on the local app server and wait for it.

POSTs /api/ai_mode/start, then polls /api/ai_mode/jobs/<id> until the job
completes, fails, or is cancelled. Exits 0 only on completion.
"""
import argparse
import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8000"


def _req(method, path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        BASE + path, data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--provider", default="nvidia")
    p.add_argument("--model", required=True)
    p.add_argument("--target-lang", default="vi")
    p.add_argument("--budget-usd", type=float, default=0.30)
    p.add_argument("--polish", action="store_true")
    p.add_argument("--images-per-request", type=int, default=2)
    p.add_argument("--translate-provider", default=None)
    p.add_argument("--translate-model", default=None)
    p.add_argument("--poll-seconds", type=int, default=60)
    p.add_argument("--max-minutes", type=int, default=280)
    args = p.parse_args()

    payload = {
        "url": args.url, "provider": args.provider, "model": args.model,
        "target_lang": args.target_lang, "budget_usd": args.budget_usd,
        "polish": args.polish, "images_per_request": args.images_per_request,
    }
    if args.translate_provider and args.translate_model:
        payload["translate_provider"] = args.translate_provider
        payload["translate_model"] = args.translate_model
    started = _req("POST", "/api/ai_mode/start", payload)
    job_id = started.get("job_id")
    if not job_id:
        print("start failed:", started, file=sys.stderr)
        return 1
    print("job:", job_id, flush=True)
    deadline = time.time() + args.max_minutes * 60
    while time.time() < deadline:
        job = _req("GET", f"/api/ai_mode/jobs/{job_id}")
        stages = " ".join(
            f"{s['key']}={s['done']}/{s['total']}" for s in job.get("stages", [])
        )
        print(f"[{time.strftime('%H:%M')}] {job.get('status')} {stages}", flush=True)
        if job.get("status") in ("completed", "failed", "cancelled"):
            r = job.get("report", {})
            print("translated:", r.get("translated"), "| unreadable:", r.get("unreadable"),
                  "| scan_errors:", len(r.get("scan_errors", [])),
                  "| translate_errors:", len(r.get("translate_errors", [])),
                  "| render_errors:", len(r.get("render_errors", [])))
            return 0 if job.get("status") == "completed" else 1
        time.sleep(args.poll_seconds)
    print("timed out waiting for the job", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
