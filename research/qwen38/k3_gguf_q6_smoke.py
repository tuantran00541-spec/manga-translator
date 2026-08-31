#!/usr/bin/env python3
"""Qwen3.8-27B Q6_K_L GGUF CPU first-token smoke for the isolated K3 lab lane.

This gate intentionally downloads only the community Q6_K_L GGUF (never the
BF16 checkpoint), verifies the pinned artifact identity, then asks llama.cpp
for exactly one greedy token. Evidence is checkpointed and flushed before each
expensive phase so an infrastructure shutdown still leaves useful job-log
breadcrumbs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import signal
import subprocess
import time
from collections import deque
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

REPO = "bartowski/Qwen3.8-27B-GGUF"
FILE = "Qwen3.8-27B-Q6_K_L.gguf"
EXPECTED_SHA256 = "a487690b9f17de581857c4ae484dab50800335bb9eb978a4fb02c0465629dc0a"
PROMPT = "Answer with exactly one English word and no explanation: Two plus two equals what?"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / (1024 * 1024)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def emit(event: str, **payload: Any) -> None:
    record = {"event": event, **payload}
    print("QWEN38_EVIDENCE " + json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--llama-cli", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    state: dict[str, Any] = {
        "schema": "qwen38-k3-gguf-q6-first-token-v2",
        "status": "INCOMPLETE",
        "phase": "start",
        "repo": REPO,
        "file": FILE,
        "expected_sha256": EXPECTED_SHA256,
        "prompt": PROMPT,
        "requested_tokens": 1,
        "context": 64,
        "temperature": 0,
        "bf16_downloaded": False,
    }

    def checkpoint(phase: str, **updates: Any) -> None:
        state.update(updates)
        state["phase"] = phase
        state["elapsed_seconds"] = time.monotonic() - started
        atomic_json(args.output, state)
        emit(phase, **updates)

    child: subprocess.Popen[str] | None = None

    def on_signal(signum: int, _frame: Any) -> None:
        checkpoint(
            "terminated_by_signal",
            status="INFRA_INTERRUPTED",
            signal=signum,
            max_child_rss_gib=rss_gib(),
        )
        if child is not None and child.poll() is None:
            child.terminate()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    try:
        checkpoint("llama_version_started")
        version_proc = subprocess.run(
            [str(args.llama_cli), "--version"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
            check=False,
        )
        version_text = version_proc.stdout.strip()
        checkpoint(
            "llama_version_completed",
            llama_version_returncode=version_proc.returncode,
            llama_version=version_text[-1000:],
        )
        if version_proc.returncode != 0:
            raise RuntimeError(f"llama-cli --version failed: {version_proc.returncode}")

        checkpoint("download_started")
        download_started = time.monotonic()
        model = Path(hf_hub_download(REPO, filename=FILE, local_dir=str(args.work_dir)))
        checkpoint(
            "download_completed",
            model_path=str(model),
            file_bytes=model.stat().st_size,
            download_seconds=time.monotonic() - download_started,
        )

        checkpoint("sha256_started")
        sha_started = time.monotonic()
        digest = sha256(model)
        sha_seconds = time.monotonic() - sha_started
        if digest != EXPECTED_SHA256:
            checkpoint(
                "sha256_mismatch",
                status="FAIL",
                sha256=digest,
                sha256_seconds=sha_seconds,
            )
            raise RuntimeError(f"GGUF sha256 mismatch: {digest}")
        checkpoint(
            "sha256_verified",
            sha256=digest,
            sha256_seconds=sha_seconds,
        )

        cmd = [
            str(args.llama_cli),
            "-m",
            str(model),
            "-ngl",
            "0",
            "-c",
            "64",
            "-n",
            "1",
            "--temp",
            "0",
            "-p",
            PROMPT,
        ]
        checkpoint("inference_started", command=cmd)
        infer_started = time.monotonic()
        child = subprocess.Popen(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        if child.stdout is None:
            raise RuntimeError("llama-cli stdout pipe unavailable")

        tail: deque[str] = deque(maxlen=256)
        first_output_seen = False
        metadata_seen = False
        load_signal_seen = False
        for line in child.stdout:
            print(line, end="", flush=True)
            tail.append(line)
            low = line.lower()
            if not first_output_seen:
                first_output_seen = True
                checkpoint("llama_first_output", first_output_line=line.strip()[:1000])
            if not metadata_seen and ("loaded meta data" in low or "model loader" in low):
                metadata_seen = True
                checkpoint("llama_metadata_seen", marker=line.strip()[:1000])
            if not load_signal_seen and (
                "load_tensors" in low
                or "llama_model_load" in low
                or "model load" in low
                or "loading model" in low
            ):
                load_signal_seen = True
                checkpoint("llama_load_progress_seen", marker=line.strip()[:1000])

        returncode = child.wait()
        infer_seconds = time.monotonic() - infer_started
        text = "".join(tail)
        semantic = "four" in text.lower()
        status = "PASS" if returncode == 0 and semantic else "FAIL"
        checkpoint(
            "inference_completed",
            status=status,
            returncode=returncode,
            semantic_four=semantic,
            inference_seconds=infer_seconds,
            max_child_rss_gib=rss_gib(),
            total_seconds=time.monotonic() - started,
            stdout_tail=text[-8000:],
        )
        if status != "PASS":
            raise SystemExit(1)
    except BaseException as exc:
        if state.get("status") == "INCOMPLETE":
            checkpoint(
                "exception",
                status="FAIL",
                error_type=type(exc).__name__,
                error=str(exc),
                max_child_rss_gib=rss_gib(),
            )
        raise


if __name__ == "__main__":
    main()
