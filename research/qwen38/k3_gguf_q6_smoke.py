#!/usr/bin/env python3
"""Qwen3.8-27B Q6_K_L GGUF CPU smoke for the isolated K3 lab lane.

Downloads the community Q6_K_L artifact directly (no BF16 checkpoint), runs a
short llama.cpp-backed text generation, and records measured wall time/RSS plus
basic artifact identity. This is a transition gate: it proves the compressed
artifact itself is usable before teaching the custom K3 streamer GGUF blocks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import subprocess
import time
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "bartowski/Qwen3.8-27B-GGUF"
FILE = "Qwen3.8-27B-Q6_K_L.gguf"
EXPECTED_SHA256 = "a487690b9f17de581857c4ae484dab50800335bb9eb978a4fb02c0465629dc0a"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / (1024 * 1024)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--llama-cli", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    model = Path(hf_hub_download(REPO, filename=FILE, local_dir=str(args.work_dir)))
    digest = sha256(model)
    if digest != EXPECTED_SHA256:
        raise RuntimeError(f"GGUF sha256 mismatch: {digest}")
    cmd = [str(args.llama_cli), "-m", str(model), "-ngl", "0", "-c", "256", "-n", "8", "--temp", "0", "-p", "Answer with exactly one English word: Two plus two equals"]
    infer_started = time.monotonic()
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=1800)
    infer_seconds = time.monotonic() - infer_started
    text = proc.stdout
    semantic = "Four" in text or "four" in text
    result = {
        "schema": "qwen38-k3-gguf-q6-smoke-v1",
        "status": "PASS" if proc.returncode == 0 and semantic else "FAIL",
        "repo": REPO,
        "file": FILE,
        "sha256": digest,
        "file_bytes": model.stat().st_size,
        "returncode": proc.returncode,
        "semantic_four": semantic,
        "inference_seconds": infer_seconds,
        "max_child_rss_gib": rss_gib(),
        "total_seconds": time.monotonic() - started,
        "stdout_tail": text[-4000:],
        "bf16_downloaded": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
