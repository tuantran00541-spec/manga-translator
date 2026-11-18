#!/usr/bin/env python3
import argparse
import base64
import io
import json
import re
import sys
import urllib.request

LLAMA = "http://127.0.0.1:8080/v1/chat/completions"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"}


def _get(url: str, timeout: int = 120) -> bytes:
    import requests

    last: Exception | None = None
    for _ in range(3):
        try:
            resp = requests.get(url, headers=UA, timeout=timeout)
            resp.raise_for_status()
            return resp.content
        except Exception as exc:
            last = exc
    import subprocess

    proc = subprocess.run(
        ["curl", "-fsSL", "-A", UA["User-Agent"], "--max-time", str(timeout), url],
        capture_output=True, timeout=timeout + 30,
    )
    if proc.returncode == 0 and proc.stdout:
        return proc.stdout
    raise RuntimeError(f"GET failed: {url}: {last}")


def chapter_images(url: str) -> list[str]:
    html = _get(url).decode("utf-8", "replace")
    match = re.search(r"ts_reader\.run\((\{.*?\})\);", html, re.S)
    if not match:
        raise RuntimeError("ts_reader payload not found on chapter page")
    data = json.loads(match.group(1).replace("\\/", "/"))
    images = data["sources"][0]["images"]
    if not images:
        raise RuntimeError("chapter page lists zero images")
    return images


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--model", default="qwen3.5-0.8b")
    args = p.parse_args()

    images = chapter_images(args.url)
    print(f"chapter lists {len(images)} images; smoking first: {images[0]}", flush=True)
    raw = _get(images[0])
    from PIL import Image

    with Image.open(io.BytesIO(raw)) as img:
        img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode()
    payload = {
        "model": args.model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Read all text visible in this manga page image. Reply with a JSON object: {\"text\": \"<all visible text, one line per line>\"}"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }],
        "response_format": {"type": "json_object"},
        "max_tokens": 8192,
        "stream": False,
    }
    req = urllib.request.Request(
        LLAMA, data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json", "Authorization": "Bearer local"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        body = json.loads(resp.read())
    content = body["choices"][0]["message"]["content"]
    print("raw reply:", content[:500], flush=True)
    data = json.loads(content)
    text = (data.get("text") or "").strip()
    if not isinstance(data, dict) or not text:
        print("SMOKE FAILED: reply is not a JSON object with non-empty text", file=sys.stderr)
        return 1
    print(f"SMOKE OK: vision call returned JSON, text chars={len(text)}", flush=True)
    print(text[:300], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
