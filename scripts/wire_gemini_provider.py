#!/usr/bin/env python3
import os
import sys

from app.secret_store import (
    provider_key_status,
    set_provider_api_key,
)

key = (os.environ.get("GEMINI_API_KEY") or "").strip()
if not key:
    print("GEMINI_API_KEY is empty", file=sys.stderr)
    raise SystemExit(1)

set_provider_api_key("gemini", key)
status = provider_key_status("gemini")
print("key status:", status)
print("WIRING OK" if status.get("configured") else "WIRING FAILED")
raise SystemExit(0 if status.get("configured") else 1)
