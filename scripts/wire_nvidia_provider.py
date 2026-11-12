#!/usr/bin/env python3
import os
import sys

from app.secret_store import (
    get_provider_config,
    provider_key_status,
    set_provider_api_key,
    set_provider_config,
)

key = (os.environ.get("NVIDIA_API_KEY") or "").strip()
if not key:
    print("NVIDIA_API_KEY is empty", file=sys.stderr)
    raise SystemExit(1)

set_provider_config(
    "nvidia",
    label="NVIDIA NIM",
    protocol="openai",
    api_base="https://integrate.api.nvidia.com/v1",
)
set_provider_api_key("nvidia", key)
config = get_provider_config("nvidia")
status = provider_key_status("nvidia")
print("config:", {k: config.get(k) for k in ("id", "label", "protocol", "api_base")})
print("key status:", status)
print("WIRING OK" if status.get("configured") else "WIRING FAILED")
raise SystemExit(0 if status.get("configured") else 1)
