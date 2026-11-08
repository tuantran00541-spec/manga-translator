#!/usr/bin/env python3
"""Register the MiniRouter provider from the MINIROUTER_API_KEY env var.

Runs on a CI runner where the real key comes from the repo secret. Prints
only whether the provider is configured, never the key.
"""
import os
import sys

from app.secret_store import (
    get_provider_config,
    provider_key_status,
    set_provider_api_key,
    set_provider_config,
)

key = (os.environ.get("MINIROUTER_API_KEY") or "").strip()
if not key:
    print("MINIROUTER_API_KEY is empty", file=sys.stderr)
    raise SystemExit(1)

set_provider_config(
    "minirouter",
    label="MiniRouter",
    protocol="openai",
    api_base="https://api.minirouter.sh/v1",
)
set_provider_api_key("minirouter", key)
config = get_provider_config("minirouter")
status = provider_key_status("minirouter")
print("config:", {k: config.get(k) for k in ("id", "label", "protocol", "api_base")})
print("key status:", status)
print("WIRING OK" if status.get("configured") else "WIRING FAILED")
raise SystemExit(0 if status.get("configured") else 1)
