#!/usr/bin/env python3
import os
import sys

from app.secret_store import (
    get_provider_config,
    provider_key_status,
    set_provider_api_key,
    set_provider_config,
)

if os.getenv("MANGA_ALLOW_LOCAL_PROVIDER") != "1":
    print("MANGA_ALLOW_LOCAL_PROVIDER is not 1, refusing to wire a local provider", file=sys.stderr)
    raise SystemExit(1)

set_provider_config(
    "qwen-local",
    label="Qwen Local",
    protocol="openai",
    api_base="http://127.0.0.1:8080/v1",
)
set_provider_api_key("qwen-local", "local")
config = get_provider_config("qwen-local")
status = provider_key_status("qwen-local")
print("config:", {k: config.get(k) for k in ("id", "label", "protocol", "api_base")})
print("key status:", status)
print("WIRING OK" if status.get("configured") else "WIRING FAILED")
raise SystemExit(0 if status.get("configured") else 1)
