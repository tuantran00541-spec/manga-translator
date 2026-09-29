#!/bin/sh
# Install once, then open any terminal and type: manga
set -e
cd "$(dirname "$0")"
for py in python3.12 python3.11 python3.10 python3; do
  if command -v "$py" >/dev/null 2>&1 && "$py" -c 'import sys; sys.exit(not (3,10) <= sys.version_info[:2] <= (3,12))'; then
    exec "$py" scripts/install.py
  fi
done
echo "Cần Python 3.10–3.12 (khuyên dùng 3.12)." >&2
exit 1
