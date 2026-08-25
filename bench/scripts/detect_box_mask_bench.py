#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.scripts.detect_box_mask_bench_v3 import main


if __name__ == "__main__":
    raise SystemExit(main())
