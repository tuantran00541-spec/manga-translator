#!/usr/bin/env python3
"""Read the pipeline stage selector and export it as step outputs.

stage.txt line 1: HOLD | B | C | D
  HOLD: no-op run (wiring check only, no downloads, no detection)
  B: yolo26s single from run 8 (artifact kiuyha-tuned-s, run 40249343243)
  C: fine-tune single from run 9 (artifact kiuyha-tuned-s, run 40340455331)
  D: new soup — line 2 must be "<run1> <run2> <md5>" (top-2 runs + md5 of
     the locally mixed .pt, which the job re-verifies after mixing)
Unknown stage or a D stage without line 2 fails fast.
"""
import os
from pathlib import Path

STAGES = {
    "B": {"run1": "40249343243"},
    "C": {"run1": "40340455331"},
}


def main():
    lines = [ln.strip() for ln in Path(".github/detect-bulk-stage/stage.txt").read_text().splitlines() if ln.strip()]
    if not lines:
        raise SystemExit("stage.txt is empty")
    stage = lines[0]
    out = {"stage": stage, "run1": "", "run2": "", "md5": ""}
    if stage == "HOLD":
        pass
    elif stage in STAGES:
        out.update(STAGES[stage])
    elif stage == "D":
        if len(lines) < 2:
            raise SystemExit("stage D needs '<run1> <run2> <md5>' on line 2 of stage.txt")
        parts = lines[1].split()
        if len(parts) != 3:
            raise SystemExit("stage D line 2 must be '<run1> <run2> <md5>'")
        out["run1"], out["run2"], out["md5"] = parts
    else:
        raise SystemExit(f"unknown stage {stage!r}")
    print(f"stage={stage} run1={out['run1']} run2={out['run2']} md5={out['md5']}")
    if "GITHUB_OUTPUT" in os.environ:
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as fh:
            for k, v in out.items():
                fh.write(f"{k}={v}\n")


if __name__ == "__main__":
    main()
