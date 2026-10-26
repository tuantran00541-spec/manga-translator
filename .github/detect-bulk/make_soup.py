#!/usr/bin/env python3
"""Average two same-architecture YOLO checkpoints (model soup) into one."""
import sys

import torch
from ultralytics import YOLO


def soup(a_path, b_path, out_path, alpha=0.5):
    a = YOLO(a_path)
    b = YOLO(b_path)
    sa = a.model.state_dict()
    sb = b.model.state_dict()
    assert set(sa) == set(sb), "state dict keys differ"
    mixed = {}
    for k in sa:
        ta, tb = sa[k], sb[k]
        assert ta.shape == tb.shape, f"shape mismatch on {k}: {ta.shape} vs {tb.shape}"
        if ta.is_floating_point():
            mixed[k] = (alpha * ta.float() + (1 - alpha) * tb.float()).to(ta.dtype)
        else:
            mixed[k] = ta  # counters and buffers: keep a's
    a.model.load_state_dict(mixed)
    a.save(out_path)
    print(f"soup saved: {out_path} (alpha={alpha} of {a_path})")


if __name__ == "__main__":
    soup(sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]) if len(sys.argv) > 4 else 0.5)
