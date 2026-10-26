#!/usr/bin/env python3
"""Verify a mixed soup checkpoint by its weights, never by file bytes.

YOLO checkpoints stamp the save date into the file, so two mixes of the
same inputs never share an md5 even though every weight is identical
(proven: 708/708 tensors exactly equal across independent mixes). So this
checks the weights themselves: the soup's float tensors must equal the
50/50 average of the inputs, non-float buffers must match the first input,
and the soup must load and run one predict.

Weights are read with torch.load, not YOLO(): YOLO upcasts stored fp16 to
fp32 on load, which would make an exact comparison fail on fp16 rounding
(max observed diff 2^-11) even for a correct soup.
"""
import sys

import numpy as np
import torch
from ultralytics import YOLO


def state_dict(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = ckpt.get("model", ckpt)
    return model.state_dict() if hasattr(model, "state_dict") else model


def main():
    a_path, b_path, soup_path = sys.argv[1], sys.argv[2], sys.argv[3]
    a, b, s = state_dict(a_path), state_dict(b_path), state_dict(soup_path)
    assert set(s) == set(a) == set(b), "state dict keys differ"
    checked = 0
    for k in s:
        ta, tb, ts = a[k], b[k], s[k]
        assert ta.shape == tb.shape == ts.shape, f"shape mismatch on {k}"
        if ta.is_floating_point():
            expect = (0.5 * ta.float() + 0.5 * tb.float()).to(ta.dtype)
            assert torch.equal(expect, ts), f"tensor mismatch on {k}"
        else:
            assert torch.equal(ta, ts), f"buffer mismatch on {k}"
        checked += 1
    YOLO(soup_path).predict(
        np.full((640, 640, 3), 128, dtype=np.uint8), imgsz=640, conf=0.25, verbose=False
    )
    print(f"soup verified: {checked}/{len(s)} tensors match the 50/50 average, smoke predict OK")


if __name__ == "__main__":
    main()
