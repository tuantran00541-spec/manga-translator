"""Turn the released comic-text-detector ONNX into a fast, any-size, mask-only model."""
from __future__ import annotations

import sys

import numpy as np
import onnx
import onnx2torch
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnx import numpy_helper, utils


class Poly(nn.Module):
    """A stride-2 k4 p1 transposed conv rewritten exactly as a 3x3 conv into 4 phases and a pixel shuffle."""

    def __init__(self, ct: nn.ConvTranspose2d):
        super().__init__()
        w = ct.weight.data
        cin, cout = w.shape[:2]
        k = torch.zeros(cout, 2, 2, cin, 3, 3)
        taps = {0: ((-1, 3), (0, 1)), 1: ((0, 2), (1, 0))}
        for a in (0, 1):
            for b in (0, 1):
                for dy, ky in taps[a]:
                    for dx, kx in taps[b]:
                        k[:, a, b, :, dy + 1, dx + 1] = w[:, :, ky, kx].t()
        self.conv = nn.Conv2d(cin, cout * 4, 3, padding=1, bias=ct.bias is not None)
        self.conv.weight.data = k.reshape(cout * 4, cin, 3, 3)
        if ct.bias is not None:
            self.conv.bias.data = ct.bias.data.repeat_interleave(4)

    def forward(self, x):
        return F.pixel_shuffle(self.conv(x), 2)


def swap(module: nn.Module) -> nn.Module:
    for name, child in module.named_children():
        if isinstance(child, nn.ConvTranspose2d) and child.kernel_size == (4, 4) and child.stride == (2, 2) \
                and child.padding == (1, 1) and child.groups == 1:
            setattr(module, name, Poly(child))
        else:
            swap(child)
    return module


def main(src: str, dst: str) -> None:
    model = swap(onnx2torch.convert(src).eval())
    x = torch.rand(1, 3, 1024, 1024)
    torch.onnx.export(model, x, dst + ".full", input_names=["images"], output_names=["blk", "seg", "det"], opset_version=17,
                      dynamic_axes={"images": {2: "h", 3: "w"}, "seg": {2: "h", 3: "w"}, "det": {2: "h", 3: "w"}, "blk": {1: "n"}},
                      dynamo=False)
    m = onnx.load(dst + ".full")
    for init in m.graph.initializer:
        a = numpy_helper.to_array(init)
        if a.dtype == np.float32:  # denormal weights make every CPU conv crawl
            init.CopyFrom(numpy_helper.from_array(np.where(np.abs(a) < 1.18e-38, 0, a).astype(np.float32), init.name))
    onnx.save(m, dst + ".full")
    utils.extract_model(dst + ".full", dst, ["images"], ["seg"])
    ref = ort.InferenceSession(src, providers=["CPUExecutionProvider"]).run(["seg"], {"images": x.numpy()})[0]
    new = ort.InferenceSession(dst, providers=["CPUExecutionProvider"]).run(None, {"images": x.numpy()})[0]
    print("max seg difference", float(np.abs(ref - new).max()))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
