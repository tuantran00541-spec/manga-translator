"""Strict adapter for the official MI-GAN 512 ONNX *pipeline* export.

The project uses 255=erase masks. The official MI-GAN pipeline uses the
opposite polarity: 255=known and 0=hole. This adapter validates the ONNX
contract before inference and composites only the requested erase pixels back
onto the original image, so a polarity/shape mistake cannot silently overwrite
unmasked artwork.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class MIGANContract:
    image_layout: str
    mask_layout: str
    output_layout: str


def erase_to_migan_keep_mask(erase_mask: np.ndarray) -> np.ndarray:
    if erase_mask is None or erase_mask.ndim != 2:
        raise ValueError("erase_mask must be a 2D array")
    return np.where(erase_mask > 127, 0, 255).astype(np.uint8)


def _layout(shape: list[Any] | tuple[Any, ...], channels: int) -> str:
    if len(shape) != 4:
        raise ValueError(f"Expected rank-4 tensor, got shape={shape!r}")
    if shape[1] == channels:
        return "nchw"
    if shape[-1] == channels:
        return "nhwc"
    # Dynamic channel dimensions are too ambiguous for a destructive benchmark.
    raise ValueError(f"Cannot prove channel layout for shape={shape!r}")


def _to_layout(array_hwc: np.ndarray, layout: str) -> np.ndarray:
    if layout == "nchw":
        if array_hwc.ndim == 2:
            return array_hwc[None, None, :, :]
        return array_hwc.transpose(2, 0, 1)[None]
    if layout == "nhwc":
        if array_hwc.ndim == 2:
            return array_hwc[None, :, :, None]
        return array_hwc[None]
    raise ValueError(layout)


def _from_layout(array: np.ndarray, layout: str) -> np.ndarray:
    if array.ndim != 4 or array.shape[0] != 1:
        raise ValueError(f"Unexpected MI-GAN output shape: {array.shape}")
    if layout == "nchw":
        return array[0].transpose(1, 2, 0)
    if layout == "nhwc":
        return array[0]
    raise ValueError(layout)


class MIGANPipelineInpainter:
    def __init__(self, model_path: str | Path):
        import onnxruntime as ort

        self.model_path = str(model_path)
        self.session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 2 or len(outputs) < 1:
            raise ValueError("Expected the official 2-input MI-GAN ONNX pipeline export")

        by_name = {item.name: item for item in inputs}
        if "image" not in by_name or "mask" not in by_name:
            raise ValueError(
                f"Expected MI-GAN pipeline inputs named image/mask, got {sorted(by_name)}"
            )
        image_meta = by_name["image"]
        mask_meta = by_name["mask"]
        output_meta = outputs[0]
        if image_meta.type != "tensor(uint8)" or mask_meta.type != "tensor(uint8)":
            raise ValueError(
                "This adapter only accepts the official uint8 ONNX pipeline export; "
                "raw float/4-channel MI-GAN models are intentionally rejected"
            )
        if output_meta.type != "tensor(uint8)":
            raise ValueError(f"Expected uint8 MI-GAN pipeline output, got {output_meta.type}")

        self.contract = MIGANContract(
            image_layout=_layout(image_meta.shape, 3),
            mask_layout=_layout(mask_meta.shape, 1),
            output_layout=_layout(output_meta.shape, 3),
        )
        self.image_name = image_meta.name
        self.mask_name = mask_meta.name
        self.output_name = output_meta.name

    def inpaint_mask(self, image_bgr: np.ndarray, erase_mask: np.ndarray) -> np.ndarray:
        if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("image_bgr must be HxWx3")
        if erase_mask.shape != image_bgr.shape[:2]:
            raise ValueError("erase mask and image dimensions must match")
        erase = erase_mask > 127
        if not np.any(erase):
            return image_bgr.copy()

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.uint8, copy=False)
        keep_mask = erase_to_migan_keep_mask(erase_mask)
        image_input = _to_layout(image_rgb, self.contract.image_layout)
        mask_input = _to_layout(keep_mask, self.contract.mask_layout)

        result = self.session.run(
            [self.output_name],
            {self.image_name: image_input, self.mask_name: mask_input},
        )[0]
        generated_rgb = _from_layout(result, self.contract.output_layout)
        if generated_rgb.shape != image_rgb.shape:
            raise ValueError(
                f"MI-GAN pipeline changed spatial size from {image_rgb.shape} to {generated_rgb.shape}"
            )
        generated_bgr = cv2.cvtColor(generated_rgb.astype(np.uint8, copy=False), cv2.COLOR_RGB2BGR)

        # Safety contract: the learned backend is never allowed to alter pixels
        # outside the requested erase mask, even if the exported pipeline does.
        output = image_bgr.copy()
        output[erase] = generated_bgr[erase]
        return output
