"""A stand-in for the comic-text-detector mask model: ink darker or lighter than set limits is letters."""
import cv2
import numpy as np


class InkModel:
    """Reads near-black (or near-white) strokes as letters, like the real model reads letter bodies but not outlines."""

    def __init__(self, dark_below=30, light_above=None):
        self.dark_below, self.light_above = dark_below, light_above
        self.shapes = []

    def run(self, _names, feeds):
        x = feeds["images"]
        self.shapes.append(x.shape)
        rgb = (x[0].transpose(1, 2, 0) * 255).astype(np.uint8)
        light = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[..., 0]
        ink = light < self.dark_below if self.light_above is None else light > self.light_above
        return [ink.astype(np.float32)[None, None]]
