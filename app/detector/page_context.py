from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Hashable

import cv2
import numpy as np


@dataclass
class PageContext:
    """Per-page image state shared by detector adapters.

    Expensive colour conversions and adapter-specific preprocess products are
    lazy and cached. Cropped child contexts retain page-space origin so adapter
    geometry can stay consistent without recomputing the parent page.
    """

    bgr: np.ndarray
    origin_x: int = 0
    origin_y: int = 0
    _rgb: np.ndarray | None = field(default=None, init=False, repr=False)
    _gray: np.ndarray | None = field(default=None, init=False, repr=False)
    _cache: dict[Hashable, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.bgr, np.ndarray):
            raise TypeError("PageContext.bgr must be a numpy array")
        if self.bgr.ndim not in (2, 3):
            raise ValueError(f"Unsupported page image rank: {self.bgr.ndim}")
        if self.bgr.shape[0] <= 0 or self.bgr.shape[1] <= 0:
            raise ValueError("PageContext image must be non-empty")

    @property
    def height(self) -> int:
        return int(self.bgr.shape[0])

    @property
    def width(self) -> int:
        return int(self.bgr.shape[1])

    @property
    def rgb(self) -> np.ndarray:
        if self._rgb is None:
            if self.bgr.ndim == 2:
                self._rgb = cv2.cvtColor(self.bgr, cv2.COLOR_GRAY2RGB)
            elif self.bgr.shape[2] == 3:
                self._rgb = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2RGB)
            else:
                raise ValueError(f"Unsupported channel count: {self.bgr.shape[2]}")
        return self._rgb

    @property
    def gray(self) -> np.ndarray:
        if self._gray is None:
            if self.bgr.ndim == 2:
                self._gray = self.bgr
            elif self.bgr.shape[2] == 3:
                self._gray = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2GRAY)
            else:
                raise ValueError(f"Unsupported channel count: {self.bgr.shape[2]}")
        return self._gray

    def cached(self, key: Hashable, builder: Callable[[], Any]) -> Any:
        if key not in self._cache:
            self._cache[key] = builder()
        return self._cache[key]

    def child(self, x1: int, y1: int, x2: int, y2: int) -> "PageContext":
        x1 = max(0, min(self.width, int(x1)))
        y1 = max(0, min(self.height, int(y1)))
        x2 = max(x1, min(self.width, int(x2)))
        y2 = max(y1, min(self.height, int(y2)))
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Child PageContext ROI must be non-empty")
        return PageContext(
            self.bgr[y1:y2, x1:x2],
            origin_x=self.origin_x + x1,
            origin_y=self.origin_y + y1,
        )
