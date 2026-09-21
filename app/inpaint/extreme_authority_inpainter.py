from __future__ import annotations

import numpy as np

from app.detector.mask_builder import build_mask
from app.inpaint.adaptive_fast_inpainter import AdaptiveFastInpainter
from app.region_policy import subtract_regions_from_mask


class ExtremeAuthorityInpainter(AdaptiveFastInpainter):
    """AdaptiveFastInpainter with a hard final write clip to detector authority."""

    def inpaint(
        self,
        image: np.ndarray,
        boxes,
        protected_regions: list[dict] | None = None,
    ) -> np.ndarray:
        authority = (
            build_mask(image.shape[:2], boxes, image)
            if boxes
            else np.zeros(image.shape[:2], dtype=np.uint8)
        )
        authority = subtract_regions_from_mask(authority, protected_regions)
        if authority is None or not np.any(authority > 127):
            return image.copy()

        candidate = super().inpaint(
            image.copy(),
            boxes,
            protected_regions=protected_regions,
        )
        result = image.copy()
        writable = authority > 127
        result[writable] = candidate[writable]
        return result
