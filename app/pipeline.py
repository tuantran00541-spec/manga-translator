from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

# ... existing imports/constants/classes remain unchanged ...

    def _reinpaint_page(self, page: dict, image, processed_dir: Path, img_path: Path) -> None:
        boxes = [
            BubbleBox(
                b["x1"], b["y1"], b["x2"], b["y2"], b["confidence"],
                _decode_mask(b.get("mask")),
            )
            for b in page["boxes"]
            if not b.get("removed")
        ]
        clean_image = self.inpainter.inpaint(image, boxes)

        manual_mask_path = processed_dir / f"manual_mask_{img_path.name}"
        if manual_mask_path.exists():
            raw = np.fromfile(str(manual_mask_path), dtype=np.uint8)
            manual_mask = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
            if manual_mask is not None and manual_mask.any():
                # The manual mask is already the union of all strokes made by
                # the user on this page.  Keep it as one mask and let the
                # inpainter process all requested regions in one invocation.
                # Previously each connected component triggered a separate
                # LaMa inference, so painting 2-3 bubbles could turn one HTTP
                # request into several expensive sequential inferences and
                # hit the request timeout.
                manual_mask = (manual_mask > 127).astype(np.uint8) * 255
                clean_image = self.inpainter.inpaint_mask(clean_image, manual_mask)

        clean_path = processed_dir / f"clean_{img_path.name}"
        write_image(clean_path, clean_image)
        page["clean"] = clean_path.as_posix()
        page["rendered"] = False



    def _process_page(self, img_path: Path, processed_dir: Path, excluded_regions: list[dict] | None = None) -> dict:
        image = read_image(img_path)
        boxes = self.detector.detect(image)
        if excluded_regions:
            boxes = [b for b in boxes if not self._box_in_excluded(b, excluded_regions)]
        clean_image = self.inpainter.inpaint(image, boxes)

        clean_path = processed_dir / f"clean_{img_path.name}"
        write_image(clean_path, clean_image)

        logger.debug(f"Processed {img_path.name}: {len(boxes)} boxes detected (after excluded filtering)")
        return {
            "clean": clean_path.as_posix(),
            "boxes": [
                {
                    "x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2,
                    "confidence": b.confidence,
                    "mask": _encode_mask(b.mask),
                }
                for b in boxes
            ],
        }

    @staticmethod
    def _box_in_excluded(box, excluded_regions: list[dict]) -> bool:
        box_cx = (box.x1 + box.x2) / 2
        box_cy = (box.y1 + box.y2) / 2
        for r in excluded_regions:
            x1 = r.get("x1", 0)
            y1 = r.get("y1", 0)
            x2 = r.get("x2", 0)
            y2 = r.get("y2", 0)
            if min(x1, x2) <= box_cx <= max(x1, x2) and min(y1, y2) <= box_cy <= max(y1, y2):
                return True
        return False
