import numpy as np
from app.detector.bubble_detector import YoloDetector, BubbleBox
from app.config import (
    BUBBLE_DETECTOR_MODEL,
    TEXT_SEGMENTER_MODEL,
    BUBBLE_CONF_THRESHOLD,
    TEXT_CONF_THRESHOLD,
)


class CombinedTextDetector:
    def __init__(self):
        self.bubble_detector = YoloDetector(BUBBLE_DETECTOR_MODEL, BUBBLE_CONF_THRESHOLD)
        self.text_detector = YoloDetector(TEXT_SEGMENTER_MODEL, TEXT_CONF_THRESHOLD)

    def detect(self, image: np.ndarray) -> list[BubbleBox]:
        bubble_boxes = self.bubble_detector.detect(image)
        text_boxes = self.text_detector.detect(image)

        result_boxes = []
        used_text_boxes = set()

        for b in bubble_boxes:
            inside_text = [
                t for i, t in enumerate(text_boxes)
                if self._is_inside(t, b)
            ]
            if inside_text:
                raw_min_x = min(t.x1 for t in inside_text) - 8
                raw_min_y = min(t.y1 for t in inside_text) - 8
                raw_max_x = max(t.x2 for t in inside_text) + 8
                raw_max_y = max(t.y2 for t in inside_text) + 8

                min_x = max(b.x1 + 3, raw_min_x)
                min_y = max(b.y1 + 3, raw_min_y)
                max_x = min(b.x2 - 3, raw_max_x)
                max_y = min(b.y2 - 3, raw_max_y)

                if max_x > min_x and max_y > min_y:
                    min_x, min_y, max_x, max_y = int(min_x), int(min_y), int(max_x), int(max_y)
                    merged_mask = self._merge_masks(inside_text, min_x, min_y, max_x, max_y)
                    merged_box = BubbleBox(
                        min_x, min_y, max_x, max_y,
                        max(t.confidence for t in inside_text),
                        merged_mask,
                    )
                    result_boxes.append(merged_box)

                for i, t in enumerate(text_boxes):
                    if self._is_inside(t, b):
                        used_text_boxes.add(i)
            else:
                w = b.x2 - b.x1
                h = b.y2 - b.y1
                margin_x = max(4, int(w * 0.04))
                margin_y = max(4, int(h * 0.04))
                inner_box = BubbleBox(
                    b.x1 + margin_x,
                    b.y1 + margin_y,
                    max(b.x1 + margin_x + 1, b.x2 - margin_x),
                    max(b.y1 + margin_y + 1, b.y2 - margin_y),
                    b.confidence,
                )
                result_boxes.append(inner_box)

        for i, t in enumerate(text_boxes):
            if i not in used_text_boxes:
                result_boxes.append(t)

        return result_boxes

    @staticmethod
    def _merge_masks(inside_text: list[BubbleBox], min_x: int, min_y: int, max_x: int, max_y: int) -> np.ndarray | None:
        if any(t.mask is None for t in inside_text):
            return None

        merged = np.zeros((max_y - min_y, max_x - min_x), dtype=np.uint8)
        for t in inside_text:
            tx1 = max(min_x, t.x1)
            ty1 = max(min_y, t.y1)
            tx2 = min(max_x, t.x2)
            ty2 = min(max_y, t.y2)
            if tx2 <= tx1 or ty2 <= ty1:
                continue
            mask_x1 = tx1 - t.x1
            mask_y1 = ty1 - t.y1
            mask_x2 = mask_x1 + (tx2 - tx1)
            mask_y2 = mask_y1 + (ty2 - ty1)
            sub_mask = t.mask[mask_y1:mask_y2, mask_x1:mask_x2]
            dest = merged[ty1 - min_y:ty2 - min_y, tx1 - min_x:tx2 - min_x]
            merged[ty1 - min_y:ty2 - min_y, tx1 - min_x:tx2 - min_x] = np.maximum(dest, sub_mask)

        return merged

    @staticmethod
    def _is_inside(text_box: BubbleBox, bubble_box: BubbleBox) -> bool:
        tc_x = (text_box.x1 + text_box.x2) / 2
        tc_y = (text_box.y1 + text_box.y2) / 2
        return (
            bubble_box.x1 <= tc_x <= bubble_box.x2
            and bubble_box.y1 <= tc_y <= bubble_box.y2
        )
