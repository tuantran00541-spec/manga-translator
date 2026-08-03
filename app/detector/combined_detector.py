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
                # Generously expand horizontal bounds (-24 to +24) so first/last letters are never cut off
                raw_min_x = min(t.x1 for t in inside_text) - 24
                raw_min_y = min(t.y1 for t in inside_text) - 10
                raw_max_x = max(t.x2 for t in inside_text) + 24
                raw_max_y = max(t.y2 for t in inside_text) + 10

                min_x = max(b.x1 + 2, raw_min_x)
                min_y = max(b.y1 + 2, raw_min_y)
                max_x = min(b.x2 - 2, raw_max_x)
                max_y = min(b.y2 - 2, raw_max_y)

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
                # Bubble without detected text inside: only keep if valid segmentation mask exists
                w = b.x2 - b.x1
                h = b.y2 - b.y1
                if b.mask is not None and b.mask.shape == (h, w) and b.mask.any():
                    margin_x = max(2, int(w * 0.03))
                    margin_y = max(2, int(h * 0.03))
                    x1 = b.x1 + margin_x
                    y1 = b.y1 + margin_y
                    x2 = max(x1 + 1, b.x2 - margin_x)
                    y2 = max(y1 + 1, b.y2 - margin_y)
                    cropped_mask = b.mask[margin_y : h - margin_y, margin_x : w - margin_x]
                    if cropped_mask.shape == (y2 - y1, x2 - x1):
                        result_boxes.append(BubbleBox(x1, y1, x2, y2, b.confidence, cropped_mask))

        # Process free text boxes (outside bubbles, e.g. captions, SFX, standalone monologues)
        standalone_text = [
            t for i, t in enumerate(text_boxes)
            if i not in used_text_boxes
        ]

        clustered_free_text = self._cluster_free_text_boxes(standalone_text)
        result_boxes.extend(clustered_free_text)

        return result_boxes

    def _cluster_free_text_boxes(self, boxes: list[BubbleBox]) -> list[BubbleBox]:
        if not boxes:
            return []

        remaining = list(boxes)
        clustered_results = []

        while remaining:
            cluster = [remaining.pop(0)]
            changed = True
            while changed:
                changed = False
                still_remaining = []
                for b in remaining:
                    if any(self._is_free_text_close(b, c) for c in cluster):
                        cluster.append(b)
                        changed = True
                    else:
                        still_remaining.append(b)
                remaining = still_remaining

            if len(cluster) == 1:
                b = cluster[0]
                # Expand single free text box horizontally (-20 / +20) to ensure full text line is covered
                min_x = max(0, b.x1 - 20)
                min_y = max(0, b.y1 - 6)
                max_x = b.x2 + 20
                max_y = b.y2 + 6
                merged_mask = self._merge_masks([b], min_x, min_y, max_x, max_y)
                clustered_results.append(BubbleBox(min_x, min_y, max_x, max_y, b.confidence, merged_mask))
            else:
                min_x = max(0, min(t.x1 for t in cluster) - 20)
                min_y = max(0, min(t.y1 for t in cluster) - 8)
                max_x = max(t.x2 for t in cluster) + 20
                max_y = max(t.y2 for t in cluster) + 8
                merged_mask = self._merge_masks(cluster, min_x, min_y, max_x, max_y)
                clustered_results.append(
                    BubbleBox(
                        min_x, min_y, max_x, max_y,
                        max(t.confidence for t in cluster),
                        merged_mask,
                    )
                )

        return clustered_results

    @staticmethod
    def _is_free_text_close(a: BubbleBox, b: BubbleBox) -> bool:
        x_overlap = not (a.x2 < b.x1 - 35 or b.x2 < a.x1 - 35)
        y_close = abs(a.y1 - b.y1) < 45 or abs(a.y2 - b.y2) < 45 or not (a.y2 < b.y1 - 40 or b.y2 < a.y1 - 40)
        return x_overlap and y_close

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
