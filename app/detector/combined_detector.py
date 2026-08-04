import numpy as np
from app.detector.bubble_detector import YoloDetector, BubbleBox, MAX_BOX_AREA_RATIO
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
        h, w = image.shape[:2]
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
                bw = b.x2 - b.x1
                bh = b.y2 - b.y1
                if b.mask is not None and b.mask.shape == (bh, bw) and b.mask.any():
                    margin_x = max(2, int(bw * 0.03))
                    margin_y = max(2, int(bh * 0.03))
                    x1 = b.x1 + margin_x
                    y1 = b.y1 + margin_y
                    x2 = max(x1 + 1, b.x2 - margin_x)
                    y2 = max(y1 + 1, b.y2 - margin_y)
                    cropped_mask = b.mask[margin_y : bh - margin_y, margin_x : bw - margin_x]
                    if cropped_mask.shape == (y2 - y1, x2 - x1):
                        result_boxes.append(BubbleBox(x1, y1, x2, y2, b.confidence, cropped_mask))

        # Process free text boxes (outside bubbles, e.g. captions, SFX, standalone monologues)
        standalone_text = [
            t for i, t in enumerate(text_boxes)
            if i not in used_text_boxes
        ]

        clustered_free_text = self._cluster_free_text_boxes(standalone_text, w, h)
        result_boxes.extend(clustered_free_text)

        return result_boxes

    def _cluster_free_text_boxes(self, boxes: list[BubbleBox], img_w: int, img_h: int) -> list[BubbleBox]:
        if not boxes:
            return []

        remaining = list(boxes)
        raw_clusters = []

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
            raw_clusters.append(cluster)

        final_clusters = []
        for cluster in raw_clusters:
            if len(cluster) > 1:
                avg_box_h = sum(b.y2 - b.y1 for b in cluster) / len(cluster)
                cluster_h = max(b.y2 for b in cluster) - min(b.y1 for b in cluster)
                if len(cluster) > 3 or cluster_h > 4.0 * avg_box_h:
                    sub_clusters = self._split_cluster_by_lines(cluster, avg_box_h)
                else:
                    sub_clusters = [cluster]
            else:
                sub_clusters = [cluster]

            for sub in sub_clusters:
                if len(sub) == 1:
                    b = sub[0]
                    min_x = max(0, b.x1 - 20)
                    min_y = max(0, b.y1 - 6)
                    max_x = min(img_w, b.x2 + 20)
                    max_y = min(img_h, b.y2 + 6)
                    cluster_area = (max_x - min_x) * (max_y - min_y)
                    if cluster_area > img_w * img_h * MAX_BOX_AREA_RATIO:
                        continue
                    merged_mask = self._merge_masks([b], min_x, min_y, max_x, max_y)
                    final_clusters.append(BubbleBox(min_x, min_y, max_x, max_y, b.confidence, merged_mask))
                else:
                    min_x = max(0, min(t.x1 for t in sub) - 20)
                    min_y = max(0, min(t.y1 for t in sub) - 8)
                    max_x = min(img_w, max(t.x2 for t in sub) + 20)
                    max_y = min(img_h, max(t.y2 for t in sub) + 8)
                    cluster_area = (max_x - min_x) * (max_y - min_y)
                    if cluster_area > img_w * img_h * MAX_BOX_AREA_RATIO:
                        continue
                    merged_mask = self._merge_masks(sub, min_x, min_y, max_x, max_y)
                    final_clusters.append(
                        BubbleBox(
                            min_x, min_y, max_x, max_y,
                            max(t.confidence for t in sub),
                            merged_mask,
                        )
                    )

        return final_clusters

    @staticmethod
    def _split_cluster_by_lines(cluster: list[BubbleBox], avg_box_h: float) -> list[list[BubbleBox]]:
        sorted_boxes = sorted(cluster, key=lambda b: (b.y1, b.x1))
        sub_clusters = []
        current_sub = []
        current_min_y = 0

        for b in sorted_boxes:
            if not current_sub:
                current_sub = [b]
                current_min_y = b.y1
            else:
                prev_b = current_sub[-1]
                new_h = max(box.y2 for box in current_sub + [b]) - min(box.y1 for box in current_sub + [b])
                if (
                    len(current_sub) >= 3
                    or new_h > 3.0 * avg_box_h
                    or (b.y1 - prev_b.y2 > 0.8 * avg_box_h and b.y1 - current_min_y > 1.5 * avg_box_h)
                ):
                    sub_clusters.append(current_sub)
                    current_sub = [b]
                    current_min_y = b.y1
                else:
                    current_sub.append(b)

        if current_sub:
            sub_clusters.append(current_sub)

        return sub_clusters

    @staticmethod
    def _is_free_text_close(a: BubbleBox, b: BubbleBox) -> bool:
        x_overlap = not (a.x2 < b.x1 - 35 or b.x2 < a.x1 - 35)
        y_close = abs(a.y1 - b.y1) < 45 or abs(a.y2 - b.y2) < 45 or not (a.y2 < b.y1 - 40 or b.y2 < a.y1 - 40)
        return x_overlap and y_close

    @staticmethod
    def _merge_masks(inside_text: list[BubbleBox], min_x: int, min_y: int, max_x: int, max_y: int) -> np.ndarray:
        merged = np.zeros((max_y - min_y, max_x - min_x), dtype=np.uint8)
        for t in inside_text:
            tx1 = max(min_x, t.x1)
            ty1 = max(min_y, t.y1)
            tx2 = min(max_x, t.x2)
            ty2 = min(max_y, t.y2)
            if tx2 <= tx1 or ty2 <= ty1:
                continue
            if t.mask is not None:
                mask_x1 = tx1 - t.x1
                mask_y1 = ty1 - t.y1
                mask_x2 = mask_x1 + (tx2 - tx1)
                mask_y2 = mask_y1 + (ty2 - ty1)
                sub_mask = t.mask[mask_y1:mask_y2, mask_x1:mask_x2]
                dest = merged[ty1 - min_y:ty2 - min_y, tx1 - min_x:tx2 - min_x]
                merged[ty1 - min_y:ty2 - min_y, tx1 - min_x:tx2 - min_x] = np.maximum(dest, sub_mask)
            else:
                merged[ty1 - min_y:ty2 - min_y, tx1 - min_x:tx2 - min_x] = 255

        return merged

    @staticmethod
    def _is_inside(text_box: BubbleBox, bubble_box: BubbleBox) -> bool:
        tc_x = (text_box.x1 + text_box.x2) / 2
        tc_y = (text_box.y1 + text_box.y2) / 2
        return (
            bubble_box.x1 <= tc_x <= bubble_box.x2
            and bubble_box.y1 <= tc_y <= bubble_box.y2
        )
