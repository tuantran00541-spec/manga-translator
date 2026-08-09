import numpy as np
import cv2
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
                avg_box_h = sum(t.y2 - t.y1 for t in inside_text) / len(inside_text)
                cluster_h = max(t.y2 for t in inside_text) - min(t.y1 for t in inside_text)
                if len(inside_text) > 3 or cluster_h > 4.0 * avg_box_h:
                    sub_groups = self._split_cluster_by_lines(inside_text, avg_box_h)
                else:
                    sub_groups = [inside_text]

                for group in sub_groups:
                    raw_min_x = min(t.x1 for t in group) - 24
                    raw_min_y = min(t.y1 for t in group) - 10
                    raw_max_x = max(t.x2 for t in group) + 24
                    raw_max_y = max(t.y2 for t in group) + 10

                    min_x = max(b.x1 - 8, raw_min_x)
                    min_y = max(b.y1 - 8, raw_min_y)
                    max_x = min(b.x2 + 8, raw_max_x)
                    max_y = min(b.y2 + 8, raw_max_y)

                    if max_x > min_x and max_y > min_y:
                        min_x, min_y, max_x, max_y = int(min_x), int(min_y), int(max_x), int(max_y)
                        merged_mask = self._merge_masks(group, min_x, min_y, max_x, max_y)
                        merged_box = BubbleBox(
                            min_x, min_y, max_x, max_y,
                            max(t.confidence for t in group),
                            merged_mask,
                        )
                        result_boxes.append(merged_box)

                for i, t in enumerate(text_boxes):
                    if self._is_inside(t, b):
                        used_text_boxes.add(i)
            else:
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

        standalone_text = [
            t for i, t in enumerate(text_boxes)
            if i not in used_text_boxes
        ]

        clustered_free_text = self._cluster_free_text_boxes(standalone_text, w, h)
        result_boxes.extend(clustered_free_text)

        result_boxes = self._refine_and_split_tall_boxes(result_boxes, image)
        result_boxes = self._apply_final_nms(result_boxes, iou_threshold=0.35)

        return result_boxes

    @staticmethod
    def _apply_final_nms(boxes: list[BubbleBox], iou_threshold: float = 0.35) -> list[BubbleBox]:
        if not boxes:
            return []
        rects = np.array([[b.x1, b.y1, max(1, b.x2 - b.x1), max(1, b.y2 - b.y1)] for b in boxes])
        scores = np.array([b.confidence for b in boxes])
        indices = cv2.dnn.NMSBoxes(
            rects.tolist(), scores.tolist(), score_threshold=0.05, nms_threshold=iou_threshold
        )
        if len(indices) == 0:
            return []
        return [boxes[i] for i in np.array(indices).flatten()]

    @staticmethod
    def _refine_and_split_tall_boxes(boxes: list[BubbleBox], img: np.ndarray) -> list[BubbleBox]:
        img_h, img_w = img.shape[:2]
        full_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

        refined_boxes = []

        for box in boxes:
            bh = box.y2 - box.y1
            bw = box.x2 - box.x1
            if bh <= 45:
                rect_mask = np.full((bh, bw), 255, dtype=np.uint8) if box.mask is None else box.mask
                refined_boxes.append(BubbleBox(box.x1, box.y1, box.x2, box.y2, box.confidence, rect_mask))
                continue

            crop_x1 = box.x1
            crop_x2 = box.x2
            y1_exp = box.y1
            y2_exp = box.y2

            exp_crop = full_gray[y1_exp:y2_exp, crop_x1:crop_x2]
            if exp_crop.size == 0:
                rect_mask = np.full((bh, bw), 255, dtype=np.uint8) if box.mask is None else box.mask
                refined_boxes.append(BubbleBox(box.x1, box.y1, box.x2, box.y2, box.confidence, rect_mask))
                continue

            crop_bg = np.percentile(exp_crop, 90)
            text_rows = exp_crop.mean(axis=1) < (crop_bg - 2)

            line_bounds = []
            in_line = False
            start_y = 0
            for y, is_text in enumerate(text_rows):
                if is_text and not in_line:
                    in_line = True
                    start_y = y
                elif not is_text and in_line:
                    in_line = False
                    if y - start_y >= 4:
                        line_bounds.append((start_y, y))

            if in_line and (len(text_rows) - start_y) >= 4:
                line_bounds.append((start_y, len(text_rows)))

            if len(line_bounds) <= 1:
                rect_mask = np.full((bh, bw), 255, dtype=np.uint8) if box.mask is None else box.mask
                refined_boxes.append(BubbleBox(box.x1, box.y1, box.x2, box.y2, box.confidence, rect_mask))
                continue

            for idx, (ly1, ly2) in enumerate(line_bounds):
                prev_y = line_bounds[idx - 1][1] if idx > 0 else 0
                next_y = line_bounds[idx + 1][0] if idx < len(line_bounds) - 1 else (y2_exp - y1_exp)

                pad_top = min(3, max(1, (ly1 - prev_y) // 2))
                pad_bot = min(3, max(1, (next_y - ly2) // 2))

                abs_y1 = max(0, y1_exp + ly1 - pad_top)
                abs_y2 = min(img_h, y1_exp + ly2 + pad_bot)

                line_strip = full_gray[abs_y1:abs_y2, crop_x1:crop_x2]
                if line_strip.size == 0:
                    abs_x1 = max(0, box.x1 - 20)
                    abs_x2 = min(img_w, box.x2 + 20)
                else:
                    col_means = line_strip.mean(axis=0)
                    strip_bg = np.percentile(col_means, 90)
                    text_col_indices = np.where(col_means < (strip_bg - 2))[0]

                    if len(text_col_indices) > 0:
                        abs_x1 = max(box.x1, crop_x1 + int(text_col_indices.min()) - 10)
                        abs_x2 = min(box.x2, crop_x1 + int(text_col_indices.max()) + 10)
                    else:
                        abs_x1 = box.x1
                        abs_x2 = box.x2

                line_h = abs_y2 - abs_y1
                line_w = abs_x2 - abs_x1

                m_y1 = abs_y1 - box.y1
                m_y2 = abs_y2 - box.y1
                m_x1 = abs_x1 - box.x1
                m_x2 = abs_x2 - box.x1

                if box.mask is not None and box.mask.ndim == 2:
                    line_mask = np.zeros((line_h, line_w), dtype=box.mask.dtype)

                    src_x1 = max(box.x1, abs_x1)
                    src_y1 = max(box.y1, abs_y1)
                    src_x2 = min(box.x2, abs_x2)
                    src_y2 = min(box.y2, abs_y2)

                    if src_x2 > src_x1 and src_y2 > src_y1:
                        src_x1i = src_x1 - box.x1
                        src_y1i = src_y1 - box.y1
                        src_x2i = src_x2 - box.x1
                        src_y2i = src_y2 - box.y1
                        dst_x1 = src_x1 - abs_x1
                        dst_y1 = src_y1 - abs_y1
                        dst_x2 = dst_x1 + (src_x2 - src_x1)
                        dst_y2 = dst_y1 + (src_y2 - src_y1)
                        line_mask[dst_y1:dst_y2, dst_x1:dst_x2] = box.mask[src_y1i:src_y2i, src_x1i:src_x2i]
                else:
                    line_mask = np.full((line_h, line_w), 255, dtype=np.uint8)

                refined_boxes.append(BubbleBox(abs_x1, abs_y1, abs_x2, abs_y2, box.confidence, line_mask))

        return refined_boxes

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
                    if max_x <= min_x or max_y <= min_y:
                        continue
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
                    if max_x <= min_x or max_y <= min_y:
                        continue
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
        lines = []
        for b in sorted_boxes:
            placed = False
            for line in lines:
                line_y1 = min(x.y1 for x in line)
                line_y2 = max(x.y2 for x in line)
                overlap = min(b.y2, line_y2) - max(b.y1, line_y1)
                min_h = min(b.y2 - b.y1, line_y2 - line_y1)
                if min_h > 0 and overlap / min_h > 0.5:
                    line.append(b)
                    placed = True
                    break
            if not placed:
                lines.append([b])

        lines.sort(key=lambda line: min(b.y1 for b in line))

        sub_clusters = []
        current_group = []
        for line in lines:
            if not current_group:
                current_group = list(line)
            else:
                group_h = max(b.y2 for b in current_group + line) - min(b.y1 for b in current_group + line)
                if group_h > 3.0 * avg_box_h:
                    sub_clusters.append(current_group)
                    current_group = list(line)
                else:
                    current_group.extend(line)
        if current_group:
            sub_clusters.append(current_group)

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
        if bubble_box.x1 <= tc_x <= bubble_box.x2 and bubble_box.y1 <= tc_y <= bubble_box.y2:
            return True

        ix1 = max(text_box.x1, bubble_box.x1)
        iy1 = max(text_box.y1, bubble_box.y1)
        ix2 = min(text_box.x2, bubble_box.x2)
        iy2 = min(text_box.y2, bubble_box.y2)

        if ix2 > ix1 and iy2 > iy1:
            intersection_area = (ix2 - ix1) * (iy2 - iy1)
            text_box_area = (text_box.x2 - text_box.x1) * (text_box.y2 - text_box.y1)
            if text_box_area > 0 and (intersection_area / text_box_area) >= 0.50:
                return True

        return False
