import numpy as np
import cv2
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import threading
import time

from app.config import BUBBLE_DETECTOR_MODEL, TEXT_SEGMENTER_MODEL
from app.detector.bubble_detector import YoloDetector, BubbleBox, MAX_BOX_AREA_RATIO
from app.detector.recovery import SecondaryTextRecovery
from app.parameters import (
    BUBBLE_DESTRUCTIVE_CONF_THRESHOLD,
    BUBBLE_GROUP_PAD_X,
    BUBBLE_GROUP_PAD_Y,
    BUBBLE_PROPOSAL_CONF_THRESHOLD,
    BUBBLE_TEXT_OVERLAP_MIN,
    DETECTOR_TALL_SPLIT_BACKGROUND_PERCENTILE,
    DETECTOR_TALL_SPLIT_CONTRAST_DELTA,
    DETECTOR_TALL_SPLIT_HEIGHT_THRESHOLD,
    DETECTOR_TALL_SPLIT_HORIZONTAL_PADDING,
    DETECTOR_TALL_SPLIT_LINE_HEIGHT_MIN,
    DETECTOR_TALL_SPLIT_LINE_PADDING_MAX,
    DETECTOR_FINAL_NMS_IOU,
    DETECTOR_NMS_SCORE_FLOOR,
    FLAT_BUBBLE_BACKGROUND_RATIO_MIN,
    FLAT_BUBBLE_BLACK_MAX,
    FLAT_BUBBLE_BLACK_MEDIAN_MAX,
    FLAT_BUBBLE_DARK_TEXT_MAX,
    FLAT_BUBBLE_INSET_MAX,
    FLAT_BUBBLE_INSET_MIN,
    FLAT_BUBBLE_INSET_RATIO,
    FLAT_BUBBLE_INTERIOR_AREA_RATIO_MIN,
    FLAT_BUBBLE_INTERIOR_PIXELS_MIN,
    FLAT_BUBBLE_LIGHT_TEXT_MIN,
    FLAT_BUBBLE_MIN_SIDE,
    FLAT_BUBBLE_PAGE_AREA_MAX,
    FLAT_BUBBLE_STROKE_CLOSE_KERNEL,
    FLAT_BUBBLE_TEXT_BBOX_PAD,
    FLAT_BUBBLE_TEXT_RATIO_MAX,
    FLAT_BUBBLE_TEXT_RATIO_MIN,
    FLAT_BUBBLE_WHITE_MEDIAN_MIN,
    FLAT_BUBBLE_WHITE_MIN,
    FREE_TEXT_CLUSTER_HEIGHT_FACTOR,
    FREE_TEXT_CLUSTER_SPLIT_COUNT,
    FREE_TEXT_GROUP_HEIGHT_FACTOR,
    FREE_TEXT_GROUP_PAD_X,
    FREE_TEXT_GROUP_PAD_Y,
    FREE_TEXT_LINE_OVERLAP_MIN,
    FREE_TEXT_SINGLE_PAD_X,
    FREE_TEXT_SINGLE_PAD_Y,
    FREE_TEXT_X_GAP,
    FREE_TEXT_Y_ALIGNMENT,
    FREE_TEXT_Y_GAP,
    TAIL_CREDIT_HEIGHT_RATIO_MAX,
    TAIL_CREDIT_WIDTH_RATIO_MIN,
    TAIL_CREDIT_Y_RATIO_MIN,
    TEXT_CONF_THRESHOLD,
    WATERMARK_EDGE_X_RATIO,
    WATERMARK_EDGE_Y_RATIO,
    WATERMARK_HORIZONTAL_ASPECT_MIN,
    WATERMARK_HORIZONTAL_HEIGHT_RATIO_MAX,
    WATERMARK_VERTICAL_ASPECT_MIN,
    WATERMARK_VERTICAL_HEIGHT_RATIO_MAX,
)


class CombinedTextDetector:
    def __init__(self):
        self.bubble_detector = YoloDetector(
            BUBBLE_DETECTOR_MODEL, BUBBLE_PROPOSAL_CONF_THRESHOLD
        )
        self.text_detector = YoloDetector(TEXT_SEGMENTER_MODEL, TEXT_CONF_THRESHOLD)
        self.recovery = SecondaryTextRecovery()
        self._metrics_local = threading.local()

    def last_metrics(self) -> dict[str, float | int]:
        """Return detector counters from the current page-processing thread."""
        metrics = getattr(self._metrics_local, "value", {})
        return {
            str(name): float(value) if str(name).endswith("_ms") else int(value)
            for name, value in metrics.items()
        }

    @staticmethod
    def _watermark_like(box: BubbleBox, w: int, h: int) -> bool:
        bw, bh = box.x2 - box.x1, box.y2 - box.y1
        aspect = bw / max(1.0, float(bh))
        edge_touch = (
            box.x1 < w * WATERMARK_EDGE_X_RATIO
            or box.x2 > w * (1.0 - WATERMARK_EDGE_X_RATIO)
        )
        vertical_edge = (
            box.y1 < h * WATERMARK_EDGE_Y_RATIO
            or box.y2 > h * (1.0 - WATERMARK_EDGE_Y_RATIO)
        )
        return bool(
            (
                edge_touch
                and aspect >= WATERMARK_HORIZONTAL_ASPECT_MIN
                and bh <= h * WATERMARK_HORIZONTAL_HEIGHT_RATIO_MAX
            )
            or (
                vertical_edge
                and aspect >= WATERMARK_VERTICAL_ASPECT_MIN
                and bh <= h * WATERMARK_VERTICAL_HEIGHT_RATIO_MAX
            )
        )

    @staticmethod
    def _tail_credit_like(box: BubbleBox, w: int, h: int) -> bool:
        """Conservative geometry for end-card/credit text on the final source tail.

        This rule is deliberately context-gated by the pipeline. Applying the
        same lower-page geometry to every slice would incorrectly downgrade
        ordinary dialogue; on the final tail, a wide compact text block is a
        high-risk credit/end-card candidate and must fail safe to review.
        """
        bw, bh = box.x2 - box.x1, box.y2 - box.y1
        return bool(
            box.y1 >= h * TAIL_CREDIT_Y_RATIO_MIN
            and bw >= w * TAIL_CREDIT_WIDTH_RATIO_MIN
            and bh <= h * TAIL_CREDIT_HEIGHT_RATIO_MAX
        )

    def _classify(
        self, box: BubbleBox, w: int, h: int, *, protect_tail_credits: bool = False
    ) -> BubbleBox:
        if self._watermark_like(box, w, h) or (
            protect_tail_credits and self._tail_credit_like(box, w, h)
        ):
            return replace(box, semantic_type="watermark", class_name="watermark",
                           safe_to_inpaint=False, ocr_eligible=False, needs_review=True)
        if box.verified_mask and "text_segmenter" in box.source_model.lower():
            return replace(box, mask_source="text_segmenter", safe_to_inpaint=True,
                           ocr_eligible=True, needs_review=False)
        if (
            box.verified_mask
            and box.semantic_type == "speech_bubble"
            and box.mask_source == "bubble_flat_contrast"
        ):
            return replace(
                box,
                safe_to_inpaint=True,
                ocr_eligible=True,
                needs_review=False,
            )
        return replace(box, safe_to_inpaint=False, ocr_eligible=False, needs_review=True)

    @staticmethod
    def _flat_bubble_text_fallback(
        image: np.ndarray,
        box: BubbleBox,
        img_w: int,
        img_h: int,
    ) -> BubbleBox | None:
        """Recover text strokes only inside a high-confidence flat speech bubble.

        v0.1 used the bubble segmentation itself as the destructive mask whenever
        the text segmenter missed a bubble. That gave good recall but could erase
        artwork. This fallback keeps the useful signal while rebuilding a narrow
        contrast mask: the bubble must pass the destructive confidence threshold,
        its interior must be overwhelmingly white or black, and only contrasting
        stroke pixels become the inpaint mask.
        """
        if (
            box.semantic_type != "speech_bubble"
            or float(box.confidence) < BUBBLE_DESTRUCTIVE_CONF_THRESHOLD
            or not box.verified_mask
        ):
            return None

        bw = int(box.x2 - box.x1)
        bh = int(box.y2 - box.y1)
        if bw < FLAT_BUBBLE_MIN_SIDE or bh < FLAT_BUBBLE_MIN_SIDE:
            return None
        if (bw * bh) > float(max(1, img_w * img_h)) * FLAT_BUBBLE_PAGE_AREA_MAX:
            return None

        crop = image[box.y1:box.y2, box.x1:box.x2]
        if crop.shape[:2] != (bh, bw):
            return None
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop

        bubble_roi = (box.mask > 127).astype(np.uint8) * 255
        inset = max(
            FLAT_BUBBLE_INSET_MIN,
            min(
                FLAT_BUBBLE_INSET_MAX,
                int(round(min(bw, bh) * FLAT_BUBBLE_INSET_RATIO)),
            ),
        )
        kernel_size = inset * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        interior = cv2.erode(bubble_roi, kernel, iterations=1) > 0
        interior_count = int(np.count_nonzero(interior))
        if interior_count < max(
            FLAT_BUBBLE_INTERIOR_PIXELS_MIN,
            int(bw * bh * FLAT_BUBBLE_INTERIOR_AREA_RATIO_MIN),
        ):
            return None

        values = gray[interior]
        if values.size < FLAT_BUBBLE_INTERIOR_PIXELS_MIN:
            return None
        median = float(np.median(values))
        white_ratio = float(np.mean(values >= FLAT_BUBBLE_WHITE_MIN))
        black_ratio = float(np.mean(values <= FLAT_BUBBLE_BLACK_MAX))

        if (
            white_ratio >= FLAT_BUBBLE_BACKGROUND_RATIO_MIN
            and median >= FLAT_BUBBLE_WHITE_MEDIAN_MIN
        ):
            strokes = (gray <= FLAT_BUBBLE_DARK_TEXT_MAX) & interior
        elif (
            black_ratio >= FLAT_BUBBLE_BACKGROUND_RATIO_MIN
            and median <= FLAT_BUBBLE_BLACK_MEDIAN_MAX
        ):
            strokes = (gray >= FLAT_BUBBLE_LIGHT_TEXT_MIN) & interior
        else:
            return None

        stroke_count = int(np.count_nonzero(strokes))
        stroke_ratio = stroke_count / float(max(1, interior_count))
        if not (
            FLAT_BUBBLE_TEXT_RATIO_MIN
            <= stroke_ratio
            <= FLAT_BUBBLE_TEXT_RATIO_MAX
        ):
            return None

        stroke_mask = strokes.astype(np.uint8) * 255
        stroke_mask = cv2.morphologyEx(
            stroke_mask,
            cv2.MORPH_CLOSE,
            np.ones(
                (FLAT_BUBBLE_STROKE_CLOSE_KERNEL, FLAT_BUBBLE_STROKE_CLOSE_KERNEL),
                np.uint8,
            ),
            iterations=1,
        )
        ys, xs = np.nonzero(stroke_mask > 0)
        if xs.size == 0 or ys.size == 0:
            return None

        pad = FLAT_BUBBLE_TEXT_BBOX_PAD
        local_x1 = max(0, int(xs.min()) - pad)
        local_y1 = max(0, int(ys.min()) - pad)
        local_x2 = min(bw, int(xs.max()) + 1 + pad)
        local_y2 = min(bh, int(ys.max()) + 1 + pad)
        if local_x2 <= local_x1 or local_y2 <= local_y1:
            return None

        text_mask = stroke_mask[local_y1:local_y2, local_x1:local_x2].copy()
        if not np.any(text_mask > 0):
            return None

        return replace(
            box,
            x1=int(box.x1 + local_x1),
            y1=int(box.y1 + local_y1),
            x2=int(box.x1 + local_x2),
            y2=int(box.y1 + local_y2),
            mask=text_mask,
            class_name="text_bubble_fallback",
            semantic_type="speech_bubble",
            mask_source="bubble_flat_contrast",
            safe_to_inpaint=True,
            ocr_eligible=True,
            needs_review=False,
        )

    def detect(
        self, image: np.ndarray, *, parallel: bool = False, protect_tail_credits: bool = False
    ) -> list[BubbleBox]:
        started_at = time.perf_counter()
        h, w = image.shape[:2]
        if parallel:
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="detector") as pool:
                def _timed_detect(detector):
                    detector_started_at = time.perf_counter()
                    boxes = detector.detect(image)
                    return boxes, (time.perf_counter() - detector_started_at) * 1000.0

                bubble_future = pool.submit(_timed_detect, self.bubble_detector)
                text_future = pool.submit(_timed_detect, self.text_detector)
                bubble_boxes, bubble_ms = bubble_future.result()
                text_boxes, text_ms = text_future.result()
        else:
            bubble_started_at = time.perf_counter()
            bubble_boxes = self.bubble_detector.detect(image)
            bubble_ms = (time.perf_counter() - bubble_started_at) * 1000.0
            text_started_at = time.perf_counter()
            text_boxes = self.text_detector.detect(image)
            text_ms = (time.perf_counter() - text_started_at) * 1000.0

        metrics: dict[str, float | int] = {
            "bubble_model_ms": round(bubble_ms, 3),
            "text_model_ms": round(text_ms, 3),
            "mser_ms": 0.0,
            "bubble_proposals": len(bubble_boxes),
            "text_proposals": len(text_boxes),
            "result_boxes": 0,
            "review_boxes": 0,
            "total_ms": 0.0,
        }

        bubble_boxes = [
            self._classify(b, w, h, protect_tail_credits=protect_tail_credits)
            for b in bubble_boxes
        ]
        text_boxes = [
            self._classify(t, w, h, protect_tail_credits=protect_tail_credits)
            for t in text_boxes
        ]
        result_boxes: list[BubbleBox] = []
        used_text_boxes: set[int] = set()

        for b in bubble_boxes:
            inside = [(i, t) for i, t in enumerate(text_boxes) if self._is_inside(t, b)]
            if inside:
                group = [t for _, t in inside]
                min_x = max(0, min(t.x1 for t in group) - BUBBLE_GROUP_PAD_X)
                min_y = max(0, min(t.y1 for t in group) - BUBBLE_GROUP_PAD_Y)
                max_x = min(w, max(t.x2 for t in group) + BUBBLE_GROUP_PAD_X)
                max_y = min(h, max(t.y2 for t in group) + BUBBLE_GROUP_PAD_Y)
                merged_mask = self._merge_masks(group, min_x, min_y, max_x, max_y)
                seed = max(group, key=lambda t: t.confidence)
                safe = merged_mask is not None and bool(np.any(merged_mask > 0))
                merged = replace(seed, x1=int(min_x), y1=int(min_y), x2=int(max_x), y2=int(max_y),
                                 mask=merged_mask, semantic_type=b.semantic_type,
                                 mask_source="text_segmenter" if safe else "none",
                                 safe_to_inpaint=safe, ocr_eligible=safe, needs_review=not safe)
                result_boxes.append(
                    self._classify(merged, w, h, protect_tail_credits=protect_tail_credits)
                )
                used_text_boxes.update(i for i, _ in inside)
            else:
                fallback = self._flat_bubble_text_fallback(image, b, w, h)
                if fallback is not None:
                    result_boxes.append(fallback)
                else:
                    result_boxes.append(
                        replace(
                            b,
                            safe_to_inpaint=False,
                            ocr_eligible=False,
                            needs_review=True,
                        )
                    )

        standalone = [t for i, t in enumerate(text_boxes) if i not in used_text_boxes]
        result_boxes.extend(self._cluster_free_text_boxes(standalone, w, h))

        recovery_started_at = time.perf_counter()
        recovered = self.recovery.detect(image, existing=result_boxes)
        metrics["mser_ms"] = round(
            (time.perf_counter() - recovery_started_at) * 1000.0, 3
        )
        result_boxes.extend(recovered)
        result_boxes = self._refine_and_split_tall_boxes(result_boxes, image)
        result_boxes = self._apply_final_nms(
            result_boxes, iou_threshold=DETECTOR_FINAL_NMS_IOU
        )
        result = [
            self._classify(b, w, h, protect_tail_credits=protect_tail_credits)
            if b.source_model != "opencv_mser" else b
            for b in result_boxes
        ]
        metrics["result_boxes"] = len(result)
        metrics["review_boxes"] = sum(bool(box.needs_review) for box in result)
        metrics["total_ms"] = round((time.perf_counter() - started_at) * 1000.0, 3)
        self._metrics_local.value = metrics
        return result

    @staticmethod
    def _apply_final_nms(
        boxes: list[BubbleBox],
        iou_threshold: float = DETECTOR_FINAL_NMS_IOU,
    ) -> list[BubbleBox]:
        if not boxes:
            return []
        result: list[BubbleBox] = []
        groups: dict[tuple[str, str], list[BubbleBox]] = {}
        for b in boxes:
            groups.setdefault((b.source_model, b.semantic_type), []).append(b)
        for members in groups.values():
            rects = np.array([[b.x1, b.y1, max(1,b.x2-b.x1), max(1,b.y2-b.y1)] for b in members])
            scores = np.array(
                [max(DETECTOR_NMS_SCORE_FLOOR, float(b.confidence)) for b in members]
            )
            indices = cv2.dnn.NMSBoxes(
                rects.tolist(),
                scores.tolist(),
                DETECTOR_NMS_SCORE_FLOOR,
                iou_threshold,
            )
            result.extend(members[int(i)] for i in np.array(indices).flatten())
        return sorted(result, key=lambda b: b.confidence, reverse=True)

    @staticmethod
    def _refine_and_split_tall_boxes(boxes: list[BubbleBox], img: np.ndarray) -> list[BubbleBox]:
        """Refine geometry without inventing segmentation pixels.

        Detector masks are evidence. A missing mask must stay missing all the way
        to the artwork-safe mask builder; turning ``None`` into a filled rectangle
        here would bypass the downstream safety policy and erase artwork.
        """
        img_h, img_w = img.shape[:2]
        full_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

        refined_boxes = []

        for box in boxes:
            bh = box.y2 - box.y1
            if bh <= DETECTOR_TALL_SPLIT_HEIGHT_THRESHOLD:
                refined_boxes.append(replace(box))
                continue

            crop_x1 = box.x1
            crop_x2 = box.x2
            y1_exp = box.y1
            y2_exp = box.y2

            exp_crop = full_gray[y1_exp:y2_exp, crop_x1:crop_x2]
            if exp_crop.size == 0:
                refined_boxes.append(replace(box))
                continue

            crop_bg = np.percentile(
                exp_crop, DETECTOR_TALL_SPLIT_BACKGROUND_PERCENTILE
            )
            text_rows = exp_crop.mean(axis=1) < (
                crop_bg - DETECTOR_TALL_SPLIT_CONTRAST_DELTA
            )

            line_bounds = []
            in_line = False
            start_y = 0
            for y, is_text in enumerate(text_rows):
                if is_text and not in_line:
                    in_line = True
                    start_y = y
                elif not is_text and in_line:
                    in_line = False
                    if y - start_y >= DETECTOR_TALL_SPLIT_LINE_HEIGHT_MIN:
                        line_bounds.append((start_y, y))

            if (
                in_line
                and (len(text_rows) - start_y)
                >= DETECTOR_TALL_SPLIT_LINE_HEIGHT_MIN
            ):
                line_bounds.append((start_y, len(text_rows)))

            if len(line_bounds) <= 1:
                refined_boxes.append(replace(box))
                continue

            for idx, (ly1, ly2) in enumerate(line_bounds):
                prev_y = line_bounds[idx - 1][1] if idx > 0 else 0
                next_y = line_bounds[idx + 1][0] if idx < len(line_bounds) - 1 else (y2_exp - y1_exp)

                pad_top = min(
                    DETECTOR_TALL_SPLIT_LINE_PADDING_MAX,
                    max(1, (ly1 - prev_y) // 2),
                )
                pad_bot = min(
                    DETECTOR_TALL_SPLIT_LINE_PADDING_MAX,
                    max(1, (next_y - ly2) // 2),
                )

                abs_y1 = max(0, y1_exp + ly1 - pad_top)
                abs_y2 = min(img_h, y1_exp + ly2 + pad_bot)

                line_strip = full_gray[abs_y1:abs_y2, crop_x1:crop_x2]
                if line_strip.size == 0:
                    abs_x1 = max(0, box.x1 - BUBBLE_GROUP_PAD_X)
                    abs_x2 = min(img_w, box.x2 + BUBBLE_GROUP_PAD_X)
                else:
                    col_means = line_strip.mean(axis=0)
                    strip_bg = np.percentile(
                        col_means, DETECTOR_TALL_SPLIT_BACKGROUND_PERCENTILE
                    )
                    text_col_indices = np.where(
                        col_means
                        < (strip_bg - DETECTOR_TALL_SPLIT_CONTRAST_DELTA)
                    )[0]

                    if len(text_col_indices) > 0:
                        abs_x1 = max(
                            box.x1,
                            crop_x1
                            + int(text_col_indices.min())
                            - DETECTOR_TALL_SPLIT_HORIZONTAL_PADDING,
                        )
                        abs_x2 = min(
                            box.x2,
                            crop_x1
                            + int(text_col_indices.max())
                            + DETECTOR_TALL_SPLIT_HORIZONTAL_PADDING,
                        )
                    else:
                        abs_x1 = box.x1
                        abs_x2 = box.x2

                line_h = abs_y2 - abs_y1
                line_w = abs_x2 - abs_x1
                if line_h <= 0 or line_w <= 0:
                    continue

                line_mask = None
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

                refined_boxes.append(
                    replace(box, x1=abs_x1, y1=abs_y1, x2=abs_x2, y2=abs_y2, mask=line_mask)
                )

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
                if (
                    len(cluster) > FREE_TEXT_CLUSTER_SPLIT_COUNT
                    or cluster_h > FREE_TEXT_CLUSTER_HEIGHT_FACTOR * avg_box_h
                ):
                    sub_clusters = self._split_cluster_by_lines(cluster, avg_box_h)
                else:
                    sub_clusters = [cluster]
            else:
                sub_clusters = [cluster]

            for sub in sub_clusters:
                if len(sub) == 1:
                    b = sub[0]
                    min_x = max(0, b.x1 - FREE_TEXT_SINGLE_PAD_X)
                    min_y = max(0, b.y1 - FREE_TEXT_SINGLE_PAD_Y)
                    max_x = min(img_w, b.x2 + FREE_TEXT_SINGLE_PAD_X)
                    max_y = min(img_h, b.y2 + FREE_TEXT_SINGLE_PAD_Y)
                    if max_x <= min_x or max_y <= min_y:
                        continue
                    cluster_area = (max_x - min_x) * (max_y - min_y)
                    if cluster_area > img_w * img_h * MAX_BOX_AREA_RATIO:
                        continue
                    merged_mask = self._merge_masks([b], min_x, min_y, max_x, max_y)
                    final_clusters.append(replace(b, x1=min_x, y1=min_y, x2=max_x, y2=max_y, mask=merged_mask))
                else:
                    min_x = max(0, min(t.x1 for t in sub) - FREE_TEXT_GROUP_PAD_X)
                    min_y = max(0, min(t.y1 for t in sub) - FREE_TEXT_GROUP_PAD_Y)
                    max_x = min(img_w, max(t.x2 for t in sub) + FREE_TEXT_GROUP_PAD_X)
                    max_y = min(img_h, max(t.y2 for t in sub) + FREE_TEXT_GROUP_PAD_Y)
                    if max_x <= min_x or max_y <= min_y:
                        continue
                    cluster_area = (max_x - min_x) * (max_y - min_y)
                    if cluster_area > img_w * img_h * MAX_BOX_AREA_RATIO:
                        continue
                    merged_mask = self._merge_masks(sub, min_x, min_y, max_x, max_y)
                    seed = max(sub, key=lambda t: t.confidence)
                    final_clusters.append(
                        replace(seed, x1=min_x, y1=min_y, x2=max_x, y2=max_y,
                                confidence=max(t.confidence for t in sub), mask=merged_mask)
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
                if (
                    min_h > 0
                    and overlap / min_h > FREE_TEXT_LINE_OVERLAP_MIN
                ):
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
                if group_h > FREE_TEXT_GROUP_HEIGHT_FACTOR * avg_box_h:
                    sub_clusters.append(current_group)
                    current_group = list(line)
                else:
                    current_group.extend(line)
        if current_group:
            sub_clusters.append(current_group)

        return sub_clusters

    @staticmethod
    def _is_free_text_close(a: BubbleBox, b: BubbleBox) -> bool:
        x_overlap = not (
            a.x2 < b.x1 - FREE_TEXT_X_GAP
            or b.x2 < a.x1 - FREE_TEXT_X_GAP
        )
        y_close = (
            abs(a.y1 - b.y1) < FREE_TEXT_Y_ALIGNMENT
            or abs(a.y2 - b.y2) < FREE_TEXT_Y_ALIGNMENT
            or not (
                a.y2 < b.y1 - FREE_TEXT_Y_GAP
                or b.y2 < a.y1 - FREE_TEXT_Y_GAP
            )
        )
        return x_overlap and y_close

    @staticmethod
    def _merge_masks(
        inside_text: list[BubbleBox], min_x: int, min_y: int, max_x: int, max_y: int
    ) -> np.ndarray | None:
        """Merge only segmentation evidence; never synthesize rectangles.

        Missing child masks are ignored. If no child contributes a real non-empty
        mask, return ``None`` so downstream policy can fail safe explicitly.
        """
        merged = np.zeros((max_y - min_y, max_x - min_x), dtype=np.uint8)
        contributed = False
        for t in inside_text:
            tx1 = max(min_x, t.x1)
            ty1 = max(min_y, t.y1)
            tx2 = min(max_x, t.x2)
            ty2 = min(max_y, t.y2)
            if tx2 <= tx1 or ty2 <= ty1 or t.mask is None:
                continue

            mask_x1 = tx1 - t.x1
            mask_y1 = ty1 - t.y1
            mask_x2 = mask_x1 + (tx2 - tx1)
            mask_y2 = mask_y1 + (ty2 - ty1)
            sub_mask = t.mask[mask_y1:mask_y2, mask_x1:mask_x2]
            if sub_mask.size == 0 or not np.any(sub_mask > 0):
                continue

            dest = merged[ty1 - min_y:ty2 - min_y, tx1 - min_x:tx2 - min_x]
            if dest.shape != sub_mask.shape:
                continue
            merged[ty1 - min_y:ty2 - min_y, tx1 - min_x:tx2 - min_x] = np.maximum(dest, sub_mask)
            contributed = True

        return merged if contributed else None

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
            if (
                text_box_area > 0
                and (intersection_area / text_box_area) >= BUBBLE_TEXT_OVERLAP_MIN
            ):
                return True

        return False
