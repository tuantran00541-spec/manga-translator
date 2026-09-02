from __future__ import annotations

from dataclasses import replace
import threading

import cv2
import numpy as np

from app.detector.bubble_detector import BubbleBox
from app.parameters import (
    MSER_CONTAINED_SAFE_SKIP_COUNT,
    MSER_CLUSTER_MIN_HEIGHT,
    MSER_CLUSTER_MIN_REGIONS,
    MSER_CLUSTER_MIN_WIDTH,
    MSER_CLUSTER_NEAR_X_FACTOR,
    MSER_CLUSTER_NEAR_Y_FACTOR,
    MSER_DELTA,
    MSER_EXISTING_IOU_SKIP,
    MSER_LINE_GAP_HEIGHT_FACTOR,
    MSER_LINE_GAP_MIN,
    MSER_LINE_OVERLAP_MIN,
    MSER_MAX_AREA,
    MSER_MIN_AREA,
    MSER_PAGE_CLUSTER_SKIP_RATIO,
    MSER_RECOVERY_PAD,
    MSER_REVIEW_CONFIDENCE,
    MSER_RESIDUAL_AREA_RATIO_MAX,
    MSER_RESIDUAL_ASPECT_MAX,
    MSER_RESIDUAL_ASPECT_MIN,
    MSER_RESIDUAL_DISTINCT_X_BUCKET,
    MSER_RESIDUAL_DISTINCT_X_MIN,
    MSER_RESIDUAL_EXISTING_IOU_SKIP,
    MSER_RESIDUAL_FINAL_IOU_SKIP,
    MSER_RESIDUAL_GRID_CELL,
    MSER_RESIDUAL_GRID_Y_RADIUS,
    MSER_RESIDUAL_GROUP_ASPECT_MIN,
    MSER_RESIDUAL_MAX_BBOX_AREA_RATIO,
    MSER_RESIDUAL_MAX_HEIGHT_RATIO,
    MSER_RESIDUAL_MAX_LINE_HEIGHT_RATIO,
    MSER_RESIDUAL_MAX_WIDTH_RATIO,
    MSER_RESIDUAL_MIN_HEIGHT,
    MSER_RESIDUAL_MIN_WIDTH,
    MSER_RESIDUAL_REVIEW_CONFIDENCE,
    MSER_RESIDUAL_SAFE_CENTER_PAD,
    MSER_REGION_AREA_MIN,
    MSER_REGION_AREA_RATIO_MAX,
    MSER_REGION_MAX_HEIGHT_RATIO,
    MSER_REGION_MAX_WIDTH_RATIO,
    MSER_REGION_MIN_SIDE,
    MSER_SAFE_CLUSTER_MIN_REGIONS,
    MSER_SAFE_CONFIDENCE,
    MSER_SAFE_MASK_RATIO_MAX,
    MSER_SAFE_MASK_RATIO_MIN,
    MSER_SAFE_PAGE_AREA_RATIO_MAX,
    MSER_SEED_CANNY_HIGH,
    MSER_SEED_CANNY_LOW,
    MSER_SEED_CONTRAST_DELTA,
    WATERMARK_EDGE_X_RATIO,
    WATERMARK_EDGE_Y_RATIO,
    WATERMARK_HORIZONTAL_ASPECT_MIN,
    WATERMARK_HORIZONTAL_HEIGHT_RATIO_MAX,
    WATERMARK_VERTICAL_ASPECT_MIN,
    WATERMARK_VERTICAL_HEIGHT_RATIO_MAX,
)


class SecondaryTextRecovery:
    """Conservative OpenCV/MSER recovery for text styles missed by the segmenter.

    Recovery is deliberately detection-first, not cleanup-first. Candidates are
    review-only unless a compact pixel mask can be reconstructed with conservative
    geometry. This prevents an outlined/SFX proposal from turning into rectangle
    inpainting while still ensuring detector misses are visible to the editor.
    """

    def __init__(self) -> None:
        self._mser = cv2.MSER_create(MSER_DELTA, MSER_MIN_AREA, MSER_MAX_AREA)
        self._mser_lock = threading.Lock()

    @staticmethod
    def _iou(a: BubbleBox, b: BubbleBox) -> float:
        x1, y1 = max(a.x1, b.x1), max(a.y1, b.y1)
        x2, y2 = min(a.x2, b.x2), min(a.y2, b.y2)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        aa = max(1, (a.x2 - a.x1) * (a.y2 - a.y1))
        bb = max(1, (b.x2 - b.x1) * (b.y2 - b.y1))
        return inter / float(aa + bb - inter)

    @staticmethod
    def _watermark_like(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> bool:
        bw, bh = x2 - x1, y2 - y1
        aspect = bw / max(1.0, float(bh))
        edge_touch = x1 < w * WATERMARK_EDGE_X_RATIO or x2 > w * (1.0 - WATERMARK_EDGE_X_RATIO)
        vertical_edge = y1 < h * WATERMARK_EDGE_Y_RATIO or y2 > h * (1.0 - WATERMARK_EDGE_Y_RATIO)
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
    def _seed_mask(gray_crop: np.ndarray) -> np.ndarray:
        if gray_crop.size == 0:
            return np.zeros_like(gray_crop, dtype=np.uint8)
        blur = cv2.GaussianBlur(gray_crop, (3, 3), 0)
        med = float(np.median(blur))
        dark = blur < max(0.0, med - MSER_SEED_CONTRAST_DELTA)
        light = blur > min(255.0, med + MSER_SEED_CONTRAST_DELTA)
        edges = cv2.Canny(blur, MSER_SEED_CANNY_LOW, MSER_SEED_CANNY_HIGH) > 0
        seed = (dark | light) & (
            cv2.dilate(
                edges.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1
            )
            > 0
        )
        mask = seed.astype(np.uint8) * 255
        if np.any(mask):
            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_CLOSE,
                np.ones((3, 3), np.uint8),
                iterations=1,
            )
        return mask

    @staticmethod
    def _line_neighbors(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
        ah, bh = a[3] - a[1], b[3] - b[1]
        overlap = min(a[3], b[3]) - max(a[1], b[1])
        min_h = min(ah, bh)
        if min_h <= 0 or overlap / float(min_h) < MSER_LINE_OVERLAP_MIN:
            return False
        horizontal_gap = max(0, max(a[0], b[0]) - min(a[2], b[2]))
        return horizontal_gap <= max(
            MSER_LINE_GAP_MIN,
            MSER_LINE_GAP_HEIGHT_FACTOR * max(ah, bh),
        )

    @classmethod
    def _residual_line_candidates(
        cls,
        raw_boxes: np.ndarray,
        image_shape: tuple[int, int],
        existing: list[BubbleBox],
    ) -> list[BubbleBox]:
        """Surface compact MSER text lines missed by the primary segmenter.

        The legacy MSER agglomeration intentionally allows multi-line clusters,
        but on dense webtoon art its distance threshold grows with cluster height
        and can chain thousands of regions into one page-sized cluster.  That
        cluster is then discarded as too large, silently hiding a real miss.

        This second pass only creates *review-only* line proposals.  It filters
        out page-scale MSER regions, groups locally aligned glyph-sized regions,
        and never manufactures an inpaint mask.  Its purpose is therefore to
        prevent a content-heavy miss from being labelled ``verified`` without
        adding a new destructive cleanup path.
        """
        h, w = image_shape
        if raw_boxes is None or len(raw_boxes) == 0 or h <= 0 or w <= 0:
            return []

        rects: list[tuple[int, int, int, int]] = []
        seen: set[tuple[int, int, int, int]] = set()
        page_area = float(max(1, w * h))
        for x, y, bw, bh in np.asarray(raw_boxes).reshape(-1, 4):
            x, y, bw, bh = map(int, (x, y, bw, bh))
            area = bw * bh
            if (
                bw < MSER_REGION_MIN_SIDE
                or bh < MSER_REGION_MIN_SIDE
                or bw > w * MSER_RESIDUAL_MAX_WIDTH_RATIO
                or bh > h * MSER_RESIDUAL_MAX_HEIGHT_RATIO
            ):
                continue
            if (
                area < MSER_REGION_AREA_MIN
                or area > page_area * MSER_RESIDUAL_AREA_RATIO_MAX
            ):
                continue
            aspect = bw / max(1.0, float(bh))
            if aspect < MSER_RESIDUAL_ASPECT_MIN or aspect > MSER_RESIDUAL_ASPECT_MAX:
                continue
            key = (x, y, bw, bh)
            if key in seen:
                continue
            seen.add(key)
            rects.append((x, y, x + bw, y + bh))

        if not rects:
            return []

        parent = list(range(len(rects)))
        rank = [0] * len(rects)

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            if rank[ra] < rank[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            if rank[ra] == rank[rb]:
                rank[ra] += 1

        cell = MSER_RESIDUAL_GRID_CELL
        grid: dict[tuple[int, int], list[int]] = {}
        for i, rect in enumerate(rects):
            cx = (rect[0] + rect[2]) // 2
            cy = (rect[1] + rect[3]) // 2
            gx, gy = cx // cell, cy // cell
            reach = max(
                1,
                int(
                    np.ceil(
                        max(
                            MSER_LINE_GAP_MIN,
                            MSER_LINE_GAP_HEIGHT_FACTOR * (rect[3] - rect[1]),
                        )
                        / cell
                    )
                )
                + 1,
            )
            for xx in range(gx - reach, gx + reach + 1):
                for yy in range(
                    gy - MSER_RESIDUAL_GRID_Y_RADIUS,
                    gy + MSER_RESIDUAL_GRID_Y_RADIUS + 1,
                ):
                    for j in grid.get((xx, yy), ()):
                        if cls._line_neighbors(rect, rects[j]):
                            union(i, j)
            grid.setdefault((gx, gy), []).append(i)

        groups: dict[int, list[tuple[int, int, int, int]]] = {}
        for i, rect in enumerate(rects):
            groups.setdefault(find(i), []).append(rect)

        out: list[BubbleBox] = []
        for group in groups.values():
            x1 = min(r[0] for r in group)
            y1 = min(r[1] for r in group)
            x2 = max(r[2] for r in group)
            y2 = max(r[3] for r in group)
            bw, bh = x2 - x1, y2 - y1
            distinct_x = len(
                {
                    int(
                        ((r[0] + r[2]) / 2.0)
                        // MSER_RESIDUAL_DISTINCT_X_BUCKET
                    )
                    for r in group
                }
            )
            if (
                distinct_x < MSER_RESIDUAL_DISTINCT_X_MIN
                or bw < MSER_RESIDUAL_MIN_WIDTH
                or bh < MSER_RESIDUAL_MIN_HEIGHT
            ):
                continue
            if (
                bh > h * MSER_RESIDUAL_MAX_LINE_HEIGHT_RATIO
                or (bw * bh) > page_area * MSER_RESIDUAL_MAX_BBOX_AREA_RATIO
            ):
                continue
            if bw / max(1.0, float(bh)) < MSER_RESIDUAL_GROUP_ASPECT_MIN:
                continue

            px1, py1 = max(0, x1 - MSER_RECOVERY_PAD), max(
                0, y1 - MSER_RECOVERY_PAD
            )
            px2, py2 = min(w, x2 + MSER_RECOVERY_PAD), min(
                h, y2 + MSER_RECOVERY_PAD
            )
            candidate = BubbleBox(
                px1,
                py1,
                px2,
                py2,
                MSER_RESIDUAL_REVIEW_CONFIDENCE,
                None,
                source_model="opencv_mser",
                class_id=0,
                class_name="text_recovery",
                semantic_type="free_text",
                mask_source="none",
                safe_to_inpaint=False,
                ocr_eligible=False,
                needs_review=True,
            )
            if any(
                cls._iou(candidate, box) > MSER_RESIDUAL_EXISTING_IOU_SKIP
                for box in existing
            ):
                continue
            cx, cy = (px1 + px2) / 2.0, (py1 + py2) / 2.0
            if any(
                box.safe_to_inpaint
                and box.x1 - MSER_RESIDUAL_SAFE_CENTER_PAD
                <= cx
                <= box.x2 + MSER_RESIDUAL_SAFE_CENTER_PAD
                and box.y1 - MSER_RESIDUAL_SAFE_CENTER_PAD
                <= cy
                <= box.y2 + MSER_RESIDUAL_SAFE_CENTER_PAD
                for box in existing
            ):
                continue
            out.append(candidate)

        return out

    def detect(self, image: np.ndarray, existing: list[BubbleBox] | None = None) -> list[BubbleBox]:
        if image is None or image.size == 0:
            return []
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        with self._mser_lock:
            _, boxes = self._mser.detectRegions(gray)
        if boxes is None or len(boxes) == 0:
            return []

        rects = []
        for x, y, bw, bh in np.asarray(boxes).reshape(-1, 4):
            x, y, bw, bh = map(int, (x, y, bw, bh))
            if (
                bw < MSER_REGION_MIN_SIDE
                or bh < MSER_REGION_MIN_SIDE
                or bw > w * MSER_REGION_MAX_WIDTH_RATIO
                or bh > h * MSER_REGION_MAX_HEIGHT_RATIO
            ):
                continue
            area = bw * bh
            if (
                area < MSER_REGION_AREA_MIN
                or area > w * h * MSER_REGION_AREA_RATIO_MAX
            ):
                continue
            rects.append((x, y, x + bw, y + bh))
        if not rects:
            return []

        remaining = rects[:]
        clusters: list[list[tuple[int, int, int, int]]] = []
        while remaining:
            cluster = [remaining.pop()]
            changed = True
            while changed:
                changed = False
                keep = []
                cx1 = min(r[0] for r in cluster)
                cy1 = min(r[1] for r in cluster)
                cx2 = max(r[2] for r in cluster)
                cy2 = max(r[3] for r in cluster)
                ch = max(8, cy2 - cy1)
                for r in remaining:
                    rx1, ry1, rx2, ry2 = r
                    near_x = not (
                        rx1 > cx2 + ch * MSER_CLUSTER_NEAR_X_FACTOR
                        or rx2 < cx1 - ch * MSER_CLUSTER_NEAR_X_FACTOR
                    )
                    near_y = not (
                        ry1 > cy2 + ch * MSER_CLUSTER_NEAR_Y_FACTOR
                        or ry2 < cy1 - ch * MSER_CLUSTER_NEAR_Y_FACTOR
                    )
                    if near_x and near_y:
                        cluster.append(r)
                        changed = True
                    else:
                        keep.append(r)
                remaining = keep
            clusters.append(cluster)

        out: list[BubbleBox] = []
        existing = existing or []
        for cluster in clusters:
            if len(cluster) < MSER_CLUSTER_MIN_REGIONS:
                continue
            x1 = max(0, min(r[0] for r in cluster) - MSER_RECOVERY_PAD)
            y1 = max(0, min(r[1] for r in cluster) - MSER_RECOVERY_PAD)
            x2 = min(w, max(r[2] for r in cluster) + MSER_RECOVERY_PAD)
            y2 = min(h, max(r[3] for r in cluster) + MSER_RECOVERY_PAD)
            bw, bh = x2 - x1, y2 - y1
            if bw < MSER_CLUSTER_MIN_WIDTH or bh < MSER_CLUSTER_MIN_HEIGHT:
                continue
            candidate = BubbleBox(
                x1,
                y1,
                x2,
                y2,
                MSER_REVIEW_CONFIDENCE,
                None,
                source_model="opencv_mser",
                class_id=0,
                class_name="text_recovery",
                semantic_type="free_text",
                mask_source="none",
                safe_to_inpaint=False,
                ocr_eligible=False,
                needs_review=True,
            )
            if any(self._iou(candidate, b) > MSER_EXISTING_IOU_SKIP for b in existing):
                continue
            contained_verified = 0
            for b in existing:
                if not b.safe_to_inpaint:
                    continue
                cx = (b.x1 + b.x2) / 2.0
                cy = (b.y1 + b.y2) / 2.0
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    contained_verified += 1
            if contained_verified >= MSER_CONTAINED_SAFE_SKIP_COUNT:
                continue

            watermark = self._watermark_like(x1, y1, x2, y2, w, h)
            crop = gray[y1:y2, x1:x2]
            mask = self._seed_mask(crop)
            ratio = float(np.count_nonzero(mask)) / float(max(1, mask.size))
            page_ratio = (bw * bh) / float(max(1, w * h))
            if page_ratio > MSER_PAGE_CLUSTER_SKIP_RATIO and any(
                b.safe_to_inpaint for b in existing
            ):
                continue
            safe = bool(
                not watermark
                and MSER_SAFE_MASK_RATIO_MIN <= ratio <= MSER_SAFE_MASK_RATIO_MAX
                and page_ratio <= MSER_SAFE_PAGE_AREA_RATIO_MAX
                and len(cluster) >= MSER_SAFE_CLUSTER_MIN_REGIONS
            )
            if safe:
                candidate = replace(
                    candidate,
                    mask=mask,
                    mask_source="opencv_mser",
                    safe_to_inpaint=True,
                    ocr_eligible=True,
                    needs_review=False,
                    confidence=MSER_SAFE_CONFIDENCE,
                )
            elif watermark:
                candidate = replace(
                    candidate, semantic_type="watermark", class_name="watermark"
                )
            out.append(candidate)

        verification_set = existing + out
        if verification_set and all(
            box.safe_to_inpaint and not box.needs_review for box in verification_set
        ):
            residual = self._residual_line_candidates(
                np.asarray(boxes), (h, w), verification_set
            )
            for candidate in residual:
                if any(
                    self._iou(candidate, box) > MSER_RESIDUAL_FINAL_IOU_SKIP
                    for box in out
                ):
                    continue
                out.append(candidate)
        return out
