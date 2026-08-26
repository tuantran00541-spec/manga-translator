from __future__ import annotations

from dataclasses import replace
import threading

import cv2
import numpy as np

from app.detector.bubble_detector import BubbleBox


class SecondaryTextRecovery:
    """Conservative OpenCV/MSER recovery for text styles missed by the segmenter.

    Recovery is deliberately detection-first, not cleanup-first. Candidates are
    review-only unless a compact pixel mask can be reconstructed with conservative
    geometry. This prevents an outlined/SFX proposal from turning into rectangle
    inpainting while still ensuring detector misses are visible to the editor.
    """

    def __init__(self) -> None:
        self._mser = cv2.MSER_create(5, 18, 120000)
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
        edge_touch = x1 < w * 0.04 or x2 > w * 0.96
        vertical_edge = y1 < h * 0.05 or y2 > h * 0.95
        return bool((edge_touch and aspect >= 4.0 and bh <= h * 0.12) or
                    (vertical_edge and aspect >= 6.0 and bh <= h * 0.08))

    @staticmethod
    def _seed_mask(gray_crop: np.ndarray) -> np.ndarray:
        if gray_crop.size == 0:
            return np.zeros_like(gray_crop, dtype=np.uint8)
        blur = cv2.GaussianBlur(gray_crop, (3, 3), 0)
        med = float(np.median(blur))
        dark = blur < max(0.0, med - 22.0)
        light = blur > min(255.0, med + 22.0)
        edges = cv2.Canny(blur, 45, 120) > 0
        seed = (dark | light) & (cv2.dilate(edges.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0)
        mask = seed.astype(np.uint8) * 255
        if np.any(mask):
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
        return mask

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
            if bw < 4 or bh < 4 or bw > w * 0.9 or bh > h * 0.65:
                continue
            area = bw * bh
            if area < 24 or area > w * h * 0.20:
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
                cx1 = min(r[0] for r in cluster); cy1 = min(r[1] for r in cluster)
                cx2 = max(r[2] for r in cluster); cy2 = max(r[3] for r in cluster)
                ch = max(8, cy2 - cy1)
                for r in remaining:
                    rx1, ry1, rx2, ry2 = r
                    near_x = not (rx1 > cx2 + ch * 1.8 or rx2 < cx1 - ch * 1.8)
                    near_y = not (ry1 > cy2 + ch * 1.3 or ry2 < cy1 - ch * 1.3)
                    if near_x and near_y:
                        cluster.append(r); changed = True
                    else:
                        keep.append(r)
                remaining = keep
            clusters.append(cluster)

        out: list[BubbleBox] = []
        existing = existing or []
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            x1 = max(0, min(r[0] for r in cluster) - 6)
            y1 = max(0, min(r[1] for r in cluster) - 6)
            x2 = min(w, max(r[2] for r in cluster) + 6)
            y2 = min(h, max(r[3] for r in cluster) + 6)
            bw, bh = x2 - x1, y2 - y1
            if bw < 12 or bh < 10:
                continue
            candidate = BubbleBox(x1, y1, x2, y2, 0.20, None,
                                  source_model="opencv_mser", class_id=0,
                                  class_name="text_recovery", semantic_type="free_text",
                                  mask_source="none", safe_to_inpaint=False,
                                  ocr_eligible=False, needs_review=True)
            if any(self._iou(candidate, b) > 0.55 for b in existing):
                continue
            contained_verified = 0
            for b in existing:
                if not b.safe_to_inpaint:
                    continue
                cx = (b.x1 + b.x2) / 2.0; cy = (b.y1 + b.y2) / 2.0
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    contained_verified += 1
            if contained_verified >= 2:
                continue

            watermark = self._watermark_like(x1, y1, x2, y2, w, h)
            crop = gray[y1:y2, x1:x2]
            mask = self._seed_mask(crop)
            ratio = float(np.count_nonzero(mask)) / float(max(1, mask.size))
            page_ratio = (bw * bh) / float(max(1, w * h))
            if page_ratio > 0.45 and any(b.safe_to_inpaint for b in existing):
                continue
            safe = bool(not watermark and 0.015 <= ratio <= 0.42 and page_ratio <= 0.035 and len(cluster) >= 3)
            if safe:
                candidate = replace(candidate, mask=mask, mask_source="opencv_mser",
                                    safe_to_inpaint=True, ocr_eligible=True,
                                    needs_review=False, confidence=0.35)
            elif watermark:
                candidate = replace(candidate, semantic_type="watermark", class_name="watermark")
            out.append(candidate)
        return out
