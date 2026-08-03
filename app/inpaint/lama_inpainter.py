import numpy as np
import cv2
from app.config import LAMA_MODEL, INPAINT_SIZE
from app.detector.bubble_detector import BubbleBox
from app.detector.mask_builder import build_mask
from app.ort_utils import make_session

CLUSTER_PADDING = 30
CROP_PADDING = 25


class Inpainter:
    def __init__(self):
        self.session = make_session(LAMA_MODEL)
        self.image_input = self.session.get_inputs()[0].name
        self.mask_input = self.session.get_inputs()[1].name

    def inpaint(self, image: np.ndarray, boxes: list[BubbleBox]) -> np.ndarray:
        if not boxes:
            return image.copy()

        result = image.copy()
        h, w = image.shape[:2]
        clusters = self._cluster_boxes(boxes)

        for cluster in clusters:
            x1 = min(b.x1 for b in cluster)
            y1 = min(b.y1 for b in cluster)
            x2 = max(b.x2 for b in cluster)
            y2 = max(b.y2 for b in cluster)
            crop_box = self._compute_crop_region(x1, y1, x2, y2, w, h)

            cx1, cy1, cx2, cy2 = crop_box
            local_boxes = [
                BubbleBox(b.x1 - cx1, b.y1 - cy1, b.x2 - cx1, b.y2 - cy1, b.confidence, b.mask)
                for b in cluster
            ]
            local_mask = build_mask((cy2 - cy1, cx2 - cx1), local_boxes)

            # Extra dilation to eliminate all text edge outlines & furigana residue
            dil_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            local_mask = cv2.dilate(local_mask, dil_kernel, iterations=1)

            result = self._paint_region(result, local_mask, crop_box)

        return result

    def inpaint_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        ys, xs = np.where(mask > 127)
        if len(ys) == 0:
            return image.copy()

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.dilate(mask, kernel, iterations=1)

        h, w = image.shape[:2]
        ys, xs = np.where(mask > 127)
        crop_box = self._compute_crop_region(
            int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()), w, h
        )
        cx1, cy1, cx2, cy2 = crop_box
        local_mask = mask[cy1:cy2, cx1:cx2]

        result = image.copy()
        return self._paint_region(result, local_mask, crop_box)

    def _paint_region(self, image: np.ndarray, local_mask: np.ndarray, crop_box: tuple) -> np.ndarray:
        cx1, cy1, cx2, cy2 = crop_box
        crop = image[cy1:cy2, cx1:cx2]
        crop_h, crop_w = crop.shape[:2]
        if crop_h < 4 or crop_w < 4:
            return image

        is_flat, fill_color = self._analyze_bg(crop, local_mask)

        if is_flat:
            return self._flat_fill(image, local_mask, crop_box, fill_color)

        return self._lama_fill(image, crop, local_mask, crop_box)

    @staticmethod
    def _analyze_bg(crop: np.ndarray, local_mask: np.ndarray) -> tuple[bool, np.ndarray]:
        bg_mask = local_mask <= 127
        if not np.any(bg_mask):
            return True, np.array([255, 255, 255], dtype=np.uint8)

        bg_pixels = crop[bg_mask]
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray_bg = gray_crop[bg_mask]

        # Ignore extreme dark ink borders when computing median
        non_ink = gray_bg >= 35
        if np.count_nonzero(non_ink) > 10:
            target_pixels = bg_pixels[non_ink]
            target_gray = gray_bg[non_ink]
        else:
            target_pixels = bg_pixels
            target_gray = gray_bg

        median_color = np.median(target_pixels, axis=0).astype(np.uint8)
        median_gray = float(np.median(target_gray))

        # Check proportion of background pixels close to median background intensity
        diffs = np.abs(target_gray.astype(np.float32) - median_gray)
        flat_ratio = float((diffs <= 22.0).mean())

        # Check 10th-90th percentile range of background gray values
        p10, p90 = np.percentile(target_gray, [10, 90])
        trimmed_range = float(p90 - p10)

        # Speech bubble / solid background criteria
        if flat_ratio >= 0.68 or trimmed_range <= 28.0:
            return True, median_color

        return False, median_color

    @staticmethod
    def _flat_fill(image: np.ndarray, local_mask: np.ndarray, crop_box: tuple, fill_color: np.ndarray) -> np.ndarray:
        cx1, cy1, cx2, cy2 = crop_box
        mask_bool = local_mask > 127
        image[cy1:cy2, cx1:cx2][mask_bool] = fill_color
        return image

    def _lama_fill(self, image: np.ndarray, crop: np.ndarray, local_mask: np.ndarray, crop_box: tuple) -> np.ndarray:
        cx1, cy1, cx2, cy2 = crop_box
        crop_h, crop_w = crop.shape[:2]

        crop_resized = cv2.resize(crop, (INPAINT_SIZE, INPAINT_SIZE))
        crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
        mask_resized = cv2.resize(local_mask, (INPAINT_SIZE, INPAINT_SIZE))

        img_blob = crop_rgb.astype(np.float32)
        if img_blob.max() > 1.0:
            img_blob = img_blob / 255.0
        img_blob = img_blob.transpose(2, 0, 1)[None]

        mask_blob = (mask_resized > 127).astype(np.float32)[None, None]

        output = self.session.run(
            None, {self.image_input: img_blob, self.mask_input: mask_blob}
        )[0]

        painted_rgb = output[0].transpose(1, 2, 0)
        if painted_rgb.max() <= 1.0:
            painted_rgb = painted_rgb * 255.0
        painted_rgb = np.clip(painted_rgb, 0, 255).astype(np.uint8)
        painted = cv2.cvtColor(painted_rgb, cv2.COLOR_RGB2BGR)
        painted = cv2.resize(painted, (crop_w, crop_h))

        mask_3d = (local_mask > 127)[:, :, None]
        image[cy1:cy2, cx1:cx2] = np.where(mask_3d, painted, image[cy1:cy2, cx1:cx2])
        return image

    @staticmethod
    def _cluster_boxes(boxes: list[BubbleBox]) -> list[list[BubbleBox]]:
        remaining = list(boxes)
        clusters = []

        while remaining:
            current = [remaining.pop(0)]
            changed = True
            while changed:
                changed = False
                still_remaining = []
                for b in remaining:
                    if any(Inpainter._boxes_close(b, c) for c in current):
                        current.append(b)
                        changed = True
                    else:
                        still_remaining.append(b)
                remaining = still_remaining
            clusters.append(current)

        return clusters

    @staticmethod
    def _boxes_close(a: BubbleBox, b: BubbleBox) -> bool:
        ax1, ay1, ax2, ay2 = a.x1 - CLUSTER_PADDING, a.y1 - CLUSTER_PADDING, a.x2 + CLUSTER_PADDING, a.y2 + CLUSTER_PADDING
        bx1, by1, bx2, by2 = b.x1, b.y1, b.x2, b.y2
        return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)

    @staticmethod
    def _compute_crop_region(x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int) -> tuple:
        x1 -= CROP_PADDING
        y1 -= CROP_PADDING
        x2 += CROP_PADDING
        y2 += CROP_PADDING

        box_w = x2 - x1
        box_h = y2 - y1
        side = max(box_w, box_h)

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        x1 = cx - side / 2
        x2 = cx + side / 2
        y1 = cy - side / 2
        y2 = cy + side / 2

        if x1 < 0:
            x2 = min(img_w, x2 - x1)
            x1 = 0
        if y1 < 0:
            y2 = min(img_h, y2 - y1)
            y1 = 0
        if x2 > img_w:
            x1 = max(0, x1 - (x2 - img_w))
            x2 = img_w
        if y2 > img_h:
            y1 = max(0, y1 - (y2 - img_h))
            y2 = img_h

        return int(x1), int(y1), int(x2), int(y2)
