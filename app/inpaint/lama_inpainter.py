import numpy as np
import cv2
from app.config import LAMA_MODEL, INPAINT_SIZE
from app.detector.bubble_detector import BubbleBox
from app.detector.mask_builder import build_mask
from app.ort_utils import make_session

CLUSTER_PADDING = 35
CROP_PADDING = 35


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

            result = self._smart_paint_region(result, local_mask, crop_box)

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
        return self._smart_paint_region(result, local_mask, crop_box)

    def _smart_paint_region(self, image: np.ndarray, local_mask: np.ndarray, crop_box: tuple) -> np.ndarray:
        cx1, cy1, cx2, cy2 = crop_box
        crop = image[cy1:cy2, cx1:cx2]
        crop_h, crop_w = crop.shape[:2]
        if crop_h < 4 or crop_w < 4:
            return image

        mask_bool = local_mask > 127
        if not np.any(mask_bool):
            return image

        # Extract ring of context pixels around mask (10px ring)
        ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        dilated_mask = cv2.dilate(local_mask, ring_kernel, iterations=1)
        ring_bool = (dilated_mask > 127) & (~mask_bool)

        if np.any(ring_bool):
            ring_pixels = crop[ring_bool]
            gray_ring = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)[ring_bool]

            # Case A: White Speech Bubble / Flat White Background (e.g. >88% pixels > 225)
            white_ratio = float((gray_ring > 225).mean())
            if white_ratio >= 0.85:
                fill_color = np.median(ring_pixels[gray_ring > 225], axis=0).astype(np.uint8) if np.any(gray_ring > 225) else np.array([255, 255, 255], dtype=np.uint8)
                crop[mask_bool] = fill_color
                image[cy1:cy2, cx1:cx2] = crop
                return image

            # Case B: Solid Black Background / Monologue Caption (e.g. >88% pixels < 30)
            black_ratio = float((gray_ring < 30).mean())
            if black_ratio >= 0.85:
                fill_color = np.median(ring_pixels[gray_ring < 30], axis=0).astype(np.uint8) if np.any(gray_ring < 30) else np.array([0, 0, 0], dtype=np.uint8)
                crop[mask_bool] = fill_color
                image[cy1:cy2, cx1:cx2] = crop
                return image

            # Case C: Solid Flat Background (std < 10)
            if float(gray_ring.std()) < 10.0:
                fill_color = np.median(ring_pixels, axis=0).astype(np.uint8)
                crop[mask_bool] = fill_color
                image[cy1:cy2, cx1:cx2] = crop
                return image

        # Case D: Complex Artwork -> Use LaMa Model
        return self._lama_fill(image, crop, local_mask, crop_box)

    def _lama_fill(self, image: np.ndarray, crop: np.ndarray, local_mask: np.ndarray, crop_box: tuple) -> np.ndarray:
        cx1, cy1, cx2, cy2 = crop_box
        crop_h, crop_w = crop.shape[:2]

        scale = INPAINT_SIZE / max(crop_h, crop_w)
        new_h, new_w = int(crop_h * scale), int(crop_w * scale)
        pad_y = (INPAINT_SIZE - new_h) // 2
        pad_x = (INPAINT_SIZE - new_w) // 2

        crop_resized = cv2.resize(crop, (new_w, new_h))
        canvas = np.zeros((INPAINT_SIZE, INPAINT_SIZE, 3), dtype=np.uint8)
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = crop_resized
        crop_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

        mask_resized = cv2.resize(local_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        mask_canvas = np.zeros((INPAINT_SIZE, INPAINT_SIZE), dtype=np.uint8)
        mask_canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = mask_resized

        img_blob = crop_rgb.astype(np.float32) / 255.0
        img_blob = img_blob.transpose(2, 0, 1)[None]

        mask_blob = (mask_canvas > 127).astype(np.float32)[None, None]

        output = self.session.run(
            None, {self.image_input: img_blob, self.mask_input: mask_blob}
        )[0]

        painted_rgb = output[0].transpose(1, 2, 0)
        if painted_rgb.max() <= 1.0:
            painted_rgb = painted_rgb * 255.0
        painted_rgb = np.clip(painted_rgb, 0, 255).astype(np.uint8)
        painted_full = cv2.cvtColor(painted_rgb, cv2.COLOR_RGB2BGR)
        painted_crop = painted_full[pad_y:pad_y + new_h, pad_x:pad_x + new_w]
        painted = cv2.resize(painted_crop, (crop_w, crop_h))

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

        aspect = max(box_w / max(1, box_h), box_h / max(1, box_w))
        if aspect > 1.8:
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_w, x2)
            y2 = min(img_h, y2)
            return int(x1), int(y1), int(x2), int(y2)

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
