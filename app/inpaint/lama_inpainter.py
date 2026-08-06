import numpy as np
import cv2
from app.config import LAMA_MODEL, INPAINT_SIZE
from app.detector.bubble_detector import BubbleBox, MAX_BOX_AREA_RATIO
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

            if (x2 - x1) * (y2 - y1) > w * h * MAX_BOX_AREA_RATIO:
                continue

            crop_box = self._compute_crop_region(x1, y1, x2, y2, w, h)

            cx1, cy1, cx2, cy2 = crop_box
            local_boxes = [
                BubbleBox(b.x1 - cx1, b.y1 - cy1, b.x2 - cx1, b.y2 - cy1, b.confidence, b.mask)
                for b in cluster
            ]
            crop_img = image[cy1:cy2, cx1:cx2]
            local_mask = build_mask((cy2 - cy1, cx2 - cx1), local_boxes, crop_img)

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

        non_mask_bool = ~mask_bool
        if np.any(non_mask_bool):
            non_mask_pixels = crop[non_mask_bool]
            gray_non_mask = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)[non_mask_bool]

            # Case A: White Speech Bubble (>= 70% of non-mask context pixels are bright > 215)
            white_mask = gray_non_mask > 215
            if float(white_mask.mean()) >= 0.70:
                fill_color = np.median(non_mask_pixels[white_mask], axis=0).astype(np.uint8) if np.any(white_mask) else np.array([255, 255, 255], dtype=np.uint8)
                crop[mask_bool] = fill_color
                image[cy1:cy2, cx1:cx2] = crop
                return image

            # Case B: Solid Black Monologue / Dark Box (>= 70% of non-mask context pixels are dark < 35)
            black_mask = gray_non_mask < 35
            if float(black_mask.mean()) >= 0.70:
                fill_color = np.median(non_mask_pixels[black_mask], axis=0).astype(np.uint8) if np.any(black_mask) else np.array([0, 0, 0], dtype=np.uint8)
                crop[mask_bool] = fill_color
                image[cy1:cy2, cx1:cx2] = crop
                return image

            # Case C: Uniform Flat Background (stddev < 12.0)
            if float(gray_non_mask.std()) < 12.0:
                fill_color = np.median(non_mask_pixels, axis=0).astype(np.uint8)
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
        pad_top = pad_y
        pad_bottom = INPAINT_SIZE - new_h - pad_y
        pad_left = pad_x
        pad_right = INPAINT_SIZE - new_w - pad_x

        canvas = cv2.copyMakeBorder(
            crop_resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REPLICATE
        )
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
        raw_clusters = []

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
            raw_clusters.append(current)

        final_clusters = []
        for cluster in raw_clusters:
            if len(cluster) > 1:
                avg_h = sum(b.y2 - b.y1 for b in cluster) / len(cluster)
                cluster_h = max(b.y2 for b in cluster) - min(b.y1 for b in cluster)
                if len(cluster) > 3 or cluster_h > 4.0 * avg_h:
                    sub_clusters = Inpainter._split_cluster_lines(cluster, avg_h)
                    final_clusters.extend(sub_clusters)
                else:
                    final_clusters.append(cluster)
            else:
                final_clusters.append(cluster)

        return final_clusters

    @staticmethod
    def _split_cluster_lines(cluster: list[BubbleBox], avg_h: float) -> list[list[BubbleBox]]:
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
                if group_h > 3.0 * avg_h:
                    sub_clusters.append(current_group)
                    current_group = list(line)
                else:
                    current_group.extend(line)
        if current_group:
            sub_clusters.append(current_group)

        return sub_clusters

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
