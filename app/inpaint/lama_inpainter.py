import numpy as np
import cv2
from app.config import LAMA_MODEL, INPAINT_SIZE
from app.detector.bubble_detector import BubbleBox, MAX_BOX_AREA_RATIO
from app.detector.mask_builder import build_mask
from app.logging_config import logger
from app.ort_utils import make_session

CLUSTER_PADDING = 35
CROP_PADDING = 35
MANUAL_CROP_PADDING = 72
MANUAL_MIN_DILATION = 9
MANUAL_MAX_DILATION = 15
MANUAL_FEATHER_RADIUS = 3
MANUAL_TILE_OVERLAP = 64


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

            if len(cluster) > 1 and (x2 - x1) * (y2 - y1) > w * h * MAX_BOX_AREA_RATIO:
                logger.warning(f"Skipping multi-box cluster ({len(cluster)} boxes) at ({x1}, {y1}, {x2}, {y2}): area exceeds MAX_BOX_AREA_RATIO")
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
        if mask is None or not np.any(mask > 127):
            return image.copy()

        binary_mask = (mask > 127).astype(np.uint8) * 255
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

        result = image.copy()
        h, w = image.shape[:2]

        for label in range(1, num_labels):
            component_mask = (labels == label).astype(np.uint8) * 255
            ys, xs = np.where(component_mask > 127)
            if len(ys) == 0:
                continue

            bbox_w = int(xs.max() - xs.min() + 1)
            bbox_h = int(ys.max() - ys.min() + 1)
            scale = max(1, min(bbox_w, bbox_h))
            kernel_size = int(np.clip(round(scale * 0.025) * 2 + 1, MANUAL_MIN_DILATION, MANUAL_MAX_DILATION))
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            dilated_comp = cv2.dilate(component_mask, kernel, iterations=1)

            dys, dxs = np.where(dilated_comp > 127)
            if len(dys) == 0:
                continue
            crop_box = self._compute_manual_crop_region(
                int(dxs.min()), int(dys.min()), int(dxs.max()), int(dys.max()), w, h
            )
            cx1, cy1, cx2, cy2 = crop_box
            local_mask = dilated_comp[cy1:cy2, cx1:cx2]

            result = self._smart_paint_region(result, local_mask, crop_box, feather=True)

        return result

    def _smart_paint_region(self, image: np.ndarray, local_mask: np.ndarray, crop_box: tuple, feather: bool = False) -> np.ndarray:
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

            white_mask = gray_non_mask > 215
            if float(white_mask.mean()) >= 0.70:
                fill_color = np.median(non_mask_pixels[white_mask], axis=0).astype(np.uint8) if np.any(white_mask) else np.array([255, 255, 255], dtype=np.uint8)
                crop[mask_bool] = fill_color
                image[cy1:cy2, cx1:cx2] = crop
                return image

            black_mask = gray_non_mask < 35
            if float(black_mask.mean()) >= 0.70:
                fill_color = np.median(non_mask_pixels[black_mask], axis=0).astype(np.uint8) if np.any(black_mask) else np.array([0, 0, 0], dtype=np.uint8)
                crop[mask_bool] = fill_color
                image[cy1:cy2, cx1:cx2] = crop
                return image

            if float(gray_non_mask.std()) < 12.0:
                fill_color = np.median(non_mask_pixels, axis=0).astype(np.uint8)
                crop[mask_bool] = fill_color
                image[cy1:cy2, cx1:cx2] = crop
                return image

        return self._lama_fill(image, crop, local_mask, crop_box, feather=feather)

    def _lama_fill(self, image: np.ndarray, crop: np.ndarray, local_mask: np.ndarray, crop_box: tuple, feather: bool = False) -> np.ndarray:
        cx1, cy1, cx2, cy2 = crop_box
        crop_h, crop_w = crop.shape[:2]

        # Manual repairs can become long after the safe physical slicer keeps
        # artwork intact. Feeding the whole long crop through a single 512x512
        # resize destroys local detail. Keep the existing single-pass behavior
        # for automatic inpaint, but tile oversized manual crops at native
        # resolution so LaMa does not have to reconstruct downsampled artwork.
        if feather and max(crop_h, crop_w) > INPAINT_SIZE:
            painted = self._lama_fill_tiled(crop, local_mask)
        else:
            painted = self._lama_fill_single(crop, local_mask)

        original_crop = image[cy1:cy2, cx1:cx2]
        if feather:
            alpha = (local_mask > 127).astype(np.float32)
            k = MANUAL_FEATHER_RADIUS * 2 + 1
            alpha = cv2.GaussianBlur(alpha, (k, k), 0)
            alpha = np.clip(alpha, 0.0, 1.0)[:, :, None]
            blended = painted.astype(np.float32) * alpha + original_crop.astype(np.float32) * (1.0 - alpha)
            image[cy1:cy2, cx1:cx2] = np.clip(blended, 0, 255).astype(np.uint8)
        else:
            mask_3d = (local_mask > 127)[:, :, None]
            image[cy1:cy2, cx1:cx2] = np.where(mask_3d, painted, original_crop)
        return image

    def _lama_fill_single(self, crop: np.ndarray, local_mask: np.ndarray) -> np.ndarray:
        crop_h, crop_w = crop.shape[:2]
        scale = INPAINT_SIZE / max(crop_h, crop_w)
        new_h = max(1, int(round(crop_h * scale)))
        new_w = max(1, int(round(crop_w * scale)))
        pad_y = (INPAINT_SIZE - new_h) // 2
        pad_x = (INPAINT_SIZE - new_w) // 2

        crop_resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC)
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

        output = self.session.run(None, {self.image_input: img_blob, self.mask_input: mask_blob})[0]
        painted_rgb = output[0].transpose(1, 2, 0)
        if painted_rgb.max() <= 1.0:
            painted_rgb = painted_rgb * 255.0
        painted_rgb = np.clip(painted_rgb, 0, 255).astype(np.uint8)
        painted_full = cv2.cvtColor(painted_rgb, cv2.COLOR_RGB2BGR)
        painted_crop = painted_full[pad_y:pad_y + new_h, pad_x:pad_x + new_w]
        return cv2.resize(painted_crop, (crop_w, crop_h), interpolation=cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA)

    def _lama_fill_tiled(self, crop: np.ndarray, local_mask: np.ndarray) -> np.ndarray:
        """Run manual LaMa on overlapping native-resolution 512px tiles."""
        h, w = crop.shape[:2]
        tile = INPAINT_SIZE
        overlap = min(MANUAL_TILE_OVERLAP, tile // 4)
        step = tile - overlap

        output = np.zeros((h, w, 3), dtype=np.float32)
        weights = np.zeros((h, w), dtype=np.float32)

        y_starts = self._tile_starts(h, tile, step)
        x_starts = self._tile_starts(w, tile, step)
        for y0 in y_starts:
            y1 = min(h, y0 + tile)
            for x0 in x_starts:
                x1 = min(w, x0 + tile)
                tile_img = crop[y0:y1, x0:x1]
                tile_mask = local_mask[y0:y1, x0:x1]
                tile_h, tile_w = tile_img.shape[:2]
                wy = self._tile_weight(tile_h, overlap, y0 > 0, y1 < h)
                wx = self._tile_weight(tile_w, overlap, x0 > 0, x1 < w)
                weight = wy[:, None] * wx[None, :]

                if not np.any(tile_mask > 127):
                    output[y0:y1, x0:x1] += tile_img.astype(np.float32) * weight[:, :, None]
                    weights[y0:y1, x0:x1] += weight
                    continue

                tile_painted = self._lama_fill_single(tile_img, tile_mask)
                output[y0:y1, x0:x1] += tile_painted.astype(np.float32) * weight[:, :, None]
                weights[y0:y1, x0:x1] += weight

        weights = np.maximum(weights, 1e-6)
        return np.clip(output / weights[:, :, None], 0, 255).astype(np.uint8)

    @staticmethod
    def _tile_starts(length: int, tile: int, step: int) -> list[int]:
        if length <= tile:
            return [0]
        starts = list(range(0, max(1, length - tile + 1), step))
        last = length - tile
        if starts[-1] != last:
            starts.append(last)
        return starts

    @staticmethod
    def _tile_weight(length: int, overlap: int, has_before: bool, has_after: bool) -> np.ndarray:
        weight = np.ones(length, dtype=np.float32)
        if overlap <= 0:
            return weight
        ramp = np.linspace(0.0, 1.0, min(overlap, length), dtype=np.float32)
        if has_before:
            weight[:len(ramp)] = np.maximum(weight[:len(ramp)], ramp)
        if has_after:
            weight[-len(ramp):] = np.maximum(weight[-len(ramp):], ramp[::-1])
        return weight

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
                    if any(Inpainter._boxes_close(b, c) for c in current) and Inpainter._can_add_to_cluster(current, b, 600):
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
    def _can_add_to_cluster(cluster: list[BubbleBox], b: BubbleBox, max_dim: int = 600) -> bool:
        x1 = min(min(box.x1 for box in cluster), b.x1)
        y1 = min(min(box.y1 for box in cluster), b.y1)
        x2 = max(max(box.x2 for box in cluster), b.x2)
        y2 = max(max(box.y2 for box in cluster), b.y2)
        return (x2 - x1) <= max_dim and (y2 - y1) <= max_dim

    @staticmethod
    def _compute_manual_crop_region(x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int) -> tuple:
        """Give manual repairs more artwork context than automatic bubble repairs."""
        x1 = max(0, x1 - MANUAL_CROP_PADDING)
        y1 = max(0, y1 - MANUAL_CROP_PADDING)
        x2 = min(img_w, x2 + MANUAL_CROP_PADDING)
        y2 = min(img_h, y2 + MANUAL_CROP_PADDING)
        return int(x1), int(y1), int(x2), int(y2)

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
