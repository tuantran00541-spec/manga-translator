import numpy as np
import cv2
from dataclasses import dataclass
from app.config import BUBBLE_IOU_THRESHOLD, ENABLE_TTA
from app.ort_utils import make_session

INPUT_SIZE = 1024
SLICE_OVERLAP = 200
MAX_BOX_WIDTH_RATIO = 0.97
MAX_BOX_AREA_RATIO = 0.35
MAX_ASPECT_RATIO = 25


@dataclass
class BubbleBox:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    mask: np.ndarray | None = None


class YoloDetector:
    def __init__(self, model_path, conf_threshold: float, use_tta: bool | None = None):
        self.session = make_session(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.conf_threshold = conf_threshold
        self.use_tta = ENABLE_TTA if use_tta is None else use_tta

    def detect(self, image: np.ndarray) -> list[BubbleBox]:
        h, w = image.shape[:2]
        if h <= INPUT_SIZE * 1.5:
            boxes = self._detect_single(image, 0, 0)
        else:
            all_boxes = []
            step = INPUT_SIZE - SLICE_OVERLAP
            y = 0
            while y < h:
                slice_h = min(INPUT_SIZE, h - y)
                slice_img = image[y:y + slice_h, :]
                boxes = self._detect_single(slice_img, 0, y)
                all_boxes.extend(boxes)
                if y + slice_h >= h:
                    break
                y += step
            boxes = self._nms_boxes(all_boxes)

        return self._filter_invalid(boxes, w, h)

    @staticmethod
    def _filter_invalid(boxes: list[BubbleBox], img_w: int, img_h: int) -> list[BubbleBox]:
        result = []
        page_area = img_w * img_h
        for b in boxes:
            box_w = b.x2 - b.x1
            box_h = b.y2 - b.y1
            if box_w <= 0 or box_h <= 0:
                continue
            if box_w > img_w * MAX_BOX_WIDTH_RATIO:
                continue
            if (box_w * box_h) > page_area * MAX_BOX_AREA_RATIO:
                continue
            aspect = box_w / box_h
            if aspect > MAX_ASPECT_RATIO or aspect < 1 / MAX_ASPECT_RATIO:
                continue
            result.append(b)
        return result

    def _detect_single(self, image: np.ndarray, offset_x: int, offset_y: int) -> list[BubbleBox]:
        if self.use_tta:
            return self._detect_single_tta(image, offset_x, offset_y)
        return self._detect_single_plain(image, offset_x, offset_y)

    def _detect_single_plain(self, image: np.ndarray, offset_x: int, offset_y: int) -> list[BubbleBox]:
        h, w = image.shape[:2]
        blob, scale, pad = self._preprocess(image)
        if blob is None:
            return []
        outputs = self.session.run(None, {self.input_name: blob})
        boxes = self._postprocess(outputs, scale, pad, w, h)
        if offset_x or offset_y:
            boxes = [
                BubbleBox(b.x1 + offset_x, b.y1 + offset_y, b.x2 + offset_x, b.y2 + offset_y, b.confidence, b.mask)
                for b in boxes
            ]
        return boxes

    def _detect_single_tta(self, image: np.ndarray, offset_x: int, offset_y: int) -> list[BubbleBox]:
        h, w = image.shape[:2]
        if h <= 0 or w <= 0:
            return []

        all_boxes = []

        all_boxes.extend(self._detect_single_plain(image, offset_x, offset_y))

        flipped = cv2.flip(image, 1)
        flipped_boxes = self._detect_single_plain(flipped, 0, 0)
        for b in flipped_boxes:
            nx1 = max(0, min(w, w - b.x2))
            nx2 = max(0, min(w, w - b.x1))
            ny1 = max(0, min(h, b.y1))
            ny2 = max(0, min(h, b.y2))
            nw = nx2 - nx1
            nh = ny2 - ny1
            if nw > 0 and nh > 0:
                mask = cv2.flip(b.mask, 1) if b.mask is not None else None
                all_boxes.append(
                    BubbleBox(
                        nx1 + offset_x,
                        ny1 + offset_y,
                        nx2 + offset_x,
                        ny2 + offset_y,
                        b.confidence,
                        mask,
                    )
                )

        small_scale = 0.85
        sh, sw = int(round(h * small_scale)), int(round(w * small_scale))
        if sh > 10 and sw > 10:
            scale_x = sw / w
            scale_y = sh / h
            small = cv2.resize(image, (sw, sh))
            small_boxes = self._detect_single_plain(small, 0, 0)
            for b in small_boxes:
                nx1 = max(0, min(w, int(round(b.x1 / scale_x))))
                ny1 = max(0, min(h, int(round(b.y1 / scale_y))))
                nx2 = max(0, min(w, int(round(b.x2 / scale_x))))
                ny2 = max(0, min(h, int(round(b.y2 / scale_y))))
                nw = nx2 - nx1
                nh = ny2 - ny1
                if nw > 0 and nh > 0:
                    mask = None
                    if b.mask is not None:
                        mask = cv2.resize(b.mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
                    all_boxes.append(
                        BubbleBox(
                            nx1 + offset_x,
                            ny1 + offset_y,
                            nx2 + offset_x,
                            ny2 + offset_y,
                            b.confidence,
                            mask,
                        )
                    )

        return self._nms_boxes(all_boxes)

    def _preprocess(self, image: np.ndarray):
        h, w = image.shape[:2]
        if h <= 0 or w <= 0:
            return None, 1.0, (0, 0)
        scale = INPUT_SIZE / max(h, w)
        nh, nw = int(h * scale), int(w * scale)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 and image.shape[2] == 3 else image
        resized = cv2.resize(img_rgb, (nw, nh))
        canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
        pad_x, pad_y = (INPUT_SIZE - nw) // 2, (INPUT_SIZE - nh) // 2
        canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
        blob = canvas.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[None]
        return blob, scale, (pad_x, pad_y)

    def _postprocess(self, outputs, scale, pad, orig_w, orig_h) -> list[BubbleBox]:
        pad_x, pad_y = pad
        out_arr = np.squeeze(outputs[0])
        if out_arr.ndim == 1:
            out_arr = out_arr[np.newaxis, :]
        if out_arr.ndim == 2 and out_arr.shape[0] < out_arr.shape[1]:
            out_arr = out_arr.T

        if out_arr.ndim != 2 or out_arr.shape[0] == 0:
            return []

        has_proto = len(outputs) > 1 and outputs[1].ndim == 4
        if has_proto:
            num_mask_coeffs = outputs[1].shape[1]
            num_classes = max(1, out_arr.shape[1] - 4 - num_mask_coeffs)
            prototypes = outputs[1][0]
        else:
            num_mask_coeffs = 0
            num_classes = max(1, out_arr.shape[1] - 4)
            prototypes = None

        if num_classes == 1:
            confidences = out_arr[:, 4].astype(np.float32, copy=False)
        else:
            class_end = 4 + num_classes
            confidences = np.max(out_arr[:, 4:class_end], axis=1).astype(
                np.float32, copy=False
            )

        keep = np.flatnonzero(confidences >= self.conf_threshold)
        if keep.size == 0:
            return []

        selected = out_arr[keep]
        conf_selected = confidences[keep]
        cx = selected[:, 0]
        cy = selected[:, 1]
        bw = selected[:, 2]
        bh = selected[:, 3]

        x1 = np.maximum(0.0, (cx - bw / 2.0 - pad_x) / scale)
        y1 = np.maximum(0.0, (cy - bh / 2.0 - pad_y) / scale)
        x2 = np.minimum(float(orig_w), (cx + bw / 2.0 - pad_x) / scale)
        y2 = np.minimum(float(orig_h), (cy + bh / 2.0 - pad_y) / scale)

        valid = ((x2 - x1) >= 4.0) & ((y2 - y1) >= 4.0)
        if not np.any(valid):
            return []

        valid_rows = np.flatnonzero(valid)
        candidates = []
        coeff_start = 4 + num_classes
        coeff_end = coeff_start + num_mask_coeffs

        for j in valid_rows.tolist():
            canvas_box = None
            mask_coeffs = None
            if has_proto and num_mask_coeffs > 0:
                canvas_box = (
                    float(cx[j] - bw[j] / 2.0),
                    float(cy[j] - bh[j] / 2.0),
                    float(cx[j] + bw[j] / 2.0),
                    float(cy[j] + bh[j] / 2.0),
                )
                mask_coeffs = selected[j, coeff_start:coeff_end].copy()

            candidates.append((
                float(x1[j]),
                float(y1[j]),
                float(x2[j]),
                float(y2[j]),
                float(conf_selected[j]),
                canvas_box,
                mask_coeffs,
            ))

        return self._nms(candidates, prototypes)

    def _decode_mask(self, mask_coeffs, prototypes, canvas_box, box_w: int, box_h: int) -> np.ndarray | None:
        if mask_coeffs is None or prototypes is None or box_w < 1 or box_h < 1:
            return None

        num_proto, mh, mw = prototypes.shape
        proto_flat = prototypes.reshape(num_proto, -1)
        logits = np.clip(mask_coeffs @ proto_flat, -88.0, 88.0)
        mask_full = 1 / (1 + np.exp(-logits.reshape(mh, mw)))

        canvas_to_proto = mw / INPUT_SIZE
        ccx1, ccy1, ccx2, ccy2 = canvas_box
        px1 = int(max(0, min(mw - 1, round(ccx1 * canvas_to_proto))))
        py1 = int(max(0, min(mh - 1, round(ccy1 * canvas_to_proto))))
        px2 = int(max(px1 + 1, min(mw, round(ccx2 * canvas_to_proto))))
        py2 = int(max(py1 + 1, min(mh, round(ccy2 * canvas_to_proto))))

        crop = mask_full[py1:py2, px1:px2]
        if crop.size == 0:
            return None

        resized = cv2.resize(crop, (box_w, box_h), interpolation=cv2.INTER_LINEAR)
        return (resized > 0.5).astype(np.uint8) * 255

    def _nms(self, candidates: list[tuple], prototypes=None) -> list[BubbleBox]:
        if not candidates:
            return []
        boxes = np.array([[c[0], c[1], c[2] - c[0], c[3] - c[1]] for c in candidates])
        corners = np.array([[c[0], c[1], c[2], c[3]] for c in candidates])
        scores = np.array([c[4] for c in candidates])
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), scores.tolist(), self.conf_threshold, BUBBLE_IOU_THRESHOLD
        )
        result = []
        for i in np.array(indices).flatten():
            x1, y1, x2, y2 = corners[i]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            _, _, _, _, _, canvas_box, mask_coeffs = candidates[i]
            mask = self._decode_mask(mask_coeffs, prototypes, canvas_box, x2 - x1, y2 - y1)
            result.append(BubbleBox(x1, y1, x2, y2, float(scores[i]), mask))
        return result

    def _nms_boxes(self, boxes: list[BubbleBox]) -> list[BubbleBox]:
        if not boxes:
            return []
        rects = np.array([[b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1] for b in boxes])
        scores = np.array([b.confidence for b in boxes])
        indices = cv2.dnn.NMSBoxes(
            rects.tolist(), scores.tolist(), self.conf_threshold, BUBBLE_IOU_THRESHOLD
        )
        return [boxes[i] for i in np.array(indices).flatten()]
