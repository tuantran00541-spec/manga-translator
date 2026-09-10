from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from app.ort_utils import make_session
from app.model_contracts import validate_detector_session
from app.parameters import (
    BUBBLE_IOU_THRESHOLD,
    DETECTOR_CONFIDENCE_MAX,
    DETECTOR_INPUT_SIZE as INPUT_SIZE,
    DETECTOR_LETTERBOX_VALUE,
    DETECTOR_MASK_HYSTERESIS_LOW_THRESHOLD,
    DETECTOR_MASK_HYSTERESIS_MIN_CORE_PIXELS,
    DETECTOR_MASK_THRESHOLD,
    DETECTOR_MAX_ASPECT_RATIO as MAX_ASPECT_RATIO,
    DETECTOR_MAX_BOX_AREA_RATIO as MAX_BOX_AREA_RATIO,
    DETECTOR_MAX_BOX_WIDTH_RATIO as MAX_BOX_WIDTH_RATIO,
    DETECTOR_MIN_BOX_SIDE,
    DETECTOR_TALL_IMAGE_FACTOR,
    DETECTOR_TEXT_MASK_DECODE_PAD,
    DETECTOR_TTA_ENABLED as ENABLE_TTA,
    DETECTOR_TTA_MIN_SIDE,
    DETECTOR_TTA_SMALL_SCALE,
    DETECTOR_WINDOW_OVERLAP as SLICE_OVERLAP,
)


@dataclass(frozen=True)
class LetterboxTransform:
    """Exact source-to-model transform for half-open pixel boxes [x1,y1,x2,y2)."""

    src_w: int
    src_h: int
    input_w: int
    input_h: int
    resized_w: int
    resized_h: int
    pad_x: int
    pad_y: int
    scale_x: float
    scale_y: float
    offset_x: int = 0
    offset_y: int = 0

    @classmethod
    def create(
        cls,
        src_w: int,
        src_h: int,
        input_w: int,
        input_h: int,
        *,
        offset_x: int = 0,
        offset_y: int = 0,
    ):
        if min(src_w, src_h, input_w, input_h) <= 0:
            raise ValueError("Letterbox dimensions must be positive")
        nominal = min(input_w / src_w, input_h / src_h)
        resized_w = max(1, min(input_w, int(src_w * nominal)))
        resized_h = max(1, min(input_h, int(src_h * nominal)))
        pad_x = (input_w - resized_w) // 2
        pad_y = (input_h - resized_h) // 2
        return cls(
            src_w=src_w,
            src_h=src_h,
            input_w=input_w,
            input_h=input_h,
            resized_w=resized_w,
            resized_h=resized_h,
            pad_x=pad_x,
            pad_y=pad_y,
            scale_x=resized_w / src_w,
            scale_y=resized_h / src_h,
            offset_x=int(offset_x),
            offset_y=int(offset_y),
        )

    def page_box_from_canvas(self, canvas_box) -> tuple[int, int, int, int]:
        import math

        x1, y1, x2, y2 = (float(v) for v in canvas_box)
        local_x1 = max(0.0, min(float(self.src_w), (x1 - self.pad_x) / self.scale_x))
        local_y1 = max(0.0, min(float(self.src_h), (y1 - self.pad_y) / self.scale_y))
        local_x2 = max(0.0, min(float(self.src_w), (x2 - self.pad_x) / self.scale_x))
        local_y2 = max(0.0, min(float(self.src_h), (y2 - self.pad_y) / self.scale_y))
        ix1 = max(0, min(self.src_w, int(math.floor(local_x1))))
        iy1 = max(0, min(self.src_h, int(math.floor(local_y1))))
        ix2 = max(ix1, min(self.src_w, int(math.ceil(local_x2))))
        iy2 = max(iy1, min(self.src_h, int(math.ceil(local_y2))))
        return (
            ix1 + self.offset_x,
            iy1 + self.offset_y,
            ix2 + self.offset_x,
            iy2 + self.offset_y,
        )

    def canvas_box_from_page(self, page_box) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = (float(v) for v in page_box)
        x1 -= self.offset_x
        x2 -= self.offset_x
        y1 -= self.offset_y
        y2 -= self.offset_y
        return (
            x1 * self.scale_x + self.pad_x,
            y1 * self.scale_y + self.pad_y,
            x2 * self.scale_x + self.pad_x,
            y2 * self.scale_y + self.pad_y,
        )


@dataclass(frozen=True)
class MaskDecodeGeometry:
    transform: LetterboxTransform
    source_box: tuple[int, int, int, int]


@dataclass
class BubbleBox:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    mask: np.ndarray | None = None
    source_model: str = "unknown"
    class_id: int = 0
    class_name: str = "unknown"
    semantic_type: str = "unknown"
    mask_source: str = "none"
    safe_to_inpaint: bool = False
    ocr_eligible: bool = False
    needs_review: bool = False
    source_role: str = "unknown"
    deferred_reason: str | None = None

    @property
    def verified_mask(self) -> bool:
        h = self.y2 - self.y1
        w = self.x2 - self.x1
        return (
            self.mask is not None
            and self.mask.ndim == 2
            and self.mask.shape == (h, w)
            and bool(np.any(self.mask > 0))
        )


class YoloDetector:
    def __init__(
        self,
        model_path,
        conf_threshold: float,
        use_tta: bool | None = None,
        *,
        model_role: str,
    ):
        self.model_path = str(model_path)
        self.source_model = Path(model_path).name
        self.model_role = str(model_role)
        self.session = make_session(model_path)
        self.contract = validate_detector_session(
            self.session,
            role=self.model_role,
            configured_input_size=INPUT_SIZE,
        )
        self.input_name = self.contract.input_name
        self.conf_threshold = conf_threshold
        self.use_tta = ENABLE_TTA if use_tta is None else use_tta

    def _class_name(self, class_id: int, num_classes: int) -> str:
        if num_classes != len(self.contract.class_names):
            return f"class_{class_id}"
        if 0 <= class_id < len(self.contract.class_names):
            return self.contract.class_names[class_id]
        return f"class_{class_id}"

    @staticmethod
    def _semantic_type(class_name: str) -> str:
        if class_name == "text_bubble":
            return "speech_bubble"
        if class_name == "text_free":
            return "free_text"
        if class_name == "text_comic":
            return "text"
        return class_name or "unknown"

    def _with_semantics(self, box: BubbleBox) -> BubbleBox:
        verified = box.verified_mask
        segmenter_evidence = box.source_role == "text_segmenter"
        safe = bool(verified and segmenter_evidence)
        return replace(
            box,
            mask_source="text_segmenter" if safe else ("model" if verified else "none"),
            safe_to_inpaint=safe,
            ocr_eligible=safe,
            needs_review=not safe,
            source_role=self.model_role,
        )

    def detect(self, image: np.ndarray) -> list[BubbleBox]:
        h, w = image.shape[:2]
        if h <= INPUT_SIZE * DETECTOR_TALL_IMAGE_FACTOR:
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

            if self.model_role == "text_segmenter":
                all_boxes.extend(self._detect_single_plain(image, 0, 0))

            boxes = self._nms_boxes(all_boxes)

        semantic_boxes = [self._with_semantics(b) for b in boxes]
        return self._filter_invalid(semantic_boxes, w, h)

    @staticmethod
    def _filter_invalid(
        boxes: list[BubbleBox], img_w: int, img_h: int
    ) -> list[BubbleBox]:
        """Retain oversized/aspect outliers as review evidence instead of dropping them.

        Postprocess already rejects non-positive boxes before this point. Width,
        page-area and aspect limits are policy heuristics, not proof that model
        evidence is false. They therefore revoke automatic erase authority but
        preserve the detected region for review with an explicit reason.
        """
        result = []
        page_area = max(1, img_w * img_h)
        for b in boxes:
            box_w = b.x2 - b.x1
            box_h = b.y2 - b.y1
            if box_w <= 0 or box_h <= 0:
                continue
            reasons: list[str] = []
            if box_w > img_w * MAX_BOX_WIDTH_RATIO:
                reasons.append("box_width_limit")
            if (box_w * box_h) > page_area * MAX_BOX_AREA_RATIO:
                reasons.append("box_area_limit")
            aspect = box_w / box_h
            if aspect > MAX_ASPECT_RATIO or aspect < 1 / MAX_ASPECT_RATIO:
                reasons.append("box_aspect_limit")
            if reasons:
                result.append(
                    replace(
                        b,
                        safe_to_inpaint=False,
                        ocr_eligible=bool(
                            b.ocr_eligible
                            or b.source_role == "text_segmenter"
                            or b.semantic_type == "free_text"
                        ),
                        needs_review=True,
                        deferred_reason="|".join(reasons),
                    )
                )
            else:
                result.append(b)
        return result

    def _detect_single(self, image: np.ndarray, offset_x: int, offset_y: int) -> list[BubbleBox]:
        if self.use_tta:
            return self._detect_single_tta(image, offset_x, offset_y)
        return self._detect_single_plain(image, offset_x, offset_y)

    def _detect_single_plain(
        self,
        image: np.ndarray,
        offset_x: int,
        offset_y: int,
    ) -> list[BubbleBox]:
        blob, transform = self._preprocess(
            image, offset_x=offset_x, offset_y=offset_y
        )
        if blob is None or transform is None:
            return []
        outputs = self.session.run(None, {self.input_name: blob})
        return self._postprocess(outputs, transform)

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
            if nx2 > nx1 and ny2 > ny1:
                mask = cv2.flip(b.mask, 1) if b.mask is not None else None
                all_boxes.append(
                    replace(
                        b,
                        x1=nx1 + offset_x,
                        y1=ny1 + offset_y,
                        x2=nx2 + offset_x,
                        y2=ny2 + offset_y,
                        mask=mask,
                    )
                )

        small_scale = DETECTOR_TTA_SMALL_SCALE
        sh, sw = int(round(h * small_scale)), int(round(w * small_scale))
        if sh > DETECTOR_TTA_MIN_SIDE and sw > DETECTOR_TTA_MIN_SIDE:
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
                        replace(
                            b,
                            x1=nx1 + offset_x,
                            y1=ny1 + offset_y,
                            x2=nx2 + offset_x,
                            y2=ny2 + offset_y,
                            mask=mask,
                        )
                    )

        return self._nms_boxes(all_boxes)

    def _preprocess(
        self,
        image: np.ndarray,
        *,
        offset_x: int = 0,
        offset_y: int = 0,
    ):
        h, w = image.shape[:2]
        if h <= 0 or w <= 0:
            return None, None
        transform = LetterboxTransform.create(
            w,
            h,
            INPUT_SIZE,
            INPUT_SIZE,
            offset_x=offset_x,
            offset_y=offset_y,
        )
        img_rgb = (
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if image.ndim == 3 and image.shape[2] == 3
            else image
        )
        resized = cv2.resize(img_rgb, (transform.resized_w, transform.resized_h))
        canvas = np.full(
            (transform.input_h, transform.input_w, 3),
            DETECTOR_LETTERBOX_VALUE,
            dtype=np.uint8,
        )
        y1, y2 = transform.pad_y, transform.pad_y + transform.resized_h
        x1, x2 = transform.pad_x, transform.pad_x + transform.resized_w
        canvas[y1:y2, x1:x2] = resized
        blob = (canvas.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        return blob, transform

    def _postprocess(
        self,
        outputs,
        transform: LetterboxTransform,
    ) -> list[BubbleBox]:
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

        class_end = 4 + num_classes
        class_scores = out_arr[:, 4:class_end].astype(np.float32, copy=False)
        if num_classes == 1:
            class_ids = np.zeros(out_arr.shape[0], dtype=np.int32)
            confidences = class_scores[:, 0]
        else:
            class_ids = np.argmax(class_scores, axis=1).astype(np.int32, copy=False)
            confidences = class_scores[np.arange(class_scores.shape[0]), class_ids]

        keep = np.flatnonzero(confidences >= self.conf_threshold)
        if keep.size == 0:
            return []

        selected = out_arr[keep]
        conf_selected = confidences[keep]
        class_selected = class_ids[keep]
        coeff_start = 4 + num_classes
        coeff_end = coeff_start + num_mask_coeffs
        candidates = []
        for j in range(selected.shape[0]):
            cx, cy, bw, bh = (float(v) for v in selected[j, :4])
            raw_canvas_box = (
                cx - bw / 2.0,
                cy - bh / 2.0,
                cx + bw / 2.0,
                cy + bh / 2.0,
            )
            source_box = transform.page_box_from_canvas(raw_canvas_box)
            x1, y1, x2, y2 = source_box
            if (
                (x2 - x1) < DETECTOR_MIN_BOX_SIDE
                or (y2 - y1) < DETECTOR_MIN_BOX_SIDE
            ):
                continue
            geometry = None
            mask_coeffs = None
            if has_proto and num_mask_coeffs > 0:
                # Prototype masks are otherwise cropped exactly at the detector
                # box.  A small text-only decode pad retains outlines, shadows
                # and glow that the box regressor legitimately clips, without
                # widening the final mask by global morphology.
                if self.model_role == "text_segmenter":
                    pad = DETECTOR_TEXT_MASK_DECODE_PAD
                    source_box = (
                        max(transform.offset_x, x1 - pad),
                        max(transform.offset_y, y1 - pad),
                        min(transform.offset_x + transform.src_w, x2 + pad),
                        min(transform.offset_y + transform.src_h, y2 + pad),
                    )
                    x1, y1, x2, y2 = source_box
                geometry = MaskDecodeGeometry(transform, source_box)
                mask_coeffs = selected[j, coeff_start:coeff_end].copy()
            candidates.append(
                (
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2),
                    float(conf_selected[j]),
                    int(class_selected[j]),
                    num_classes,
                    geometry,
                    mask_coeffs,
                )
            )
        return self._nms(candidates, prototypes)

    def _decode_mask(
        self,
        mask_coeffs,
        prototypes,
        geometry,
        box_w: int,
        box_h: int,
    ) -> np.ndarray | None:
        if mask_coeffs is None or prototypes is None or box_w < 1 or box_h < 1:
            return None

        num_proto, mh, mw = prototypes.shape
        logits = np.clip(
            mask_coeffs @ prototypes.reshape(num_proto, -1),
            -88.0,
            88.0,
        )
        probability_map = 1 / (1 + np.exp(-logits.reshape(mh, mw)))

        if isinstance(geometry, MaskDecodeGeometry):
            canvas_box = geometry.transform.canvas_box_from_page(geometry.source_box)
            input_w = geometry.transform.input_w
            input_h = geometry.transform.input_h
        else:
            # Legacy seven-field test/plugin candidates remain supported, but
            # production candidates always carry the explicit transform above.
            canvas_box = geometry
            input_w = INPUT_SIZE
            input_h = INPUT_SIZE
        if canvas_box is None:
            return None

        cx1, cy1, cx2, cy2 = (float(v) for v in canvas_box)
        proto_scale_x = mw / float(input_w)
        proto_scale_y = mh / float(input_h)
        px1 = max(0, min(mw - 1, int(np.floor(cx1 * proto_scale_x))))
        py1 = max(0, min(mh - 1, int(np.floor(cy1 * proto_scale_y))))
        px2 = max(px1 + 1, min(mw, int(np.ceil(cx2 * proto_scale_x))))
        py2 = max(py1 + 1, min(mh, int(np.ceil(cy2 * proto_scale_y))))

        crop = probability_map[py1:py2, px1:px2]
        if crop.size == 0:
            return None
        probabilities = cv2.resize(
            crop,
            (box_w, box_h),
            interpolation=cv2.INTER_LINEAR,
        )
        return self._decode_text_mask_hysteresis(probabilities)

    @staticmethod
    def _decode_text_mask_hysteresis(probabilities: np.ndarray) -> np.ndarray:
        """Keep only low-confidence support connected to a confident core.

        This preserves a glyph's anti-aliased edge, outline and nearby shadow
        when the segmenter sees them, but cannot bridge to unrelated artwork as
        a dilation would. A detection with no confident core has no destructive
        mask authority and is handled by the normal review path.
        """
        if probabilities.size == 0:
            return np.zeros(probabilities.shape, dtype=np.uint8)
        core = probabilities >= DETECTOR_MASK_THRESHOLD
        if int(np.count_nonzero(core)) < DETECTOR_MASK_HYSTERESIS_MIN_CORE_PIXELS:
            return np.zeros(probabilities.shape, dtype=np.uint8)
        low = probabilities >= min(
            DETECTOR_MASK_THRESHOLD,
            DETECTOR_MASK_HYSTERESIS_LOW_THRESHOLD,
        )
        labels_count, labels = cv2.connectedComponents(low.astype(np.uint8), connectivity=8)
        if labels_count <= 1:
            return np.zeros(probabilities.shape, dtype=np.uint8)
        keep = np.unique(labels[core])
        keep = keep[keep > 0]
        if keep.size == 0:
            return np.zeros(probabilities.shape, dtype=np.uint8)
        return np.isin(labels, keep).astype(np.uint8) * 255

    @staticmethod
    def _candidate_fields(candidate: tuple) -> tuple[float, int, int, object, object]:
        """Normalize current and v0.1 candidate tuple layouts.

        v0.1 tests and third-party callers may still pass the legacy seven-field
        tuple ``(x1, y1, x2, y2, score, canvas_box, mask_coeffs)``. Production
        v0.2 candidates append class provenance before the mask metadata. Keep
        that compatibility without weakening class-aware NMS for new detections.
        """
        if len(candidate) >= 9 and isinstance(candidate[5], (int, np.integer)):
            return float(candidate[4]), int(candidate[5]), int(candidate[6]), candidate[7], candidate[8]
        if len(candidate) >= 7:
            return float(candidate[4]), 0, 1, candidate[5], candidate[6]
        raise ValueError("Invalid detector candidate tuple")

    @staticmethod
    def _box_iou(a: BubbleBox, b: BubbleBox) -> float:
        ix1 = max(a.x1, b.x1)
        iy1 = max(a.y1, b.y1)
        ix2 = min(a.x2, b.x2)
        iy2 = min(a.y2, b.y2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        intersection = (ix2 - ix1) * (iy2 - iy1)
        area_a = max(0, a.x2 - a.x1) * max(0, a.y2 - a.y1)
        area_b = max(0, b.x2 - b.x1) * max(0, b.y2 - b.y1)
        return intersection / float(max(1, area_a + area_b - intersection))

    @staticmethod
    def _merge_text_mask_evidence(boxes: list[BubbleBox]) -> BubbleBox:
        """Union duplicate text masks instead of discarding their pixels.

        Horizontal-flip TTA, detector windows, and shared seam passes can return
        slightly different extents for the same text block. Conventional NMS
        keeps only the highest score; a tighter high-score box can therefore
        discard the right or bottom of a fuller lower-score mask. Only verified
        text-segmenter pixels are merged here, so this never invents a rectangle.
        """
        base = max(boxes, key=lambda box: float(box.confidence))
        evidence = [box for box in boxes if box.verified_mask]
        if not evidence:
            return replace(base)

        x1 = min(box.x1 for box in evidence)
        y1 = min(box.y1 for box in evidence)
        x2 = max(box.x2 for box in evidence)
        y2 = max(box.y2 for box in evidence)
        merged = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        for box in evidence:
            mask = box.mask
            box_w = box.x2 - box.x1
            box_h = box.y2 - box.y1
            if mask.shape != (box_h, box_w):
                mask = cv2.resize(
                    mask, (box_w, box_h), interpolation=cv2.INTER_NEAREST
                )
            dy1 = box.y1 - y1
            dx1 = box.x1 - x1
            target = merged[dy1 : dy1 + box_h, dx1 : dx1 + box_w]
            merged[dy1 : dy1 + box_h, dx1 : dx1 + box_w] = np.maximum(
                target, mask
            )

        safe = bool(
            np.any(merged > 0)
            and any(bool(box.safe_to_inpaint) for box in evidence)
        )
        mask_source = (
            base.mask_source
            if base.verified_mask
            else max(evidence, key=lambda box: float(box.confidence)).mask_source
        )
        return replace(
            base,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            mask=merged,
            mask_source=mask_source,
            safe_to_inpaint=safe,
            ocr_eligible=any(bool(box.ocr_eligible) for box in boxes),
            needs_review=any(bool(box.needs_review) for box in boxes),
        )

    @classmethod
    def _nms_box_group(
        cls,
        members: list[BubbleBox],
        *,
        score_threshold: float,
        iou_threshold: float,
    ) -> list[BubbleBox]:
        if not members:
            return []
        rects = np.array(
            [[b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1] for b in members]
        )
        scores = np.array(
            [max(score_threshold, float(b.confidence)) for b in members]
        )
        raw_indices = cv2.dnn.NMSBoxes(
            rects.tolist(), scores.tolist(), score_threshold, iou_threshold
        )
        kept = [int(i) for i in np.array(raw_indices).flatten()]
        if not kept:
            return []

        if members[kept[0]].source_role != "text_segmenter":
            return [members[i] for i in kept]

        buckets = {index: [members[index]] for index in kept}
        kept_set = set(kept)
        for index, box in enumerate(members):
            if index in kept_set or not box.verified_mask:
                continue
            target = max(
                kept,
                key=lambda kept_index: cls._box_iou(
                    box, members[kept_index]
                ),
            )
            if cls._box_iou(box, members[target]) > iou_threshold:
                buckets[target].append(box)

        return [cls._merge_text_mask_evidence(buckets[index]) for index in kept]

    @staticmethod
    def _candidate_iou(a: tuple, b: tuple) -> float:
        ax1, ay1, ax2, ay2 = (float(v) for v in a[:4])
        bx1, by1, bx2, by2 = (float(v) for v in b[:4])
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        aa = max(1.0, (ax2 - ax1) * (ay2 - ay1))
        bb = max(1.0, (bx2 - bx1) * (by2 - by1))
        return inter / max(1.0, aa + bb - inter)

    def _plan_candidate_buckets(
        self,
        subset: list[tuple],
    ) -> tuple[list[int], dict[int, list[int]]]:
        """Plan class-local suppression before allocating any mask pixels."""
        if not subset:
            return [], {}
        rects = [
            [
                int(candidate[0]),
                int(candidate[1]),
                max(0, int(candidate[2]) - int(candidate[0])),
                max(0, int(candidate[3]) - int(candidate[1])),
            ]
            for candidate in subset
        ]
        scores = [
            max(self.conf_threshold, float(self._candidate_fields(candidate)[0]))
            for candidate in subset
        ]
        raw_indices = cv2.dnn.NMSBoxes(
            rects,
            scores,
            self.conf_threshold,
            BUBBLE_IOU_THRESHOLD,
        )
        kept = [int(index) for index in np.asarray(raw_indices).reshape(-1)]
        if not kept:
            return [], {}
        buckets = {index: [index] for index in kept}
        kept_set = set(kept)
        for index, candidate in enumerate(subset):
            if index in kept_set:
                continue
            target = max(
                kept,
                key=lambda kept_index: self._candidate_iou(
                    candidate, subset[kept_index]
                ),
            )
            if self._candidate_iou(candidate, subset[target]) > BUBBLE_IOU_THRESHOLD:
                buckets[target].append(index)
        return kept, buckets

    @staticmethod
    def _prototype_crop_bounds(
        geometry,
        prototypes: np.ndarray,
    ) -> tuple[int, int, int, int] | None:
        if geometry is None or prototypes is None or prototypes.ndim != 3:
            return None
        _num_proto, mh, mw = prototypes.shape
        if isinstance(geometry, MaskDecodeGeometry):
            canvas_box = geometry.transform.canvas_box_from_page(geometry.source_box)
            input_w = geometry.transform.input_w
            input_h = geometry.transform.input_h
        else:
            canvas_box = geometry
            input_w = INPUT_SIZE
            input_h = INPUT_SIZE
        if canvas_box is None:
            return None
        cx1, cy1, cx2, cy2 = (float(v) for v in canvas_box)
        scale_x = mw / float(input_w)
        scale_y = mh / float(input_h)
        px1 = max(0, min(mw - 1, int(np.floor(cx1 * scale_x))))
        py1 = max(0, min(mh - 1, int(np.floor(cy1 * scale_y))))
        px2 = max(px1 + 1, min(mw, int(np.ceil(cx2 * scale_x))))
        py2 = max(py1 + 1, min(mh, int(np.ceil(cy2 * scale_y))))
        return px1, py1, px2, py2

    def _decode_candidate_masks_roi_batch(
        self,
        subset: list[tuple],
        member_indices: list[int],
        prototypes: np.ndarray,
        *,
        batch_size: int = 8,
    ) -> dict[int, np.ndarray | None]:
        """Batch coefficient products inside the union prototype ROI only."""
        decoded: dict[int, np.ndarray | None] = {index: None for index in member_indices}
        if prototypes is None or prototypes.ndim != 3:
            return decoded
        num_proto, _mh, _mw = prototypes.shape
        prepared = []
        for index in member_indices:
            candidate = subset[index]
            _score, _cid, _classes, geometry, coeffs = self._candidate_fields(candidate)
            bounds = self._prototype_crop_bounds(geometry, prototypes)
            if coeffs is None or bounds is None:
                continue
            coeffs = np.asarray(coeffs, dtype=np.float32).reshape(-1)
            if coeffs.size != num_proto:
                continue
            prepared.append((index, coeffs, bounds))
        if not prepared:
            return decoded

        ux1 = min(item[2][0] for item in prepared)
        uy1 = min(item[2][1] for item in prepared)
        ux2 = max(item[2][2] for item in prepared)
        uy2 = max(item[2][3] for item in prepared)
        proto_roi = prototypes[:, uy1:uy2, ux1:ux2].reshape(num_proto, -1)
        if proto_roi.size == 0:
            return decoded

        batch_size = max(1, int(batch_size))
        for start in range(0, len(prepared), batch_size):
            chunk = prepared[start:start + batch_size]
            coeff_matrix = np.stack([item[1] for item in chunk], axis=0)
            logits_batch = np.clip(coeff_matrix @ proto_roi, -88.0, 88.0)
            roi_h, roi_w = uy2 - uy1, ux2 - ux1
            logits_batch = logits_batch.reshape(len(chunk), roi_h, roi_w)
            for row, (index, _coeffs, bounds) in enumerate(chunk):
                px1, py1, px2, py2 = bounds
                crop_logits = logits_batch[
                    row,
                    py1 - uy1:py2 - uy1,
                    px1 - ux1:px2 - ux1,
                ]
                if crop_logits.size == 0:
                    continue
                probabilities = 1.0 / (1.0 + np.exp(-crop_logits))
                x1, y1, x2, y2 = map(int, subset[index][:4])
                box_w, box_h = x2 - x1, y2 - y1
                if box_w < 1 or box_h < 1:
                    continue
                resized = cv2.resize(
                    probabilities,
                    (box_w, box_h),
                    interpolation=cv2.INTER_LINEAR,
                )
                decoded[index] = self._decode_text_mask_hysteresis(resized)
        return decoded

    def _candidate_to_box(
        self,
        candidate: tuple,
        mask: np.ndarray | None,
    ) -> BubbleBox:
        x1, y1, x2, y2 = map(int, candidate[:4])
        score, cid, num_classes, _geometry, _coeffs = self._candidate_fields(candidate)
        class_name = self._class_name(int(cid), int(num_classes))
        return BubbleBox(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            confidence=min(float(score), DETECTOR_CONFIDENCE_MAX),
            mask=mask,
            source_model=getattr(self, "source_model", "unknown"),
            class_id=int(cid),
            class_name=class_name,
            semantic_type=self._semantic_type(class_name),
            source_role=self.model_role,
        )

    def _nms(self, candidates: list[tuple], prototypes=None) -> list[BubbleBox]:
        if not candidates:
            return []
        result: list[BubbleBox] = []
        by_class: dict[int, list[int]] = {}
        for idx, candidate in enumerate(candidates):
            _, class_id, _, _, _ = self._candidate_fields(candidate)
            by_class.setdefault(class_id, []).append(idx)

        for _class_id, member_indices in by_class.items():
            subset = [candidates[index] for index in member_indices]
            kept, buckets = self._plan_candidate_buckets(subset)
            if not kept:
                continue

            has_mask_model = bool(
                prototypes is not None
                and getattr(self, "model_role", "unknown") == "text_segmenter"
            )
            if not has_mask_model:
                result.extend(
                    self._candidate_to_box(subset[index], None)
                    for index in kept
                )
                continue

            for kept_index in kept:
                contributors = buckets.get(kept_index, [kept_index])
                masks = self._decode_candidate_masks_roi_batch(
                    subset,
                    contributors,
                    prototypes,
                )
                decoded = [
                    self._candidate_to_box(subset[index], masks.get(index))
                    for index in contributors
                ]
                result.append(self._merge_text_mask_evidence(decoded))

        result.sort(key=lambda box: box.confidence, reverse=True)
        return result

    def _nms_boxes(self, boxes: list[BubbleBox]) -> list[BubbleBox]:
        if not boxes:
            return []
        result: list[BubbleBox] = []
        by_class: dict[tuple[str, str, int], list[BubbleBox]] = {}
        for b in boxes:
            by_class.setdefault(
                (b.source_role, b.source_model, b.class_id), []
            ).append(b)
        for members in by_class.values():
            result.extend(
                self._nms_box_group(
                    members,
                    score_threshold=self.conf_threshold,
                    iou_threshold=BUBBLE_IOU_THRESHOLD,
                )
            )
        result.sort(key=lambda b: b.confidence, reverse=True)
        return result
