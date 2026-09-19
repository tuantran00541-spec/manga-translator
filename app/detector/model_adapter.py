from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.detector.bubble_detector import LetterboxTransform, MaskDecodeGeometry
from app.detector.evidence import DetectionEvidence, EvidenceKind
from app.detector.page_context import PageContext
from app.model_contracts import validate_detector_tensor_session
from app.ort_utils import make_session
from app.parameters import (
    BUBBLE_IOU_THRESHOLD,
    DETECTOR_CONFIDENCE_MAX,
    DETECTOR_LETTERBOX_VALUE,
    DETECTOR_MASK_HYSTERESIS_LOW_THRESHOLD,
    DETECTOR_MASK_HYSTERESIS_MIN_CORE_PIXELS,
    DETECTOR_MASK_THRESHOLD,
    DETECTOR_MIN_BOX_SIDE,
    DETECTOR_TEXT_MASK_DECODE_PAD,
)

__all__ = (
    "DetectorAdapter",
    "ModelCapabilities",
    "RawModelOutput",
    "YoloV8Adapter",
    "Yolo26SegAdapter",
)


@dataclass(frozen=True)
class ModelCapabilities:
    """What a model can emit, not what the pipeline is allowed to erase."""

    boxes: bool
    masks: bool
    classes: tuple[str, ...]
    static_input: tuple[int, int] | None
    batch: int | None

    def __post_init__(self) -> None:
        if not self.boxes and not self.masks:
            raise ValueError("Detector capability must expose boxes and/or masks")
        if self.batch is not None and self.batch <= 0:
            raise ValueError("batch must be positive or None")
        if self.static_input is not None and min(self.static_input) <= 0:
            raise ValueError("static_input dimensions must be positive")


@dataclass(frozen=True)
class RawModelOutput:
    tensors: tuple[Any, ...]
    transform: Any = None


class DetectorAdapter(ABC):
    """Boundary between model-specific inference/decode and pipeline policy."""

    source: str
    capabilities: ModelCapabilities

    @abstractmethod
    def infer(self, context: PageContext) -> RawModelOutput:
        raise NotImplementedError

    @abstractmethod
    def decode(self, raw: RawModelOutput) -> list[DetectionEvidence]:
        raise NotImplementedError

    def detect(self, context: PageContext) -> list[DetectionEvidence]:
        return self.decode(self.infer(context))


class YoloV8Adapter(DetectorAdapter):
    """Compatibility adapter over the existing YOLOv8 decoder.

    This is deliberately evidence-only: it never calls YoloDetector._with_semantics
    and therefore cannot grant destructive authority.
    """

    def __init__(self, detector, *, source: str | None = None) -> None:
        self.detector = detector
        role = str(detector.model_role)
        self.source = source or (
            "yolov8_text_segmenter"
            if role == "text_segmenter"
            else f"yolov8_{role}"
        )
        contract = detector.contract
        self.capabilities = ModelCapabilities(
            boxes=True,
            masks=bool(contract.provides_prototypes),
            classes=tuple(contract.class_names),
            static_input=(int(contract.input_height), int(contract.input_width)),
            batch=1,
        )

    def infer(self, context: PageContext) -> RawModelOutput:
        blob, transform = self.detector._preprocess(
            context.bgr,
            offset_x=context.origin_x,
            offset_y=context.origin_y,
        )
        if blob is None or transform is None:
            return RawModelOutput((), None)
        outputs = self.detector.session.run(
            None,
            {self.detector.input_name: blob},
        )
        return RawModelOutput(tuple(outputs), transform)

    def decode(self, raw: RawModelOutput) -> list[DetectionEvidence]:
        if not raw.tensors or raw.transform is None:
            return []
        boxes = self.detector._postprocess(raw.tensors, raw.transform)
        result: list[DetectionEvidence] = []
        for box in boxes:
            verified = bool(box.verified_mask)
            result.append(
                DetectionEvidence(
                    bbox=(int(box.x1), int(box.y1), int(box.x2), int(box.y2)),
                    confidence=float(box.confidence),
                    semantic=str(box.semantic_type),
                    source=self.source,
                    evidence_kind=EvidenceKind.MASK if verified else EvidenceKind.BOX,
                    mask=box.mask,
                    class_id=int(box.class_id),
                    class_name=str(box.class_name),
                    metadata={"source_model": str(box.source_model)},
                )
            )
        return result


@dataclass(frozen=True)
class _SegCandidate:
    bbox: tuple[int, int, int, int]
    confidence: float
    class_id: int
    geometry: MaskDecodeGeometry
    mask_coeffs: np.ndarray


class Yolo26SegAdapter(DetectorAdapter):
    """Decode a raw three-class YOLO26 segmentation model as evidence only.

    The adapter validates tensor geometry directly and never borrows a YOLOv8
    role or cleanup policy. Even a high-quality YOLO26 text mask remains plain
    DetectionEvidence until AuthorityPolicy explicitly promotes its source.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        conf_threshold: float,
        input_size: int,
        class_names: tuple[str, ...] = ("frame", "text", "balloon"),
        provider_override: str | None = None,
        source: str = "yolo26_segmentation",
    ) -> None:
        if not class_names:
            raise ValueError("YOLO26 adapter requires at least one class")
        self.model_path = str(model_path)
        self.source_model = Path(model_path).name
        self.source = str(source)
        self.conf_threshold = float(conf_threshold)
        self.input_size = int(input_size)
        self.class_names = tuple(str(name) for name in class_names)
        self.session = make_session(
            model_path,
            provider_override=provider_override,
        )
        self.contract = validate_detector_tensor_session(
            self.session,
            configured_input_size=self.input_size,
            expected_feature_count=4 + len(self.class_names) + 32,
            prototype_channels=32,
        )
        self.input_name = self.contract.input_name
        self.capabilities = ModelCapabilities(
            boxes=True,
            masks=True,
            classes=self.class_names,
            static_input=(self.contract.input_height, self.contract.input_width),
            batch=1,
        )

    def _preprocess(self, context: PageContext) -> tuple[np.ndarray, LetterboxTransform]:
        key = (
            "detector_letterbox",
            self.input_size,
            self.input_size,
            int(DETECTOR_LETTERBOX_VALUE),
        )

        def build():
            transform = LetterboxTransform.create(
                context.width,
                context.height,
                self.input_size,
                self.input_size,
                offset_x=context.origin_x,
                offset_y=context.origin_y,
            )
            resized = cv2.resize(
                context.rgb,
                (transform.resized_w, transform.resized_h),
            )
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

        return context.cached(key, build)

    def infer(self, context: PageContext) -> RawModelOutput:
        blob, transform = self._preprocess(context)
        outputs = self.session.run(None, {self.input_name: blob})
        return RawModelOutput(tuple(outputs), transform)

    @staticmethod
    def _candidate_iou(a: _SegCandidate, b: _SegCandidate) -> float:
        ax1, ay1, ax2, ay2 = a.bbox
        bx1, by1, bx2, by2 = b.bbox
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1, (bx2 - bx1) * (by2 - by1))
        return inter / float(max(1, area_a + area_b - inter))

    @staticmethod
    def _decode_text_mask_hysteresis(probabilities: np.ndarray) -> np.ndarray:
        if probabilities.size == 0:
            return np.zeros(probabilities.shape, dtype=np.uint8)
        core = probabilities >= DETECTOR_MASK_THRESHOLD
        if int(np.count_nonzero(core)) < DETECTOR_MASK_HYSTERESIS_MIN_CORE_PIXELS:
            return np.zeros(probabilities.shape, dtype=np.uint8)
        low = probabilities >= min(
            DETECTOR_MASK_THRESHOLD,
            DETECTOR_MASK_HYSTERESIS_LOW_THRESHOLD,
        )
        labels_count, labels = cv2.connectedComponents(
            low.astype(np.uint8),
            connectivity=8,
        )
        if labels_count <= 1:
            return np.zeros(probabilities.shape, dtype=np.uint8)
        keep = np.unique(labels[core])
        keep = keep[keep > 0]
        if keep.size == 0:
            return np.zeros(probabilities.shape, dtype=np.uint8)
        return np.isin(labels, keep).astype(np.uint8) * 255

    @staticmethod
    def _prototype_crop_bounds(
        candidate: _SegCandidate,
        prototypes: np.ndarray,
    ) -> tuple[int, int, int, int]:
        _channels, height, width = prototypes.shape
        transform = candidate.geometry.transform
        canvas_box = transform.canvas_box_from_page(candidate.geometry.source_box)
        x1, y1, x2, y2 = (float(value) for value in canvas_box)
        scale_x = width / float(transform.input_w)
        scale_y = height / float(transform.input_h)
        px1 = max(0, min(width - 1, int(np.floor(x1 * scale_x))))
        py1 = max(0, min(height - 1, int(np.floor(y1 * scale_y))))
        px2 = max(px1 + 1, min(width, int(np.ceil(x2 * scale_x))))
        py2 = max(py1 + 1, min(height, int(np.ceil(y2 * scale_y))))
        return px1, py1, px2, py2

    def _decode_masks(
        self,
        candidates: list[_SegCandidate],
        indices: list[int],
        prototypes: np.ndarray,
        *,
        batch_size: int = 8,
    ) -> dict[int, np.ndarray | None]:
        decoded: dict[int, np.ndarray | None] = {index: None for index in indices}
        if prototypes.ndim != 3:
            return decoded
        channels, _height, _width = prototypes.shape
        prepared = []
        for index in indices:
            candidate = candidates[index]
            coeffs = np.asarray(candidate.mask_coeffs, dtype=np.float32).reshape(-1)
            if coeffs.size != channels:
                continue
            prepared.append(
                (index, coeffs, self._prototype_crop_bounds(candidate, prototypes))
            )
        if not prepared:
            return decoded

        ux1 = min(item[2][0] for item in prepared)
        uy1 = min(item[2][1] for item in prepared)
        ux2 = max(item[2][2] for item in prepared)
        uy2 = max(item[2][3] for item in prepared)
        proto_roi = prototypes[:, uy1:uy2, ux1:ux2].reshape(channels, -1)
        if proto_roi.size == 0:
            return decoded

        batch_size = max(1, int(batch_size))
        roi_h, roi_w = uy2 - uy1, ux2 - ux1
        for start in range(0, len(prepared), batch_size):
            chunk = prepared[start:start + batch_size]
            coeff_matrix = np.stack([item[1] for item in chunk], axis=0)
            logits = np.clip(coeff_matrix @ proto_roi, -88.0, 88.0)
            logits = logits.reshape(len(chunk), roi_h, roi_w)
            for row, (index, _coeffs, bounds) in enumerate(chunk):
                px1, py1, px2, py2 = bounds
                crop_logits = logits[
                    row,
                    py1 - uy1:py2 - uy1,
                    px1 - ux1:px2 - ux1,
                ]
                if crop_logits.size == 0:
                    continue
                probabilities = 1.0 / (1.0 + np.exp(-crop_logits))
                x1, y1, x2, y2 = candidates[index].bbox
                if x2 <= x1 or y2 <= y1:
                    continue
                resized = cv2.resize(
                    probabilities,
                    (x2 - x1, y2 - y1),
                    interpolation=cv2.INTER_LINEAR,
                )
                decoded[index] = self._decode_text_mask_hysteresis(resized)
        return decoded

    def _to_evidence(
        self,
        candidate: _SegCandidate,
        mask: np.ndarray | None,
    ) -> DetectionEvidence:
        class_name = self.class_names[candidate.class_id]
        verified = bool(mask is not None and np.any(mask > 0))
        return DetectionEvidence(
            bbox=candidate.bbox,
            confidence=min(float(candidate.confidence), DETECTOR_CONFIDENCE_MAX),
            semantic=class_name,
            source=self.source,
            evidence_kind=EvidenceKind.MASK if verified else EvidenceKind.BOX,
            mask=mask,
            class_id=int(candidate.class_id),
            class_name=class_name,
            metadata={"source_model": self.source_model},
        )

    def _merge_evidence(
        self,
        items: list[DetectionEvidence],
    ) -> DetectionEvidence:
        base = max(items, key=lambda item: item.confidence)
        masked = [item for item in items if item.verified_mask]
        if not masked:
            return base

        x1 = min(item.bbox[0] for item in masked)
        y1 = min(item.bbox[1] for item in masked)
        x2 = max(item.bbox[2] for item in masked)
        y2 = max(item.bbox[3] for item in masked)
        merged = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        for item in masked:
            ix1, iy1, ix2, iy2 = item.bbox
            mask = item.mask
            if mask is None:
                continue
            expected = (iy2 - iy1, ix2 - ix1)
            if mask.shape != expected:
                mask = cv2.resize(
                    mask,
                    (expected[1], expected[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            dx, dy = ix1 - x1, iy1 - y1
            target = merged[dy:dy + expected[0], dx:dx + expected[1]]
            merged[dy:dy + expected[0], dx:dx + expected[1]] = np.maximum(
                target,
                mask,
            )

        return DetectionEvidence(
            bbox=(x1, y1, x2, y2),
            confidence=base.confidence,
            semantic=base.semantic,
            source=base.source,
            evidence_kind=EvidenceKind.MASK,
            mask=merged,
            class_id=base.class_id,
            class_name=base.class_name,
            metadata=base.metadata,
        )

    def _suppress(
        self,
        candidates: list[_SegCandidate],
        prototypes: np.ndarray,
    ) -> list[DetectionEvidence]:
        result: list[DetectionEvidence] = []
        by_class: dict[int, list[_SegCandidate]] = {}
        for candidate in candidates:
            by_class.setdefault(candidate.class_id, []).append(candidate)

        for members in by_class.values():
            rects = [
                [
                    item.bbox[0],
                    item.bbox[1],
                    item.bbox[2] - item.bbox[0],
                    item.bbox[3] - item.bbox[1],
                ]
                for item in members
            ]
            scores = [
                max(self.conf_threshold, float(item.confidence))
                for item in members
            ]
            raw_indices = cv2.dnn.NMSBoxes(
                rects,
                scores,
                self.conf_threshold,
                BUBBLE_IOU_THRESHOLD,
            )
            kept = [int(index) for index in np.asarray(raw_indices).reshape(-1)]
            if not kept:
                continue

            buckets = {index: [index] for index in kept}
            kept_set = set(kept)
            for index, candidate in enumerate(members):
                if index in kept_set:
                    continue
                target = max(
                    kept,
                    key=lambda kept_index: self._candidate_iou(
                        candidate,
                        members[kept_index],
                    ),
                )
                if (
                    self._candidate_iou(candidate, members[target])
                    > BUBBLE_IOU_THRESHOLD
                ):
                    buckets[target].append(index)

            for kept_index in kept:
                contributors = buckets.get(kept_index, [kept_index])
                masks = self._decode_masks(members, contributors, prototypes)
                decoded = [
                    self._to_evidence(members[index], masks.get(index))
                    for index in contributors
                ]
                result.append(self._merge_evidence(decoded))

        result.sort(key=lambda item: item.confidence, reverse=True)
        return result

    def decode(self, raw: RawModelOutput) -> list[DetectionEvidence]:
        if len(raw.tensors) < 2 or raw.transform is None:
            return []
        output0 = np.squeeze(np.asarray(raw.tensors[0]))
        prototypes_raw = np.asarray(raw.tensors[1])
        if output0.ndim == 1:
            output0 = output0[np.newaxis, :]
        if output0.ndim == 2 and output0.shape[0] < output0.shape[1]:
            output0 = output0.T
        if output0.ndim != 2 or output0.shape[0] == 0:
            return []
        if prototypes_raw.ndim != 4 or prototypes_raw.shape[0] != 1:
            return []
        prototypes = prototypes_raw[0]

        class_end = 4 + len(self.class_names)
        class_scores = output0[:, 4:class_end].astype(np.float32, copy=False)
        class_ids = np.argmax(class_scores, axis=1).astype(np.int32, copy=False)
        confidences = class_scores[np.arange(class_scores.shape[0]), class_ids]
        keep = np.flatnonzero(confidences >= self.conf_threshold)
        if keep.size == 0:
            return []

        selected = output0[keep]
        selected_conf = confidences[keep]
        selected_classes = class_ids[keep]
        coeff_start = class_end
        coeff_end = coeff_start + prototypes.shape[0]
        candidates: list[_SegCandidate] = []
        transform = raw.transform
        for row in range(selected.shape[0]):
            cx, cy, box_w, box_h = (float(value) for value in selected[row, :4])
            source_box = transform.page_box_from_canvas(
                (
                    cx - box_w / 2.0,
                    cy - box_h / 2.0,
                    cx + box_w / 2.0,
                    cy + box_h / 2.0,
                )
            )
            x1, y1, x2, y2 = source_box
            if (
                (x2 - x1) < DETECTOR_MIN_BOX_SIDE
                or (y2 - y1) < DETECTOR_MIN_BOX_SIDE
            ):
                continue

            pad = int(DETECTOR_TEXT_MASK_DECODE_PAD)
            source_box = (
                max(transform.offset_x, x1 - pad),
                max(transform.offset_y, y1 - pad),
                min(transform.offset_x + transform.src_w, x2 + pad),
                min(transform.offset_y + transform.src_h, y2 + pad),
            )
            candidates.append(
                _SegCandidate(
                    bbox=tuple(int(value) for value in source_box),
                    confidence=float(selected_conf[row]),
                    class_id=int(selected_classes[row]),
                    geometry=MaskDecodeGeometry(transform, source_box),
                    mask_coeffs=selected[row, coeff_start:coeff_end].copy(),
                )
            )
        return self._suppress(candidates, prototypes)
