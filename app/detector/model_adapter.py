from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from app.detector.evidence import DetectionEvidence, EvidenceKind
from app.detector.page_context import PageContext


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
