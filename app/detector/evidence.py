from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Mapping

import numpy as np


class AuthorityLevel(str, Enum):
    """Cleanup authority is a policy result, never a model capability."""

    PROPOSAL_ONLY = "proposal_only"
    REVIEW_ONLY = "review_only"
    VERIFIED_TEXT_MASK = "verified_text_mask"


class EvidenceKind(str, Enum):
    BOX = "box"
    MASK = "mask"
    STRUCTURAL = "structural"
    RECOVERY = "recovery"


@dataclass(frozen=True)
class DetectionEvidence:
    bbox: tuple[int, int, int, int]
    confidence: float
    semantic: str
    source: str
    evidence_kind: EvidenceKind
    mask: np.ndarray | None = None
    class_id: int | None = None
    class_name: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def width(self) -> int:
        return int(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> int:
        return int(self.bbox[3] - self.bbox[1])

    @property
    def verified_mask(self) -> bool:
        return (
            self.width > 0
            and self.height > 0
            and self.mask is not None
            and self.mask.ndim == 2
            and self.mask.shape == (self.height, self.width)
            and bool(np.any(self.mask > 0))
        )


@dataclass(frozen=True)
class AuthorityDecision:
    level: AuthorityLevel
    mask_source: str
    safe_to_inpaint: bool
    ocr_eligible: bool
    needs_review: bool


class AuthorityPolicy:
    """Central cleanup policy for detector evidence.

    Adapters describe what a model observed. This policy decides whether that
    evidence may modify artwork. A mask existing is intentionally insufficient:
    the producer must be explicitly trusted for destructive text masks.
    """

    def __init__(
        self,
        *,
        trusted_text_mask_sources: frozenset[str] | None = None,
    ) -> None:
        self.trusted_text_mask_sources = (
            frozenset({"yolov8_text_segmenter"})
            if trusted_text_mask_sources is None
            else frozenset(trusted_text_mask_sources)
        )

    def decide(self, evidence: DetectionEvidence) -> AuthorityDecision:
        if (
            evidence.verified_mask
            and evidence.source in self.trusted_text_mask_sources
        ):
            return AuthorityDecision(
                level=AuthorityLevel.VERIFIED_TEXT_MASK,
                mask_source="text_segmenter",
                safe_to_inpaint=True,
                ocr_eligible=True,
                needs_review=False,
            )

        if evidence.verified_mask:
            return AuthorityDecision(
                level=AuthorityLevel.REVIEW_ONLY,
                mask_source="model",
                safe_to_inpaint=False,
                ocr_eligible=False,
                needs_review=True,
            )

        return AuthorityDecision(
            level=AuthorityLevel.PROPOSAL_ONLY,
            mask_source="none",
            safe_to_inpaint=False,
            ocr_eligible=False,
            needs_review=True,
        )

    def apply_legacy_yolov8_box(self, box, *, producer_role: str):
        """Compatibility bridge preserving current YoloDetector semantics.

        This keeps production output neutral while moving the destructive
        decision out of the decoder. New adapters should emit
        DetectionEvidence directly instead of calling this bridge.
        """

        verified = bool(getattr(box, "verified_mask", False))
        observed_role = str(getattr(box, "source_role", "unknown"))
        source = (
            "yolov8_text_segmenter"
            if observed_role == "text_segmenter"
            else f"legacy_yolov8_{observed_role}"
        )
        evidence = DetectionEvidence(
            bbox=(int(box.x1), int(box.y1), int(box.x2), int(box.y2)),
            confidence=float(box.confidence),
            semantic=str(getattr(box, "semantic_type", "unknown")),
            source=source,
            evidence_kind=EvidenceKind.MASK if verified else EvidenceKind.BOX,
            mask=getattr(box, "mask", None),
            class_id=int(getattr(box, "class_id", 0)),
            class_name=str(getattr(box, "class_name", "unknown")),
        )
        decision = self.decide(evidence)
        return replace(
            box,
            mask_source=decision.mask_source,
            safe_to_inpaint=decision.safe_to_inpaint,
            ocr_eligible=decision.ocr_eligible,
            needs_review=decision.needs_review,
            source_role=str(producer_role),
        )


DEFAULT_AUTHORITY_POLICY = AuthorityPolicy()
