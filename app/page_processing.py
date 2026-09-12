from __future__ import annotations

import copy
import time
import uuid
from pathlib import Path

import cv2

from app.detector.bubble_detector import BubbleBox
from app.detector.combined_detector import CombinedTextDetector
from app.image_io import encode_mask, read_image, write_image
from app.manifest_utils import assign_stable_detector_box_ids
from app.mask_store import decode_mask_value
from app.parameters import (
    DETECTION_CONTENT_STD_MIN,
    DETECTOR_FINAL_NMS_IOU,
    DETECTOR_RESIDUE_VERIFY_ENABLED,
)


class PageProcessingMixin:
    def _process_page(
        self,
        img_path: Path,
        processed_dir: Path,
        excluded_regions: list[dict] | None = None,
        existing_boxes: list[dict] | None = None,
        stitch_core: dict | None = None,
        supplemental_detections: list[BubbleBox] | None = None,
        seam_context_unavailable: bool = False,
        *,
        parallel_detectors: bool = False,
    ) -> dict:
        started_at = time.perf_counter()
        read_started_at = started_at
        image = read_image(img_path)
        read_ms = (time.perf_counter() - read_started_at) * 1000.0
        detector_kwargs = {"parallel": parallel_detectors}

        core_bounds: tuple[int, int] | None = None
        if isinstance(stitch_core, dict):
            try:
                core_y1 = max(0, min(int(stitch_core.get("core_y1", 0)), image.shape[0]))
                core_y2 = max(core_y1, min(int(stitch_core.get("core_y2", image.shape[0])), image.shape[0]))
            except (TypeError, ValueError):
                core_y1, core_y2 = 0, image.shape[0]
            if core_y2 > core_y1:
                core_bounds = (core_y1, core_y2)

        use_core_detector = (
            core_bounds is not None
            and core_bounds != (0, image.shape[0])
        )
        detect_started_at = time.perf_counter()
        if use_core_detector:
            core_y1, core_y2 = core_bounds
            core_image = image[core_y1:core_y2, :]
            detected = [
                self._shift_detection_y(box, core_y1)
                for box in self.detector.detect(core_image, **detector_kwargs)
            ]
        else:
            detected = self.detector.detect(image, **detector_kwargs)
        detector_metrics = self.detector.last_metrics()

        if supplemental_detections:
            detected = CombinedTextDetector._apply_final_nms(
                detected + list(supplemental_detections),
                iou_threshold=DETECTOR_FINAL_NMS_IOU,
            )
        # Exclusion rectangles protect pixels rather than suppressing
        # detector/OCR evidence. Automatic erase authority is clipped later.
        detect_ms = (time.perf_counter() - detect_started_at) * 1000.0

        existing_boxes = copy.deepcopy(existing_boxes or [])
        detector_records = [
            {
                "x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2,
                "confidence": b.confidence, "_mask_array": b.mask,
                "source_model": b.source_model, "class_id": b.class_id,
                "class_name": b.class_name, "semantic_type": b.semantic_type,
                "mask_source": b.mask_source, "safe_to_inpaint": bool(b.safe_to_inpaint),
                "ocr_eligible": bool(b.ocr_eligible), "needs_review": bool(b.needs_review),
                "source_role": b.source_role, "deferred_reason": b.deferred_reason,
            }
            for b in detected
        ]
        assign_stable_detector_box_ids(detector_records, existing_boxes)
        old_by_id = {str(b.get("id")): b for b in existing_boxes if isinstance(b, dict) and b.get("id")}

        effective_boxes: list[BubbleBox] = []
        for record in detector_records:
            old = old_by_id.get(str(record.get("id")))
            if old is not None:
                if old.get("removed"):
                    record["removed"] = True
                if old.get("geometry_overridden"):
                    record["geometry_overridden"] = True
                    if isinstance(old.get("detector_anchor"), dict):
                        record["detector_anchor"] = copy.deepcopy(old["detector_anchor"])
                    for key in ("x1", "y1", "x2", "y2"):
                        record[key] = int(old[key])
                    record["_mask_array"] = None
                    record["safe_to_inpaint"] = False
                    # Geometry explicitly confirmed by the user remains a valid
                    # non-destructive OCR target even though rectangle inpaint
                    # authority is handled separately below.
                    record["ocr_eligible"] = True
                    record["needs_review"] = True

            if core_bounds is not None:
                core_y1, core_y2 = core_bounds
                record_y1 = int(record["y1"])
                record_y2 = int(record["y2"])
                overlap_only = record_y2 <= core_y1 or record_y1 >= core_y2
                if overlap_only:
                    record["overlap_context_only"] = True
                    record["ocr_eligible"] = False
                else:
                    record.pop("overlap_context_only", None)

            skip_auto_overlap_inpaint = bool(
                record.get("overlap_context_only")
                and not record.get("geometry_overridden")
            )
            if (
                not record.get("removed")
                and not skip_auto_overlap_inpaint
                and not record.get("deferred_reason")
                and (record.get("safe_to_inpaint") or record.get("geometry_overridden"))
            ):
                _effective = BubbleBox(
                    int(record["x1"]), int(record["y1"]), int(record["x2"]), int(record["y2"]),
                    float(record.get("confidence", 1.0)), record.get("_mask_array"),
                    source_model=str(record.get("source_model") or "unknown"),
                    class_id=int(record.get("class_id") or 0),
                    class_name=str(record.get("class_name") or "unknown"),
                    semantic_type=str(record.get("semantic_type") or "unknown"),
                    mask_source=str(record.get("mask_source") or "none"),
                    safe_to_inpaint=True, ocr_eligible=bool(record.get("ocr_eligible")),
                    needs_review=bool(record.get("needs_review")),
                    source_role=str(record.get("source_role") or "unknown"),
                    deferred_reason=record.get("deferred_reason"),
                )
                if record.get("geometry_overridden"):
                    _effective.allow_rectangle_fallback = True
                effective_boxes.append(_effective)

        for old in existing_boxes:
            if not isinstance(old, dict) or not old.get("manual") or old.get("removed"):
                continue
            box_h = int(old.get("y2", 0)) - int(old.get("y1", 0))
            box_w = int(old.get("x2", 0)) - int(old.get("x1", 0))
            if box_h <= 0 or box_w <= 0:
                continue
            mask_arr = decode_mask_value(old.get("mask"))
            if mask_arr is not None and mask_arr.shape != (box_h, box_w):
                try:
                    mask_arr = cv2.resize(mask_arr, (box_w, box_h), interpolation=cv2.INTER_NEAREST)
                except Exception:
                    mask_arr = None
            effective_boxes.append(BubbleBox(
                int(old["x1"]), int(old["y1"]), int(old["x2"]), int(old["y2"]),
                float(old.get("confidence", 1.0)), mask_arr,
            ))

        auto_inpaint_started_at = time.perf_counter()
        clean_image = self.inpainter.inpaint(
            image,
            effective_boxes,
            protected_regions=excluded_regions,
        )
        auto_inpaint_ms = (time.perf_counter() - auto_inpaint_started_at) * 1000.0
        auto_inpaint_metrics = self.inpainter.last_metrics()

        auto_clean_path = self._auto_clean_path(processed_dir, img_path)
        manual_mask_path = self._manual_mask_path(processed_dir, img_path)
        manual_lama_mask_path = self._manual_mask_path(
            processed_dir, img_path, force_lama=True
        )
        tmp_auto_clean_path = None
        if (
            auto_clean_path.exists()
            or manual_mask_path.exists()
            or manual_lama_mask_path.exists()
        ):
            tmp_auto_clean_path = processed_dir / f"auto_clean_{img_path.name}.{uuid.uuid4().hex[:12]}.tmp.png"
            write_image(tmp_auto_clean_path, clean_image)

        manual_mask_posix = None
        manual_lama_mask_posix = None
        manual_inpaint_ms = 0.0
        manual_inpaint_metrics: dict[str, int] = {}
        manual_passes = (
            ("manual_mask", manual_mask_path, False),
            ("manual_lama_mask", manual_lama_mask_path, True),
        )
        for mask_field, mask_path, force_lama in manual_passes:
            manual_mask = self._read_manual_mask(mask_path, clean_image.shape[:2])
            if manual_mask is None:
                continue
            manual_inpaint_started_at = time.perf_counter()
            clean_image = self.inpainter.inpaint_mask(
                clean_image.copy(), manual_mask, force_lama=force_lama
            )
            manual_inpaint_ms += (
                time.perf_counter() - manual_inpaint_started_at
            ) * 1000.0
            for metric_name, metric_value in self.inpainter.last_metrics().items():
                manual_inpaint_metrics[metric_name] = (
                    manual_inpaint_metrics.get(metric_name, 0) + int(metric_value)
                )
            if mask_field == "manual_mask":
                manual_mask_posix = mask_path.as_posix()
            else:
                manual_lama_mask_posix = mask_path.as_posix()

        residue_started_at = time.perf_counter()
        residue_boxes: list[BubbleBox] = []
        if DETECTOR_RESIDUE_VERIFY_ENABLED and effective_boxes:
            # This is deliberately after every automatic/manual inpaint pass.
            # A detector mask authorizes deletion; it does not certify that the
            # resulting pixels no longer look like text.
            residue_boxes = self.detector.verify_post_inpaint_residue(
                clean_image,
                effective_boxes,
            )
        residue_verify_ms = (time.perf_counter() - residue_started_at) * 1000.0

        tmp_clean_path = processed_dir / f"clean_{img_path.name}.{uuid.uuid4().hex[:12]}.tmp.png"
        write_started_at = time.perf_counter()
        write_image(tmp_clean_path, clean_image)
        write_ms = (time.perf_counter() - write_started_at) * 1000.0

        decision_fields = (
            "x1", "y1", "x2", "y2", "confidence", "source_model",
            "source_role", "class_name", "semantic_type", "deferred_reason",
        )
        unverified_regions = [
            {k: record.get(k) for k in decision_fields}
            for record in detector_records
            if record.get("needs_review") or not record.get("safe_to_inpaint")
        ]
        deferred_regions = [
            {k: record.get(k) for k in decision_fields}
            for record in detector_records
            if record.get("deferred_reason")
        ]
        residue_regions = [
            {k: getattr(box, k) for k in decision_fields}
            for box in residue_boxes
        ]
        detection_issues = []
        if seam_context_unavailable:
            detection_issues.append("seam_context_unavailable")
        if unverified_regions:
            detection_issues.append("unverified_regions")
        if deferred_regions:
            detection_issues.append("deferred_regions")
        if residue_regions:
            detection_issues.append("post_inpaint_text_residue")
        if not detector_records:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if float(gray.std()) > DETECTION_CONTENT_STD_MIN:
                detection_issues.append("content_heavy_zero_box")
        detection_state = "needs_review" if detection_issues else "verified"
        inpainter = self.inpainter
        inpaint_model_path = getattr(inpainter, "lama_model_path", None)
        inpaint_session_loaded = bool(getattr(inpainter, "session_loaded", False))
        inpaint_dynamic = bool(
            getattr(inpainter, "dynamic_lama", False)
            if inpaint_session_loaded
            else getattr(inpainter, "_prefer_dynamic", False)
        )
        serialized_inference = bool(
            getattr(inpainter, "serialized_inference", not inpaint_dynamic)
        )
        processing_metrics = {
            "timing_ms": {
                "read": round(read_ms, 3),
                "detect": round(detect_ms, 3),
                "auto_inpaint": round(auto_inpaint_ms, 3),
                "manual_inpaint": round(manual_inpaint_ms, 3),
                "write": round(write_ms, 3),
                "residue_verify": round(residue_verify_ms, 3),
                "total": round((time.perf_counter() - started_at) * 1000.0, 3),
            },
            "detector": {
                **detector_metrics,
                "records": len(detector_records),
                "authorized": len(effective_boxes),
                "review_only": len(unverified_regions),
                "deferred": len(deferred_regions),
                "post_inpaint_residue": len(residue_regions),
            },
            "auto_inpaint": auto_inpaint_metrics,
            "manual_inpaint": manual_inpaint_metrics,
            "model": {
                "active": Path(inpaint_model_path).name if inpaint_model_path else None,
                "dynamic": inpaint_dynamic,
                "serialized_inference": serialized_inference,
                "serialization_scope": (
                    "global" if serialized_inference else None
                ),
                "session_loaded": inpaint_session_loaded,
            },
        }

        res = {
            "tmp_clean": tmp_clean_path.as_posix(),
            "boxes": [
                {
                    **{k: v for k, v in record.items() if k != "_mask_array"},
                    "mask": encode_mask(record.get("_mask_array")),
                }
                for record in detector_records
            ],
            "detection_state": detection_state,
            "detection_issues": detection_issues,
            "unverified_regions": unverified_regions,
            "deferred_regions": deferred_regions,
            "residue_regions": residue_regions,
            "cleanup_verified": not bool(detection_issues),
            "needs_review": bool(detection_issues),
            "processing_metrics": processing_metrics,
        }
        if tmp_auto_clean_path is not None:
            res["tmp_auto_clean"] = tmp_auto_clean_path.as_posix()
        if manual_mask_posix:
            res["manual_mask"] = manual_mask_posix
        if manual_lama_mask_posix:
            res["manual_lama_mask"] = manual_lama_mask_posix
        return res
