from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one source match, got {count}")
    return text.replace(old, new, 1)


def patch_bubble_detector() -> None:
    path = ROOT / "app/detector/bubble_detector.py"
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        '    source_role: str = "unknown"\n',
        '    source_role: str = "unknown"\n    deferred_reason: str | None = None\n',
        "BubbleBox deferred reason",
    )
    old = '''    def detect(self, image: np.ndarray) -> list[BubbleBox]:\n        h, w = image.shape[:2]\n        if h <= INPUT_SIZE * DETECTOR_TALL_IMAGE_FACTOR:\n            boxes = self._detect_single(image, 0, 0)\n        else:\n            all_boxes = []\n            step = INPUT_SIZE - SLICE_OVERLAP\n            y = 0\n            while y < h:\n                slice_h = min(INPUT_SIZE, h - y)\n                slice_img = image[y:y + slice_h, :]\n                boxes = self._detect_single(slice_img, 0, y)\n                all_boxes.extend(boxes)\n                if y + slice_h >= h:\n                    break\n                y += step\n\n            if self.model_role == "text_segmenter":\n                all_boxes.extend(self._detect_single_plain(image, 0, 0))\n\n            boxes = self._nms_boxes(all_boxes)\n\n        return [self._with_semantics(b) for b in self._filter_invalid(boxes, w, h)]\n\n    @staticmethod\n    def _filter_invalid(boxes: list[BubbleBox], img_w: int, img_h: int) -> list[BubbleBox]:\n        result = []\n        page_area = img_w * img_h\n        for b in boxes:\n            box_w = b.x2 - b.x1\n            box_h = b.y2 - b.y1\n            if box_w <= 0 or box_h <= 0:\n                continue\n            if box_w > img_w * MAX_BOX_WIDTH_RATIO:\n                continue\n            if (box_w * box_h) > page_area * MAX_BOX_AREA_RATIO:\n                continue\n            aspect = box_w / box_h\n            if aspect > MAX_ASPECT_RATIO or aspect < 1 / MAX_ASPECT_RATIO:\n                continue\n            result.append(b)\n        return result\n'''
    new = '''    def detect(self, image: np.ndarray) -> list[BubbleBox]:\n        h, w = image.shape[:2]\n        if h <= INPUT_SIZE * DETECTOR_TALL_IMAGE_FACTOR:\n            boxes = self._detect_single(image, 0, 0)\n        else:\n            all_boxes = []\n            step = INPUT_SIZE - SLICE_OVERLAP\n            y = 0\n            while y < h:\n                slice_h = min(INPUT_SIZE, h - y)\n                slice_img = image[y:y + slice_h, :]\n                boxes = self._detect_single(slice_img, 0, y)\n                all_boxes.extend(boxes)\n                if y + slice_h >= h:\n                    break\n                y += step\n\n            if self.model_role == "text_segmenter":\n                all_boxes.extend(self._detect_single_plain(image, 0, 0))\n\n            boxes = self._nms_boxes(all_boxes)\n\n        semantic_boxes = [self._with_semantics(b) for b in boxes]\n        return self._filter_invalid(semantic_boxes, w, h)\n\n    @staticmethod\n    def _filter_invalid(\n        boxes: list[BubbleBox], img_w: int, img_h: int\n    ) -> list[BubbleBox]:\n        \"\"\"Retain oversized/aspect outliers as review evidence instead of dropping them.\n\n        Postprocess already rejects non-positive boxes before this point. Width,\n        page-area and aspect limits are policy heuristics, not proof that model\n        evidence is false. They therefore revoke automatic erase authority but\n        preserve the detected region for review with an explicit reason.\n        \"\"\"\n        result = []\n        page_area = max(1, img_w * img_h)\n        for b in boxes:\n            box_w = b.x2 - b.x1\n            box_h = b.y2 - b.y1\n            if box_w <= 0 or box_h <= 0:\n                continue\n            reasons: list[str] = []\n            if box_w > img_w * MAX_BOX_WIDTH_RATIO:\n                reasons.append("box_width_limit")\n            if (box_w * box_h) > page_area * MAX_BOX_AREA_RATIO:\n                reasons.append("box_area_limit")\n            aspect = box_w / box_h\n            if aspect > MAX_ASPECT_RATIO or aspect < 1 / MAX_ASPECT_RATIO:\n                reasons.append("box_aspect_limit")\n            if reasons:\n                result.append(\n                    replace(\n                        b,\n                        safe_to_inpaint=False,\n                        ocr_eligible=bool(\n                            b.ocr_eligible\n                            or b.source_role == "text_segmenter"\n                            or b.semantic_type == "free_text"\n                        ),\n                        needs_review=True,\n                        deferred_reason="|".join(reasons),\n                    )\n                )\n            else:\n                result.append(b)\n        return result\n'''
    text = replace_once(text, old, new, "detector retention policy")
    path.write_text(text, encoding="utf-8")


def patch_combined_detector() -> None:
    path = ROOT / "app/detector/combined_detector.py"
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        '''    def _classify(self, box: BubbleBox) -> BubbleBox:\n        if box.verified_mask and box.source_role == "text_segmenter":\n''',
        '''    def _classify(self, box: BubbleBox) -> BubbleBox:\n        if box.deferred_reason:\n            return replace(\n                box,\n                safe_to_inpaint=False,\n                ocr_eligible=bool(\n                    box.ocr_eligible\n                    or box.source_role == "text_segmenter"\n                    or box.semantic_type == "free_text"\n                ),\n                needs_review=True,\n            )\n        if box.verified_mask and box.source_role == "text_segmenter":\n''',
        "combined deferred authority",
    )
    old_group = '''                merged_mask = self._merge_masks(group, min_x, min_y, max_x, max_y)\n                seed = max(group, key=lambda t: t.confidence)\n                safe = merged_mask is not None and bool(np.any(merged_mask > 0))\n                merged = replace(seed, x1=int(min_x), y1=int(min_y), x2=int(max_x), y2=int(max_y),\n                                 mask=merged_mask, semantic_type=b.semantic_type,\n                                 mask_source="text_segmenter" if safe else "none",\n                                 safe_to_inpaint=safe, ocr_eligible=True, needs_review=not safe)\n'''
    new_group = '''                merged_mask = self._merge_masks(group, min_x, min_y, max_x, max_y)\n                seed = max(group, key=lambda t: t.confidence)\n                deferred_reason = next(\n                    (t.deferred_reason for t in group if t.deferred_reason),\n                    None,\n                )\n                safe = (\n                    deferred_reason is None\n                    and merged_mask is not None\n                    and bool(np.any(merged_mask > 0))\n                )\n                merged = replace(\n                    seed,\n                    x1=int(min_x), y1=int(min_y), x2=int(max_x), y2=int(max_y),\n                    mask=merged_mask, semantic_type=b.semantic_type,\n                    mask_source="text_segmenter" if safe else "none",\n                    safe_to_inpaint=safe, ocr_eligible=True, needs_review=not safe,\n                    deferred_reason=deferred_reason,\n                )\n'''
    text = replace_once(text, old_group, new_group, "bubble text group deferral")
    text = text.replace(
        '''                    if cluster_area > img_w * img_h * MAX_BOX_AREA_RATIO:\n                        continue\n                    merged_mask = self._merge_masks([b], min_x, min_y, max_x, max_y)\n''',
        '''                    if cluster_area > img_w * img_h * MAX_BOX_AREA_RATIO:\n                        # Padding is optional layout context. Never lose the\n                        # detector's original mask just because the padded box\n                        # would exceed the broad area heuristic.\n                        final_clusters.append(replace(b))\n                        continue\n                    merged_mask = self._merge_masks([b], min_x, min_y, max_x, max_y)\n''',
        1,
    )
    text = text.replace(
        '''                    if cluster_area > img_w * img_h * MAX_BOX_AREA_RATIO:\n                        continue\n                    merged_mask = self._merge_masks(sub, min_x, min_y, max_x, max_y)\n''',
        '''                    if cluster_area > img_w * img_h * MAX_BOX_AREA_RATIO:\n                        # Preserve every child mask as an independent region.\n                        # Downstream inpaint can safely split these again; a\n                        # broad grouping heuristic must never erase evidence.\n                        final_clusters.extend(replace(member) for member in sub)\n                        continue\n                    merged_mask = self._merge_masks(sub, min_x, min_y, max_x, max_y)\n''',
        1,
    )
    if "Preserve every child mask as an independent region." not in text:
        raise SystemExit("free-text oversized group source contract changed")
    path.write_text(text, encoding="utf-8")


def patch_inpainter() -> None:
    path = ROOT / "app/inpaint/lama_inpainter.py"
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        '            "skipped_clusters": 0,\n',
        '            "skipped_clusters": 0,\n            "split_clusters": 0,\n',
        "inpaint split metric",
    )
    old = '''        clusters = self._cluster_boxes(boxes)\n        self._metrics_local.value["clusters"] = len(clusters)\n\n        for cluster in clusters:\n            x1 = min(b.x1 for b in cluster)\n            y1 = min(b.y1 for b in cluster)\n            x2 = max(b.x2 for b in cluster)\n            y2 = max(b.y2 for b in cluster)\n\n            if len(cluster) > 1 and (x2 - x1) * (y2 - y1) > w * h * MAX_BOX_AREA_RATIO:\n                self._metric_add("skipped_clusters")\n                logger.warning(f"Skipping multi-box cluster ({len(cluster)} boxes) at ({x1}, {y1}, {x2}, {y2}): area exceeds MAX_BOX_AREA_RATIO")\n                continue\n\n            crop_box = self._compute_crop_region(x1, y1, x2, y2, w, h)\n'''
    new = '''        raw_clusters = self._cluster_boxes(boxes)\n        clusters: list[list[BubbleBox]] = []\n        for cluster in raw_clusters:\n            parts = self._split_oversized_cluster_area(cluster, w, h)\n            if len(parts) > 1:\n                self._metric_add("split_clusters", len(parts) - 1)\n            clusters.extend(parts)\n        self._metrics_local.value["clusters"] = len(clusters)\n\n        for cluster in clusters:\n            x1 = min(b.x1 for b in cluster)\n            y1 = min(b.y1 for b in cluster)\n            x2 = max(b.x2 for b in cluster)\n            y2 = max(b.y2 for b in cluster)\n\n            crop_box = self._compute_crop_region(x1, y1, x2, y2, w, h)\n'''
    text = replace_once(text, old, new, "inpaint oversized cluster retention")
    marker = '''    @staticmethod\n    def _split_cluster_lines(cluster: list[BubbleBox], avg_h: float) -> list[list[BubbleBox]]:\n'''
    helper = '''    @staticmethod\n    def _split_oversized_cluster_area(\n        cluster: list[BubbleBox],\n        img_w: int,\n        img_h: int,\n    ) -> list[list[BubbleBox]]:\n        \"\"\"Split an oversized group without dropping any child evidence.\"\"\"\n        if not cluster:\n            return []\n        page_limit = max(1.0, float(img_w * img_h) * MAX_BOX_AREA_RATIO)\n        pending = [list(cluster)]\n        result: list[list[BubbleBox]] = []\n        while pending:\n            group = pending.pop()\n            x1 = min(b.x1 for b in group)\n            y1 = min(b.y1 for b in group)\n            x2 = max(b.x2 for b in group)\n            y2 = max(b.y2 for b in group)\n            area = max(0, x2 - x1) * max(0, y2 - y1)\n            if len(group) <= 1 or area <= page_limit:\n                result.append(group)\n                continue\n\n            span_x = x2 - x1\n            span_y = y2 - y1\n            if span_x >= span_y:\n                ordered = sorted(group, key=lambda b: ((b.x1 + b.x2), b.y1, b.x1))\n            else:\n                ordered = sorted(group, key=lambda b: ((b.y1 + b.y2), b.x1, b.y1))\n            midpoint = max(1, len(ordered) // 2)\n            left = ordered[:midpoint]\n            right = ordered[midpoint:]\n            if not right:\n                result.extend([[b] for b in ordered])\n                continue\n            pending.append(right)\n            pending.append(left)\n\n        result.sort(\n            key=lambda group: (\n                min(b.y1 for b in group),\n                min(b.x1 for b in group),\n            )\n        )\n        return result\n\n'''
    text = replace_once(text, marker, helper + marker, "inpaint split helper")
    text = replace_once(
        text,
        '                    source_role=b.source_role,\n',
        '                    source_role=b.source_role,\n                    deferred_reason=b.deferred_reason,\n',
        "local deferred provenance",
    )
    path.write_text(text, encoding="utf-8")


def patch_pipeline() -> None:
    path = ROOT / "app/pipeline.py"
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        '''                    target_page["unverified_regions"] = list(\n                        page_data.get("unverified_regions") or []\n                    )\n                    target_page["needs_review"] = bool(page_data.get("needs_review"))\n''',
        '''                    target_page["unverified_regions"] = list(\n                        page_data.get("unverified_regions") or []\n                    )\n                    target_page["deferred_regions"] = list(\n                        page_data.get("deferred_regions") or []\n                    )\n                    target_page["needs_review"] = bool(page_data.get("needs_review"))\n''',
        "persist deferred decisions",
    )
    text = replace_once(
        text,
        '''                "mask_source": b.mask_source, "safe_to_inpaint": bool(b.safe_to_inpaint),\n                "ocr_eligible": bool(b.ocr_eligible), "needs_review": bool(b.needs_review),\n''',
        '''                "mask_source": b.mask_source, "safe_to_inpaint": bool(b.safe_to_inpaint),\n                "ocr_eligible": bool(b.ocr_eligible), "needs_review": bool(b.needs_review),\n                "source_role": b.source_role, "deferred_reason": b.deferred_reason,\n''',
        "serialize detector decisions",
    )
    text = replace_once(
        text,
        '''                and (record.get("safe_to_inpaint") or record.get("geometry_overridden"))\n            ):\n''',
        '''                and not record.get("deferred_reason")\n                and (record.get("safe_to_inpaint") or record.get("geometry_overridden"))\n            ):\n''',
        "deferred automatic authority",
    )
    text = replace_once(
        text,
        '''                    safe_to_inpaint=True, ocr_eligible=bool(record.get("ocr_eligible")),\n                    needs_review=bool(record.get("needs_review")),\n                )\n''',
        '''                    safe_to_inpaint=True, ocr_eligible=bool(record.get("ocr_eligible")),\n                    needs_review=bool(record.get("needs_review")),\n                    source_role=str(record.get("source_role") or "unknown"),\n                    deferred_reason=record.get("deferred_reason"),\n                )\n''',
        "effective detector provenance",
    )
    text = replace_once(
        text,
        '''                safe_to_inpaint=safe_to_inpaint,\n                ocr_eligible=bool(box.get("ocr_eligible")),\n                needs_review=bool(box.get("needs_review")),\n            )\n''',
        '''                safe_to_inpaint=safe_to_inpaint,\n                ocr_eligible=bool(box.get("ocr_eligible")),\n                needs_review=bool(box.get("needs_review")),\n                source_role=str(box.get("source_role") or "unknown"),\n                deferred_reason=box.get("deferred_reason"),\n            )\n''',
        "reinpaint decision provenance",
    )
    old_unverified = '''        unverified_regions = [\n            {k: record.get(k) for k in ("x1", "y1", "x2", "y2", "confidence", "source_model", "class_name", "semantic_type")}\n            for record in detector_records\n            if record.get("needs_review") or not record.get("safe_to_inpaint")\n        ]\n        detection_issues = []\n'''
    new_unverified = '''        decision_fields = (\n            "x1", "y1", "x2", "y2", "confidence", "source_model",\n            "source_role", "class_name", "semantic_type", "deferred_reason",\n        )\n        unverified_regions = [\n            {k: record.get(k) for k in decision_fields}\n            for record in detector_records\n            if record.get("needs_review") or not record.get("safe_to_inpaint")\n        ]\n        deferred_regions = [\n            {k: record.get(k) for k in decision_fields}\n            for record in detector_records\n            if record.get("deferred_reason")\n        ]\n        detection_issues = []\n'''
    text = replace_once(text, old_unverified, new_unverified, "decision region records")
    text = replace_once(
        text,
        '''        if unverified_regions:\n            detection_issues.append("unverified_regions")\n''',
        '''        if unverified_regions:\n            detection_issues.append("unverified_regions")\n        if deferred_regions:\n            detection_issues.append("deferred_regions")\n''',
        "deferred detection issue",
    )
    text = replace_once(
        text,
        '''                "review_only": len(unverified_regions),\n            },\n''',
        '''                "review_only": len(unverified_regions),\n                "deferred": len(deferred_regions),\n            },\n''',
        "deferred detector metric",
    )
    text = replace_once(
        text,
        '''            "unverified_regions": unverified_regions,\n            "needs_review": bool(detection_issues),\n''',
        '''            "unverified_regions": unverified_regions,\n            "deferred_regions": deferred_regions,\n            "needs_review": bool(detection_issues),\n''',
        "return deferred decisions",
    )
    path.write_text(text, encoding="utf-8")


def patch_release_cleanup() -> None:
    path = ROOT / "app/routers/render_commit.py"
    text = path.read_text(encoding="utf-8")
    text = replace_once(text, "import copy\nimport os\nimport uuid\n", "import copy\nimport uuid\n", "render unused os")
    path.write_text(text, encoding="utf-8")


def patch_sanity() -> None:
    path = ROOT / "scripts/backend_foundation_sanity.py"
    text = path.read_text(encoding="utf-8")
    old_tail = '''model_contract_checks()\ngeometry_contract_checks()\npublication_safety_checks()\nwindows_locked_reader_check()\nprint("backend foundation sanity: phases 1-2 PASS")\n'''
    new_tail = r'''def evidence_retention_checks():
    import types
    import numpy as np

    if "onnxruntime" not in sys.modules:
        sys.modules["onnxruntime"] = types.ModuleType("onnxruntime")

    from app.detector.bubble_detector import BubbleBox, YoloDetector
    from app.detector.combined_detector import CombinedTextDetector
    from app.detector.mask_builder import build_mask
    from app.inpaint.lama_inpainter import Inpainter

    giant = BubbleBox(
        0, 100, 980, 120, 0.9,
        np.full((20, 980), 255, np.uint8),
        source_model="segmenter.onnx",
        class_name="text_comic",
        semantic_type="text",
        mask_source="text_segmenter",
        safe_to_inpaint=True,
        ocr_eligible=True,
        source_role="text_segmenter",
    )
    retained = YoloDetector._filter_invalid([giant], 1000, 1000)
    check(len(retained) == 1, "giant SFX evidence disappeared")
    check(
        retained[0].deferred_reason
        and "box_width_limit" in retained[0].deferred_reason
        and retained[0].needs_review
        and not retained[0].safe_to_inpaint,
        "giant SFX was not explicitly deferred",
    )
    review_mask = build_mask((1000, 1000), retained)
    check(
        not np.any(review_mask > 0),
        "review-only region gained automatic destructive authority",
    )

    first = BubbleBox(
        200, 50, 800, 350, 0.9,
        np.full((300, 600), 255, np.uint8),
        source_model="segmenter.onnx", class_name="text_comic",
        semantic_type="text", mask_source="text_segmenter",
        safe_to_inpaint=True, ocr_eligible=True,
        source_role="text_segmenter",
    )
    second = BubbleBox(
        200, 350, 800, 650, 0.8,
        np.full((300, 600), 255, np.uint8),
        source_model="segmenter.onnx", class_name="text_comic",
        semantic_type="text", mask_source="text_segmenter",
        safe_to_inpaint=True, ocr_eligible=True,
        source_role="text_segmenter",
    )

    detector = object.__new__(CombinedTextDetector)
    grouped = detector._cluster_free_text_boxes([first, second], 1000, 1000)
    check(len(grouped) == 2, "oversized free-text group dropped evidence")
    check(
        all(box.mask is not None and np.any(box.mask > 0) for box in grouped),
        "oversized free-text split lost a child mask",
    )

    split = Inpainter._split_oversized_cluster_area(
        [first, second], 1000, 1000
    )
    flat = [box for group in split for box in group]
    check(len(split) == 2 and len(flat) == 2, "two-box inpaint split lost evidence")
    combined_mask = build_mask((1000, 1000), flat)
    check(
        combined_mask[100, 300] > 0 and combined_mask[500, 300] > 0,
        "two-box reproduction did not retain both masks",
    )

    pipeline_source = (ROOT / "app/pipeline.py").read_text(encoding="utf-8")
    check(
        'target_page["deferred_regions"]' in pipeline_source
        and '"deferred_reason": b.deferred_reason' in pipeline_source
        and 'and not record.get("deferred_reason")' in pipeline_source,
        "saved/deferred decision contract missing",
    )


evidence_retention_checks()
model_contract_checks()
geometry_contract_checks()
publication_safety_checks()
windows_locked_reader_check()
print("backend foundation sanity: phases 1-3 PASS")
'''
    text = replace_once(text, old_tail, new_tail, "phase 3 sanity tail")
    path.write_text(text, encoding="utf-8")


def main() -> None:
    patch_bubble_detector()
    patch_combined_detector()
    patch_inpainter()
    patch_pipeline()
    patch_release_cleanup()
    patch_sanity()
    print("phase 3 patch applied")


if __name__ == "__main__":
    main()
