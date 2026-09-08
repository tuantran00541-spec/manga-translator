from pathlib import Path


def replace_exact(path: Path, old: str, new: str) -> None:
    source = path.read_text(encoding="utf-8")
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected exactly one anchor, found {count}")
    path.write_text(source.replace(old, new), encoding="utf-8")


parameters = Path("app/parameters.py")
combined = Path("app/detector/combined_detector.py")
analyzer = Path("scripts/profile_stage_breakdown.py")
phase7 = Path("scripts/backend_phase7_sanity.py")

replace_exact(
    parameters,
    'DETECTOR_FREE_TEXT_GRAYSCALE_FALLBACK = _env_bool(\n    "MANGA_DETECTOR_FREE_TEXT_GRAYSCALE_FALLBACK", True\n)\n',
    'DETECTOR_FREE_TEXT_GRAYSCALE_FALLBACK = _env_bool(\n    "MANGA_DETECTOR_FREE_TEXT_GRAYSCALE_FALLBACK", True\n)\n'
    'DETECTOR_GRAYSCALE_FALLBACK_PAD_X = _env_int(\n'
    '    "MANGA_DETECTOR_GRAYSCALE_FALLBACK_PAD_X", 96, minimum=0, maximum=512\n'
    ')\n'
    'DETECTOR_GRAYSCALE_FALLBACK_PAD_Y = _env_int(\n'
    '    "MANGA_DETECTOR_GRAYSCALE_FALLBACK_PAD_Y", 96, minimum=0, maximum=512\n'
    ')\n'
    'DETECTOR_GRAYSCALE_FALLBACK_MAX_ROIS = _env_int(\n'
    '    "MANGA_DETECTOR_GRAYSCALE_FALLBACK_MAX_ROIS", 2, minimum=1, maximum=8\n'
    ')\n'
    'DETECTOR_GRAYSCALE_FALLBACK_MAX_SOURCE_SIDE = _env_int(\n'
    '    "MANGA_DETECTOR_GRAYSCALE_FALLBACK_MAX_SOURCE_SIDE",\n'
    '    1344,\n'
    '    minimum=256,\n'
    '    maximum=4096,\n'
    ')\n',
)

replace_exact(
    combined,
    '    DETECTOR_FREE_TEXT_GRAYSCALE_FALLBACK,\n    DETECTOR_NMS_SCORE_FLOOR,\n',
    '    DETECTOR_FREE_TEXT_GRAYSCALE_FALLBACK,\n'
    '    DETECTOR_GRAYSCALE_FALLBACK_MAX_ROIS,\n'
    '    DETECTOR_GRAYSCALE_FALLBACK_MAX_SOURCE_SIDE,\n'
    '    DETECTOR_GRAYSCALE_FALLBACK_PAD_X,\n'
    '    DETECTOR_GRAYSCALE_FALLBACK_PAD_Y,\n'
    '    DETECTOR_NMS_SCORE_FLOOR,\n',
)

methods_anchor = '    def detect(self, image: np.ndarray, *, parallel: bool = False) -> list[BubbleBox]:\n'
methods = '''    @staticmethod
    def _plan_grayscale_fallback_rois(
        image_shape: tuple[int, int],
        proposals: list[BubbleBox],
    ) -> tuple[list[tuple[int, int, int, int]], int]:
        """Bound grayscale retry to compact free-text regions.

        The old fallback re-ran the text model over the entire page, which is
        especially expensive on tall slices because the adaptive detector may
        execute several windows plus a full-image pass.  These ROIs are recall
        insurance only: proposals that do not fit the bounded retry remain
        review-only through the normal detector path.
        """
        h, w = (int(image_shape[0]), int(image_shape[1]))
        if h <= 0 or w <= 0 or not proposals:
            return [], 0

        max_side = max(1, int(DETECTOR_GRAYSCALE_FALLBACK_MAX_SOURCE_SIDE))
        max_rois = max(1, int(DETECTOR_GRAYSCALE_FALLBACK_MAX_ROIS))
        ranked = sorted(
            proposals,
            key=lambda box: (
                -float(box.confidence),
                max(1, int(box.x2 - box.x1) * int(box.y2 - box.y1)),
                int(box.y1),
                int(box.x1),
            ),
        )
        rois: list[tuple[int, int, int, int]] = []
        deferred = 0

        def bounded_axis(
            start: int,
            end: int,
            bound: int,
            pad: int,
        ) -> tuple[int, int] | None:
            start = max(0, min(int(start), bound))
            end = max(start, min(int(end), bound))
            if end <= start or end - start > max_side:
                return None
            padded_start = max(0, start - int(pad))
            padded_end = min(bound, end + int(pad))
            if padded_end - padded_start <= max_side:
                return padded_start, padded_end
            target = min(max_side, bound)
            center = (start + end) // 2
            window_start = max(0, min(bound - target, center - target // 2))
            if window_start > start:
                window_start = start
            if window_start + target < end:
                window_start = end - target
            window_start = max(0, min(bound - target, window_start))
            return window_start, window_start + target

        for proposal in ranked:
            xs = bounded_axis(
                proposal.x1,
                proposal.x2,
                w,
                DETECTOR_GRAYSCALE_FALLBACK_PAD_X,
            )
            ys = bounded_axis(
                proposal.y1,
                proposal.y2,
                h,
                DETECTOR_GRAYSCALE_FALLBACK_PAD_Y,
            )
            if xs is None or ys is None:
                deferred += 1
                continue
            candidate = (xs[0], ys[0], xs[1], ys[1])

            duplicate = False
            for index, current in enumerate(rois):
                ix1, iy1 = max(candidate[0], current[0]), max(candidate[1], current[1])
                ix2, iy2 = min(candidate[2], current[2]), min(candidate[3], current[3])
                if ix2 <= ix1 or iy2 <= iy1:
                    continue
                inter = (ix2 - ix1) * (iy2 - iy1)
                candidate_area = max(1, (candidate[2] - candidate[0]) * (candidate[3] - candidate[1]))
                current_area = max(1, (current[2] - current[0]) * (current[3] - current[1]))
                if inter / float(min(candidate_area, current_area)) < 0.70:
                    continue
                union = (
                    min(candidate[0], current[0]),
                    min(candidate[1], current[1]),
                    max(candidate[2], current[2]),
                    max(candidate[3], current[3]),
                )
                if (
                    union[2] - union[0] <= max_side
                    and union[3] - union[1] <= max_side
                ):
                    rois[index] = union
                    duplicate = True
                    break
            if duplicate:
                continue
            if len(rois) >= max_rois:
                deferred += 1
                continue
            rois.append(candidate)

        return rois, deferred

    def _grayscale_text_retry(
        self,
        image: np.ndarray,
        proposals: list[BubbleBox],
    ) -> tuple[list[BubbleBox], dict[str, int]]:
        """Run grayscale text segmentation only inside bounded proposal ROIs."""
        h, w = image.shape[:2]
        rois, deferred = self._plan_grayscale_fallback_rois((h, w), proposals)
        recovered: list[BubbleBox] = []
        source_pixels = 0
        for x1, y1, x2, y2 in rois:
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            source_pixels += int((x2 - x1) * (y2 - y1))
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
            gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            # The ROI is bounded below the tall-page retry regime. Calling the
            # detector's single-image path preserves configured TTA while
            # avoiding adaptive whole-page windows and their redundant full pass.
            boxes = self.text_detector._detect_single(gray_bgr, x1, y1)
            boxes = [self.text_detector._with_semantics(box) for box in boxes]
            boxes = self.text_detector._filter_invalid(boxes, w, h)
            recovered.extend(self._classify(box) for box in boxes)

        recovered = [
            text
            for text in recovered
            if any(self._is_inside(text, proposal) for proposal in proposals)
        ]
        if recovered:
            recovered = [
                self._classify(box)
                for box in YoloDetector._nms_box_group(
                    recovered,
                    score_threshold=TEXT_CONF_THRESHOLD,
                    iou_threshold=DETECTOR_FINAL_NMS_IOU,
                )
            ]
        return recovered, {
            "calls": len(rois),
            "source_pixels": source_pixels,
            "deferred_regions": int(deferred),
        }

'''
replace_exact(combined, methods_anchor, methods + methods_anchor)

replace_exact(
    combined,
    '            "text_grayscale_fallback_runs": 0,\n            "text_grayscale_fallback_ms": 0.0,\n            "text_grayscale_fallback_proposals": 0,\n',
    '            "text_grayscale_fallback_runs": 0,\n'
    '            "text_grayscale_fallback_calls": 0,\n'
    '            "text_grayscale_fallback_source_pixels": 0,\n'
    '            "text_grayscale_fallback_deferred_regions": 0,\n'
    '            "text_grayscale_fallback_ms": 0.0,\n'
    '            "text_grayscale_fallback_proposals": 0,\n',
)

old_fallback = '''        if DETECTOR_FREE_TEXT_GRAYSCALE_FALLBACK and unmatched_free_text:
            fallback_started_at = time.perf_counter()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            fallback_boxes = [
                self._classify(box)
                for box in self.text_detector.detect(gray_bgr)
            ]
            fallback_boxes = [
                text
                for text in fallback_boxes
                if any(
                    self._is_inside(text, bubble)
                    for bubble in unmatched_free_text
                )
            ]
            metrics["text_grayscale_fallback_runs"] = 1
            metrics["text_grayscale_fallback_ms"] = round(
                (time.perf_counter() - fallback_started_at) * 1000.0, 3
            )
            metrics["text_grayscale_fallback_proposals"] = len(fallback_boxes)
            if fallback_boxes:
                text_boxes = [
                    self._classify(box)
                    for box in YoloDetector._nms_box_group(
                        text_boxes + fallback_boxes,
                        score_threshold=TEXT_CONF_THRESHOLD,
                        iou_threshold=DETECTOR_FINAL_NMS_IOU,
                    )
                ]
'''
new_fallback = '''        if DETECTOR_FREE_TEXT_GRAYSCALE_FALLBACK and unmatched_free_text:
            fallback_started_at = time.perf_counter()
            fallback_boxes, fallback_metrics = self._grayscale_text_retry(
                image,
                unmatched_free_text,
            )
            metrics["text_grayscale_fallback_runs"] = int(
                bool(fallback_metrics["calls"])
            )
            metrics["text_grayscale_fallback_calls"] = int(
                fallback_metrics["calls"]
            )
            metrics["text_grayscale_fallback_source_pixels"] = int(
                fallback_metrics["source_pixels"]
            )
            metrics["text_grayscale_fallback_deferred_regions"] = int(
                fallback_metrics["deferred_regions"]
            )
            metrics["text_grayscale_fallback_ms"] = round(
                (time.perf_counter() - fallback_started_at) * 1000.0, 3
            )
            metrics["text_grayscale_fallback_proposals"] = len(fallback_boxes)
            if fallback_boxes:
                text_boxes = [
                    self._classify(box)
                    for box in YoloDetector._nms_box_group(
                        text_boxes + fallback_boxes,
                        score_threshold=TEXT_CONF_THRESHOLD,
                        iou_threshold=DETECTOR_FINAL_NMS_IOU,
                    )
                ]
'''
replace_exact(combined, old_fallback, new_fallback)

replace_exact(
    analyzer,
    '    detector = detector_breakdown(metrics)\n    auto = _section(metrics, "auto_inpaint")\n\n    read_ms = _first_number(metrics.get("read_ms"), metrics.get("read_decode_ms"))\n',
    '    detector = detector_breakdown(metrics)\n'
    '    auto = _section(metrics, "auto_inpaint")\n'
    '    timing = _section(metrics, "timing_ms")\n\n'
    '    read_ms = _first_number(\n'
    '        metrics.get("read_ms"), metrics.get("read_decode_ms"), timing.get("read")\n'
    '    )\n',
)
replace_exact(
    analyzer,
    '        metrics.get("auto_inpaint_ms"),\n        auto.get("total_ms"),\n        metrics.get("inpaint_ms"),\n',
    '        metrics.get("auto_inpaint_ms"),\n'
    '        auto.get("total_ms"),\n'
    '        metrics.get("inpaint_ms"),\n'
    '        timing.get("auto_inpaint"),\n',
)
replace_exact(
    analyzer,
    '    write_ms = _first_number(metrics.get("write_ms"), metrics.get("write_encode_ms"))\n    page_total = _first_number(metrics.get("total_ms"), metrics.get("wall_ms"))\n',
    '    write_ms = _first_number(\n'
    '        metrics.get("write_ms"), metrics.get("write_encode_ms"), timing.get("write")\n'
    '    )\n'
    '    page_total = _first_number(\n'
    '        metrics.get("total_ms"), metrics.get("wall_ms"), timing.get("total")\n'
    '    )\n',
)

replace_exact(
    phase7,
    '    # Inclusive timer sums must remain a separately labelled diagnostic view.\n',
    '    nested = page_stage_breakdown(\n'
    '        {\n'
    '            "index": 5,\n'
    '            "metrics": {\n'
    '                "timing_ms": {\n'
    '                    "read": 4.0, "detect": 100.0, "auto_inpaint": 30.0,\n'
    '                    "write": 6.0, "total": 145.0,\n'
    '                },\n'
    '                "detector": {"total_ms": 100.0},\n'
    '                "auto_inpaint": {"lama_model_ms": 20.0},\n'
    '            },\n'
    '        }\n'
    '    )\n'
    '    check(nested["read_decode_ms"] == 4.0, "nested timing read was ignored")\n'
    '    check(nested["auto_inpaint"]["total_ms"] == 30.0, "nested inpaint timing was ignored")\n'
    '    check(nested["write_encode_ms"] == 6.0, "nested write timing was ignored")\n'
    '    check(nested["page_total_ms"] == 145.0, "nested page total was ignored")\n'
    '    check(nested["orchestration_other_ms"] == 5.0, "nested residual attribution is wrong")\n\n'
    '    # Inclusive timer sums must remain a separately labelled diagnostic view.\n',
)

print("phase11 ROI grayscale retry + profiler schema patch applied")
