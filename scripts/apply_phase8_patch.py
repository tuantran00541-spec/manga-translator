from pathlib import Path

path = Path("app/detector/recovery.py")
source = path.read_text(encoding="utf-8")
old_init = '''        self._mser_lock = threading.Lock()
        self._primitive_local = threading.local()
'''
new_init = '''        self._mser_lock = threading.Lock()
        self._primitive_local = threading.local()
        self._candidate_local = threading.local()
'''
if source.count(old_init) != 1:
    raise RuntimeError(f"phase8 recovery init anchor expected once, got {source.count(old_init)}")
source = source.replace(old_init, new_init, 1)

start_marker = "    def detect(self, image: np.ndarray, existing: list[BubbleBox] | None = None) -> list[BubbleBox]:\n"
start = source.find(start_marker)
if start < 0:
    raise RuntimeError("phase8 recovery detect anchor missing")

new_tail = r'''    def _build_base_candidates(
        self,
        image: np.ndarray,
        gray: np.ndarray,
        boxes: np.ndarray,
    ) -> tuple[tuple[BubbleBox, float], ...]:
        """Build expensive MSER cluster/mask evidence once per exact image object.

        This cache deliberately stops before any check involving ``existing``.
        First-pass bubble proposals and final merged detector evidence therefore
        retain independent filtering semantics even when they share the same raw
        MSER primitives, clusters and reconstructed review masks.
        """
        state = getattr(self._candidate_local, "value", None)
        if state is not None:
            image_ref = state.get("image_ref")
            if (
                image_ref is not None
                and image_ref() is image
                and state.get("shape") == tuple(image.shape[:2])
            ):
                return state["candidates"]

        h, w = image.shape[:2]
        rects: list[tuple[int, int, int, int]] = []
        for x, y, bw, bh in np.asarray(boxes).reshape(-1, 4):
            x, y, bw, bh = map(int, (x, y, bw, bh))
            if (
                bw < MSER_REGION_MIN_SIDE
                or bh < MSER_REGION_MIN_SIDE
                or bw > w * MSER_REGION_MAX_WIDTH_RATIO
                or bh > h * MSER_REGION_MAX_HEIGHT_RATIO
            ):
                continue
            area = bw * bh
            if (
                area < MSER_REGION_AREA_MIN
                or area > w * h * MSER_REGION_AREA_RATIO_MAX
            ):
                continue
            rects.append((x, y, x + bw, y + bh))

        remaining = rects[:]
        clusters: list[list[tuple[int, int, int, int]]] = []
        while remaining:
            cluster = [remaining.pop()]
            changed = True
            while changed:
                changed = False
                keep = []
                cx1 = min(r[0] for r in cluster)
                cy1 = min(r[1] for r in cluster)
                cx2 = max(r[2] for r in cluster)
                cy2 = max(r[3] for r in cluster)
                ch = max(8, cy2 - cy1)
                for r in remaining:
                    rx1, ry1, rx2, ry2 = r
                    near_x = not (
                        rx1 > cx2 + ch * MSER_CLUSTER_NEAR_X_FACTOR
                        or rx2 < cx1 - ch * MSER_CLUSTER_NEAR_X_FACTOR
                    )
                    near_y = not (
                        ry1 > cy2 + ch * MSER_CLUSTER_NEAR_Y_FACTOR
                        or ry2 < cy1 - ch * MSER_CLUSTER_NEAR_Y_FACTOR
                    )
                    if near_x and near_y:
                        cluster.append(r)
                        changed = True
                    else:
                        keep.append(r)
                remaining = keep
            clusters.append(cluster)

        base: list[tuple[BubbleBox, float]] = []
        for cluster in clusters:
            if len(cluster) < MSER_CLUSTER_MIN_REGIONS:
                continue
            x1 = max(0, min(r[0] for r in cluster) - MSER_RECOVERY_PAD)
            y1 = max(0, min(r[1] for r in cluster) - MSER_RECOVERY_PAD)
            x2 = min(w, max(r[2] for r in cluster) + MSER_RECOVERY_PAD)
            y2 = min(h, max(r[3] for r in cluster) + MSER_RECOVERY_PAD)
            bw, bh = x2 - x1, y2 - y1
            if bw < MSER_CLUSTER_MIN_WIDTH or bh < MSER_CLUSTER_MIN_HEIGHT:
                continue
            candidate = BubbleBox(
                x1,
                y1,
                x2,
                y2,
                MSER_REVIEW_CONFIDENCE,
                None,
                source_model="opencv_mser",
                class_id=0,
                class_name="text_recovery",
                semantic_type="free_text",
                mask_source="none",
                safe_to_inpaint=False,
                ocr_eligible=True,
                needs_review=True,
            )
            crop = gray[y1:y2, x1:x2]
            mask = self._seed_mask(crop)
            ratio = float(np.count_nonzero(mask)) / float(max(1, mask.size))
            page_ratio = (bw * bh) / float(max(1, w * h))
            review_mask_valid = bool(
                MSER_SAFE_MASK_RATIO_MIN <= ratio <= MSER_SAFE_MASK_RATIO_MAX
                and not self._mask_component_spans_crop(mask)
                and page_ratio <= MSER_SAFE_PAGE_AREA_RATIO_MAX
                and len(cluster) >= MSER_SAFE_CLUSTER_MIN_REGIONS
            )
            if review_mask_valid:
                candidate = replace(
                    candidate,
                    mask=mask,
                    mask_source="opencv_mser_review",
                    safe_to_inpaint=False,
                    ocr_eligible=True,
                    needs_review=True,
                    confidence=MSER_SAFE_CONFIDENCE,
                )
            base.append((candidate, page_ratio))

        candidates = tuple(base)
        self._candidate_local.value = {
            "image_ref": weakref.ref(image),
            "shape": tuple(image.shape[:2]),
            "candidates": candidates,
        }
        return candidates

    @staticmethod
    def _clone_base_candidate(candidate: BubbleBox) -> BubbleBox:
        mask = candidate.mask
        return replace(
            candidate,
            mask=mask.copy() if mask is not None else None,
        )

    def detect(self, image: np.ndarray, existing: list[BubbleBox] | None = None) -> list[BubbleBox]:
        if image is None or image.size == 0:
            return []
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        boxes = self._extract_primitives(image, gray)
        if boxes.size == 0:
            return []

        base_candidates = self._build_base_candidates(image, gray, boxes)
        if not base_candidates:
            return []

        out: list[BubbleBox] = []
        existing = existing or []
        has_safe_existing = any(b.safe_to_inpaint for b in existing)
        for base_candidate, page_ratio in base_candidates:
            candidate = self._clone_base_candidate(base_candidate)
            if any(self._iou(candidate, b) > MSER_EXISTING_IOU_SKIP for b in existing):
                continue
            contained_verified = 0
            for b in existing:
                if not b.safe_to_inpaint:
                    continue
                cx = (b.x1 + b.x2) / 2.0
                cy = (b.y1 + b.y2) / 2.0
                if candidate.x1 <= cx <= candidate.x2 and candidate.y1 <= cy <= candidate.y2:
                    contained_verified += 1
            if contained_verified >= MSER_CONTAINED_SAFE_SKIP_COUNT:
                continue
            if page_ratio > MSER_PAGE_CLUSTER_SKIP_RATIO and has_safe_existing:
                continue
            out.append(candidate)

        # Review-only MSER proposals must not suppress additional review
        # recovery. Residual lines are permitted when independent existing
        # evidence is already verified; they still never gain erase authority.
        verification_set = existing + out
        if existing and all(
            box.safe_to_inpaint and not box.needs_review for box in existing
        ):
            residual = self._residual_line_candidates(
                np.asarray(boxes), (h, w), verification_set
            )
            for candidate in residual:
                if any(
                    self._iou(candidate, box) > MSER_RESIDUAL_FINAL_IOU_SKIP
                    for box in out
                ):
                    continue
                out.append(candidate)
        return out
'''
source = source[:start] + new_tail
path.write_text(source, encoding="utf-8")
print("phase8 recovery base-candidate cache patch applied")
