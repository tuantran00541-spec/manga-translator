from pathlib import Path


def rep(path, old, new, count=1):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    found = text.count(old)
    if found != count:
        raise RuntimeError(f"{path}: expected {count} x {old[:90]!r}, found {found}")
    p.write_text(text.replace(old, new, count), encoding="utf-8")


def splice(path, start_marker, end_marker, replacement):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    start = text.find(start_marker)
    if start < 0:
        raise RuntimeError(f"{path}: missing start marker {start_marker!r}")
    end = text.find(end_marker, start)
    if end < 0:
        raise RuntimeError(f"{path}: missing end marker {end_marker!r}")
    p.write_text(text[:start] + replacement + text[end:], encoding="utf-8")
# Repaint/edit: preserve beats every destructive mask; skipped/stale pages are not editable.
rep("app/pipeline_editing.py", 'from app.parameters import MANUAL_MASK_THRESHOLD\n', 'from app.parameters import MANUAL_MASK_THRESHOLD\nfrom app.region_policy import subtract_regions_from_mask\n')
# repaint_mask snapshot
rep("app/pipeline_editing.py", '''                page = manifest["pages"][page_index]
                img_path = Path(page["original"])
                boxes_snapshot = copy.deepcopy(page.get("boxes", []))
                target_clean_revision = int(page.get("clean_revision") or 0) + 1
                manual_mask_posix = page.get("manual_mask")
''', '''                page = manifest["pages"][page_index]
                if page.get("skipped"):
                    raise ValueError("Cannot repaint a skipped page; unskip it first")
                if page.get("process_required"):
                    raise ValueError("Cannot repaint a page that requires processing")
                img_path = Path(page["original"])
                boxes_snapshot = copy.deepcopy(page.get("boxes", []))
                preserve_regions = copy.deepcopy(page.get("preserve_regions", []))
                target_clean_revision = int(page.get("clean_revision") or 0) + 1
                manual_mask_posix = page.get("manual_mask")
''')
rep("app/pipeline_editing.py", '            if existing_mask is not None and bin_mask is not None:\n                accumulated_mask = np.maximum(existing_mask, bin_mask)\n', '''            existing_mask = subtract_regions_from_mask(existing_mask, preserve_regions)
            bin_mask = subtract_regions_from_mask(bin_mask, preserve_regions)
            if bin_mask is not None and not np.any(bin_mask):
                raise ValueError("Repaint mask falls entirely inside a preserve region")
            if existing_mask is not None and bin_mask is not None:
                accumulated_mask = np.maximum(existing_mask, bin_mask)
''')
rep("app/pipeline_editing.py", '                        reuse_auto_clean=True,\n                    )\n', '                        reuse_auto_clean=True,\n                        preserve_regions=preserve_regions,\n                    )\n')
# add/update/remove snapshots and reinpaint calls
rep("app/pipeline_editing.py", '                target_page = manifest["pages"][page_index]\n                boxes_snapshot = copy.deepcopy(target_page.get("boxes", []))\n                boxes_snapshot.append(copy.deepcopy(new_box))\n', '                target_page = manifest["pages"][page_index]\n                preserve_regions = copy.deepcopy(target_page.get("preserve_regions", []))\n                boxes_snapshot = copy.deepcopy(target_page.get("boxes", []))\n                boxes_snapshot.append(copy.deepcopy(new_box))\n')
rep("app/pipeline_editing.py", '                boxes_snapshot = copy.deepcopy(target_boxes)\n                self._apply_box_geometry(boxes_snapshot[box_index], new_geometry)\n                manual_mask_posix = target_page.get("manual_mask")\n', '                boxes_snapshot = copy.deepcopy(target_boxes)\n                self._apply_box_geometry(boxes_snapshot[box_index], new_geometry)\n                preserve_regions = copy.deepcopy(target_page.get("preserve_regions", []))\n                manual_mask_posix = target_page.get("manual_mask")\n')
rep("app/pipeline_editing.py", '                img_path = Path(page["original"])\n                manual_mask_posix = page.get("manual_mask")\n', '                img_path = Path(page["original"])\n                preserve_regions = copy.deepcopy(page.get("preserve_regions", []))\n                manual_mask_posix = page.get("manual_mask")\n')
rep("app/pipeline_editing.py", '                    manual_mask_posix=manual_mask_posix,\n                    manual_lama_mask_posix=manual_lama_mask_posix,\n                )\n', '                    manual_mask_posix=manual_mask_posix,\n                    manual_lama_mask_posix=manual_lama_mask_posix,\n                    preserve_regions=preserve_regions,\n                )\n', 3)
# reset snapshot
rep("app/pipeline_editing.py", '''                page = manifest["pages"][page_index]
                img_path = Path(page["original"])
                boxes_snapshot = copy.deepcopy(page.get("boxes", []))
                target_clean_revision = int(page.get("clean_revision") or 0) + 1

            manual_mask_path = self._manual_mask_path(processed_dir, img_path)
''', '''                page = manifest["pages"][page_index]
                if page.get("skipped"):
                    raise ValueError("Cannot reset repaint state on a skipped page; unskip it first")
                if page.get("process_required"):
                    raise ValueError("Cannot reset repaint state before processing")
                img_path = Path(page["original"])
                boxes_snapshot = copy.deepcopy(page.get("boxes", []))
                preserve_regions = copy.deepcopy(page.get("preserve_regions", []))
                target_clean_revision = int(page.get("clean_revision") or 0) + 1

            manual_mask_path = self._manual_mask_path(processed_dir, img_path)
''')
rep("app/pipeline_editing.py", '                    reuse_auto_clean=True,\n                    apply_manual_mask=False,\n                )\n', '                    reuse_auto_clean=True,\n                    apply_manual_mask=False,\n                    preserve_regions=preserve_regions,\n                )\n')
# _do_reinpaint protection
rep("app/pipeline_editing.py", '        reuse_auto_clean: bool = False,\n        apply_manual_mask: bool = True,\n    ) -> str:\n', '        reuse_auto_clean: bool = False,\n        apply_manual_mask: bool = True,\n        preserve_regions: list[dict] | None = None,\n    ) -> str:\n')
rep("app/pipeline_editing.py", '            clean_image = self.inpainter.inpaint(image, boxes_objects)\n', '            clean_image = self.inpainter.inpaint(image, boxes_objects, protected_regions=preserve_regions)\n')
rep("app/pipeline_editing.py", '''                if manual_mask is not None:
                    clean_image = self.inpainter.inpaint_mask(
                        clean_image, manual_mask, force_lama=force_lama
                    )
''', '''                manual_mask = subtract_regions_from_mask(manual_mask, preserve_regions)
                if manual_mask is not None and np.any(manual_mask):
                    clean_image = self.inpainter.inpaint_mask(
                        clean_image, manual_mask, force_lama=force_lama
                    )
''')

# Schemas + AI-friendly rectangle repaint endpoint.
splice("app/schemas.py", "class SaveExcludedRegionsRequest(BaseModel):", "class ResetManualMaskRequest(BaseModel):", '''class SavePreserveRegionsRequest(BaseModel):
    chapter_id: str
    page_index: int = Field(ge=0)
    preserve_regions: list[RegionModel]

    @field_validator("preserve_regions")
    @classmethod
    def _region_count(cls, value: list[RegionModel]) -> list[RegionModel]:
        if len(value) > MAX_RENDER_TRANSLATIONS:
            raise ValueError("Too many preserve regions")
        return value


class SaveExcludedRegionsRequest(BaseModel):
    """Deprecated wire alias for pre-v4 clients."""
    chapter_id: str
    page_index: int = Field(ge=0)
    excluded_regions: list[RegionModel]


class RepaintRegionsRequest(BaseModel):
    chapter_id: str
    page_index: int = Field(ge=0)
    regions: list[RegionModel]
    mode: Literal["standard", "lama"] = "standard"

    @field_validator("regions")
    @classmethod
    def _regions(cls, value: list[RegionModel]) -> list[RegionModel]:
        if not value:
            raise ValueError("At least one repaint region is required")
        if len(value) > MAX_RENDER_TRANSLATIONS:
            raise ValueError("Too many repaint regions")
        return value


''')
rep("app/routers/editor.py", '    RemoveBoxRequest,\n    ResetManualMaskRequest,\n', '    RemoveBoxRequest,\n    RepaintRegionsRequest,\n    ResetManualMaskRequest,\n')
rep("app/routers/editor.py", 'def _reconcile_translation_after_ocr_edit(req: UpdateTextObjectRequest) -> dict:\n', '''def _regions_to_repaint_mask(image_shape: tuple[int, ...], regions) -> np.ndarray:
    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    for region in regions:
        item = region.model_dump() if hasattr(region, "model_dump") else dict(region)
        x1, y1, x2, y2 = (int(item[k]) for k in ("x1", "y1", "x2", "y2"))
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height or x2 <= x1 or y2 <= y1:
            raise ValueError(f"Repaint region ({x1},{y1})-({x2},{y2}) exceeds page dimensions ({width}x{height})")
        mask[y1:y2, x1:x2] = 255
    if not np.any(mask):
        raise ValueError("Repaint regions produced an empty mask")
    return mask


def _reconcile_translation_after_ocr_edit(req: UpdateTextObjectRequest) -> dict:
''')
rep("app/routers/editor.py", '@router.post("/repaint_mask")\nasync def repaint_mask(\n', '''@router.post("/repaint_regions")
async def repaint_regions(req: RepaintRegionsRequest) -> dict:
    validate_chapter_id(req.chapter_id)
    manifest = load_manifest_raw(req.chapter_id)
    pages = manifest.get("pages", [])
    if req.page_index < 0 or req.page_index >= len(pages):
        raise HTTPException(400, f"Invalid page_index: {req.page_index}")
    page = pages[req.page_index]
    if page.get("skipped") or page.get("process_required"):
        raise HTTPException(409, "Page must be active and processed before repaint")
    img_path = Path(page["original"])
    if not img_path.is_file():
        raise HTTPException(404, f"Original page image not found: page_{req.page_index:03d}")
    try:
        image = await run_in_threadpool(read_image, img_path)
        mask_array = _regions_to_repaint_mask(image.shape, req.regions)
        result = await run_in_threadpool(
            pipeline.repaint_mask, req.chapter_id, req.page_index, mask_array,
            force_lama=req.mode == "lama",
        )
        return urlify_manifest(result)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/repaint_mask")
async def repaint_mask(
''')
rep("app/routers/editor.py", '    if page_index < 0 or page_index >= len(pages):\n        raise HTTPException(400, f"Invalid page_index: {page_index}")\n\n    img_path = Path(pages[page_index]["original"])\n', '    if page_index < 0 or page_index >= len(pages):\n        raise HTTPException(400, f"Invalid page_index: {page_index}")\n    if pages[page_index].get("skipped") or pages[page_index].get("process_required"):\n        raise HTTPException(409, "Page must be active and processed before repaint")\n\n    img_path = Path(pages[page_index]["original"])\n')
