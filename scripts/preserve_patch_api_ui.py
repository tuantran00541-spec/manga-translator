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
# Preserve APIs. Changing policy invalidates clean state and requires one reprocess.
rep("app/routers/chapters.py", 'from app.manifest_utils import get_manifest_lock, invalidate_page_render, load_manifest_raw, save_manifest_raw, urlify_manifest\n', 'from app.manifest_utils import bump_page_revision, get_manifest_lock, get_page_lock, invalidate_page_render, load_manifest_raw, save_manifest_raw, urlify_manifest\n')
rep("app/routers/chapters.py", 'from app.upload_utils import read_upload_limited\n', 'from app.upload_utils import read_upload_limited\nfrom app.text_objects import ensure_page_text_objects\n')
rep("app/routers/chapters.py", '    RegionModel,\n    SaveExcludedRegionsRequest,\n    SkipPagesRequest,\n', '    RegionModel,\n    SaveExcludedRegionsRequest,\n    SavePreserveRegionsRequest,\n    SkipPagesRequest,\n')
rep("app/routers/chapters.py", '        raise HTTPException(400, "Too many excluded regions")\n', '        raise HTTPException(400, "Too many preserve regions")\n')
chapters = Path("app/routers/chapters.py")
text = chapters.read_text(encoding="utf-8")
start = text.index('@router.post("/save_excluded_regions")')
replacement = '''def _set_page_preserve_regions(chapter_id: str, page_index: int, regions: list[RegionModel]) -> dict:
    validate_chapter_id(chapter_id)
    if page_index < 0:
        raise HTTPException(400, f"Invalid page_index: {page_index}")
    payload = _region_payload(regions)
    with get_page_lock(chapter_id, page_index), get_manifest_lock(chapter_id):
        manifest = load_manifest_raw(chapter_id)
        pages = manifest.get("pages", [])
        if page_index >= len(pages):
            raise HTTPException(400, f"Invalid page_index: {page_index}")
        page = pages[page_index]
        changed = page.get("preserve_regions", []) != payload
        page["preserve_regions"] = payload
        if changed:
            ensure_page_text_objects(page)
            if not page.get("skipped"):
                page["process_required"] = True
                if page.get("clean") is not None:
                    page["clean"] = None
                    bump_page_revision(page, "clean_revision")
            invalidate_page_render(manifest, page_index)
            save_manifest_raw(chapter_id, manifest)
            pipeline._sync_output_dir(chapter_id, manifest, [page_index])
    return urlify_manifest(manifest)


@router.post("/save_preserve_regions")
def save_preserve_regions(req: SavePreserveRegionsRequest) -> dict:
    return _set_page_preserve_regions(req.chapter_id, req.page_index, req.preserve_regions)


@router.post("/chapters/{chapter_id}/pages/{page_index}/preserve-regions")
def set_page_preserve_regions(chapter_id: str, page_index: int, regions: list[RegionModel]) -> dict:
    return _set_page_preserve_regions(chapter_id, page_index, regions)


@router.post("/save_excluded_regions", deprecated=True)
def save_excluded_regions(req: SaveExcludedRegionsRequest) -> dict:
    return _set_page_preserve_regions(req.chapter_id, req.page_index, req.excluded_regions)


@router.post("/chapters/{chapter_id}/pages/{page_index}/excluded-regions", deprecated=True)
def set_page_excluded_regions(chapter_id: str, page_index: int, regions: list[RegionModel]) -> dict:
    return _set_page_preserve_regions(chapter_id, page_index, regions)
'''
chapters.write_text(text[:start] + replacement, encoding="utf-8")

# UI/API naming. Keep old CSS class names as implementation details to avoid style churn.
api_path = Path("app/static/js/api.js")
api = api_path.read_text(encoding="utf-8")
for old, new in (
    ("flushExcludedRegionSaves", "flushPreserveRegionSaves"),
    ("_excludedRegionSaveStates", "_preserveRegionSaveStates"),
    ("_cloneExcludedRegions", "_clonePreserveRegions"),
    ("_excludedRegionSaveKey", "_preserveRegionSaveKey"),
    ("_drainExcludedRegionSaves", "_drainPreserveRegionSaves"),
    ("_ensureExcludedRegionSave", "_ensurePreserveRegionSave"),
    ("saveExcludedRegions", "savePreserveRegions"),
    ("excludedRegions", "preserveRegions"),
    ("excluded_regions", "preserve_regions"),
    ("/api/save_excluded_regions", "/api/save_preserve_regions"),
    ("vùng loại trừ", "vùng giữ nguyên"),
    ("Vùng loại trừ", "Vùng giữ nguyên"),
):
    api = api.replace(old, new)
api = api.replace('''      const serverRegions = data?.pages?.[state.pageIndex]?.preserve_regions;
      currentManifest.pages[state.pageIndex].preserve_regions =
        _clonePreserveRegions(Array.isArray(serverRegions) ? serverRegions : regions);
''', '''      const serverPage = data?.pages?.[state.pageIndex];
      if (serverPage && typeof serverPage === "object") {
        Object.assign(currentManifest.pages[state.pageIndex], serverPage);
      }
      const serverRegions = serverPage?.preserve_regions;
      currentManifest.pages[state.pageIndex].preserve_regions =
        _clonePreserveRegions(Array.isArray(serverRegions) ? serverRegions : regions);
''')
api_path.write_text(api, encoding="utf-8")

preview_path = Path("app/static/js/preview.js")
preview = preview_path.read_text(encoding="utf-8")
for old, new in (
    ("excluded_regions", "preserve_regions"),
    ("saveExcludedRegions", "savePreserveRegions"),
    ("vùng loại trừ", "vùng giữ nguyên"),
    ("Vùng loại trừ", "Vùng giữ nguyên"),
    ("Xóa vùng cấm này", "Xóa vùng giữ nguyên này"),
):
    preview = preview.replace(old, new)
preview = preview.replace('''    state: item.skipped ? "skipped" : (item.preserve_regions?.length ? "review" : "ready"),
    stateLabel: item.skipped ? "Bỏ qua" : (item.preserve_regions?.length ? `${item.preserve_regions.length} vùng giữ nguyên` : "Sẵn sàng"),
''', '''    state: item.skipped ? "skipped" : (item.process_required || item.preserve_regions?.length ? "review" : "ready"),
    stateLabel: item.skipped ? "Bỏ qua" : (item.process_required ? "Cần xử lý" : (item.preserve_regions?.length ? `${item.preserve_regions.length} vùng giữ nguyên` : "Sẵn sàng")),
''')
preview = preview.replace('  status.textContent = page.skipped ? "Đã bỏ qua" : "Sẵn sàng xử lý";', '  status.textContent = page.skipped ? "Đã bỏ qua" : (page.process_required ? "Cần xử lý" : "Đã xử lý");')
preview_path.write_text(preview, encoding="utf-8")

# Export defense-in-depth.
rep("app/routers/export.py", 'from app.text_objects import ensure_page_text_objects\n', 'from app.text_objects import ensure_page_text_objects\nfrom app.region_policy import text_object_in_preserve_region\n')
rep("app/routers/export.py", '            or obj.get("source_missing")\n        ):\n', '            or obj.get("source_missing")\n            or text_object_in_preserve_region(page, obj)\n        ):\n')
