from __future__ import annotations

from contextlib import nullcontext
import tempfile
from pathlib import Path

import pytest
from fastapi import HTTPException

from app.visual_qc.cache import load_region_qc_cache, store_region_qc_cache
from app.visual_qc.service import ChapterQCService


def test_region_pair_cache_is_bound_to_original_file_revision():
    page = {"source_revision": 2, "clean_revision": 7}
    result = {"region_id": "P0001-R01", "status": "pass", "issues": []}
    clean_rev = (100, 200, 300)
    original_rev = (400, 500, 600)

    store_region_qc_cache(
        page,
        region_id="P0001-R01",
        model="gemini-3.7-flash",
        mode="region-pair",
        clean_file_revision=clean_rev,
        source_file_revision=original_rev,
        result=result,
    )

    assert load_region_qc_cache(
        page,
        region_id="P0001-R01",
        model="gemini-3.7-flash",
        mode="region-pair",
        clean_file_revision=clean_rev,
        source_file_revision=original_rev,
    ) == result
    assert load_region_qc_cache(
        page,
        region_id="P0001-R01",
        model="gemini-3.7-flash",
        mode="region-pair",
        clean_file_revision=clean_rev,
        source_file_revision=(401, 500, 600),
    ) is None


def test_chapter_qc_rejects_manual_mask_outside_processed_chapter_root():
    service = ChapterQCService(
        runner=object(),
        manifest_loader=lambda _chapter_id: {},
        manifest_saver=lambda _chapter_id, _manifest: None,
        manifest_lock=lambda _chapter_id: nullcontext(),
        api_key_provider=lambda: "test-key",
    )

    with tempfile.TemporaryDirectory() as tmp:
        outside = Path(tmp) / "outside-mask.png"
        outside.write_bytes(b"not-even-read")
        manifest = {"pages": [{"manual_mask": str(outside)}]}
        with pytest.raises(HTTPException) as cm:
            service._load_manual_masks(manifest, "deadbeef")

    assert cm.value.status_code == 403
