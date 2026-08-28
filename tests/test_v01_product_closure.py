import asyncio
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import app.config as config
import app.manifest_utils as manifest_utils
import app.routers.export as export_router
import app.routers.editorial as editorial_router
import app.routers.image as image_router
import app.routers.render_commit as render_commit
import app.routers.translation as translation_router
from app.translation.deepseek import TranslationResult


CHAPTER_ID = "c0ffee12"


def test_processed_chapter_closes_translate_render_export_loop():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        raw = root / "raw"
        processed = root / "processed"
        output = root / "output"
        raw_page_dir = raw / CHAPTER_ID
        processed_page_dir = processed / CHAPTER_ID
        output_page_dir = output / CHAPTER_ID
        for path in (raw_page_dir, processed_page_dir, output_page_dir):
            path.mkdir(parents=True, exist_ok=True)

        original = raw_page_dir / "000.png"
        clean = processed_page_dir / "clean_000.png"
        image = np.full((120, 180, 3), 245, dtype=np.uint8)
        cv2.rectangle(image, (20, 30), (160, 90), (255, 255, 255), -1)
        assert cv2.imwrite(str(original), image)
        assert cv2.imwrite(str(clean), image)

        patchers = [
            patch.object(config, "RAW_DIR", raw),
            patch.object(config, "PROCESSED_DIR", processed),
            patch.object(config, "OUTPUT_DIR", output),
            patch.object(manifest_utils, "PROCESSED_DIR", processed),
            patch.object(render_commit, "OUTPUT_DIR", output),
            patch.object(export_router, "OUTPUT_DIR", output),
            patch.object(image_router, "RAW_DIR", raw),
            patch.object(image_router, "PROCESSED_DIR", processed),
            patch.object(image_router, "OUTPUT_DIR", output),
        ]
        for patcher in patchers:
            patcher.start()
        try:
            manifest_utils.save_manifest_raw(
                CHAPTER_ID,
                {
                    "chapter_id": CHAPTER_ID,
                    "source_url": None,
                    "pages": [
                        {
                            "original": original.as_posix(),
                            "clean": clean.as_posix(),
                            "boxes": [
                                {
                                    "id": "box_deadbeef",
                                    "origin": "detector",
                                    "x1": 30,
                                    "y1": 35,
                                    "x2": 150,
                                    "y2": 85,
                                    "confidence": 0.95,
                                    "ocr_text": "Hello",
                                    "ocr_source": "manual",
                                }
                            ],
                            "text_objects": [],
                            "skipped": False,
                            "excluded_regions": [],
                            "source_page": 0,
                            "slice_index": 0,
                            "source_revision": 1,
                            "process_revision": 1,
                            "clean_revision": 1,
                            "render_revision": 0,
                            "rendered": False,
                        }
                    ],
                    "workflow": {"stage": "editor", "page_index": 0},
                },
            )

            fake_translation = TranslationResult(
                translations={"text_deadbeef": "Xin chào"},
                usage={"prompt_tokens": 20, "completion_tokens": 5},
                estimated_cost_usd=0.00001,
                model="deepseek-v4-flash",
            )
            request = translation_router.TranslateChapterRequest(
                chapter_id=CHAPTER_ID,
                source_lang="en",
                target_lang="vi",
                budget_usd=0.02,
            )
            with (
                patch.object(translation_router, "get_deepseek_api_key", return_value="test-key"),
                patch.object(translation_router.translator, "translate", return_value=fake_translation),
            ):
                translated = asyncio.run(translation_router.translate_chapter(request))

            assert translated["translation_run"]["translated"] == 1
            page = manifest_utils.load_manifest_raw(CHAPTER_ID)["pages"][0]
            assert len(page["text_objects"]) == 1
            assert page["text_objects"][0]["source_boxes"] == ["box_deadbeef"]
            assert page["text_objects"][0]["translation"] == "Xin chào"

            # P0 editorial flow: machine translation must be proofed before typeset/export.
            review = editorial_router.set_script_review(
                editorial_router.ScriptReviewRequest(
                    chapter_id=CHAPTER_ID,
                    page_index=0,
                    object_id="text_deadbeef",
                    status="reviewed",
                )
            )
            assert review["status"] == "reviewed"
            assert review["script_review_fingerprint"]

            rendered = export_router.render_chapter(CHAPTER_ID)
            assert rendered["chapter_render"]["rendered"] == 1
            assert image_router._current_rendered_path(
                CHAPTER_ID,
                0,
                manifest_utils.load_manifest_raw(CHAPTER_ID),
            ) is not None

            qc = editorial_router.set_final_qc_page(
                editorial_router.FinalQCPageRequest(
                    chapter_id=CHAPTER_ID, page_index=0, approved=True
                )
            )
            assert qc["ready_for_export"] is True

            response = export_router.export_chapter(CHAPTER_ID)
            archive_path = Path(response.path)
            assert archive_path.is_file()
            with zipfile.ZipFile(archive_path) as archive:
                assert archive.namelist() == ["page_001.png"]
                payload = archive.read("page_001.png")
                assert payload.startswith(b"\x89PNG")
        finally:
            for patcher in reversed(patchers):
                patcher.stop()
