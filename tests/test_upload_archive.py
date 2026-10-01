import io
import zipfile

import cv2
import numpy as np

import app.pipeline as pipeline_module
from app.pipeline import ChapterPipeline


def test_a_broken_image_in_an_archive_is_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_module, "RAW_DIR", tmp_path)
    pipeline = ChapterPipeline()
    built = {}
    monkeypatch.setattr(pipeline, "_build_chapter_from_raw_paths",
                        lambda chapter_id, paths, **kwargs: built.setdefault("paths", paths))
    good = cv2.imencode(".png", np.full((40, 40, 3), 200, np.uint8))[1].tobytes()
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("001.png", good)
        z.writestr("002.png", b"not an image")
        z.writestr("003.png", good)
    pipeline.create_chapter_from_uploads("a1b2c3d4", [("chapter.zip", archive.getvalue())])
    assert [path.name for path in built["paths"]] == ["000.png", "001.png"]


def test_archive_pages_keep_page_order_and_skip_mac_resource_files(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_module, "RAW_DIR", tmp_path)
    pipeline = ChapterPipeline()
    built = {}
    monkeypatch.setattr(pipeline, "_build_chapter_from_raw_paths",
                        lambda chapter_id, paths, **kwargs: built.setdefault("paths", paths))
    pages = {n: cv2.imencode(".png", np.full((40, 40, 3), n, np.uint8))[1].tobytes() for n in (2, 10)}
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("Chapter 1/10.png", pages[10])
        z.writestr("Chapter 1/._2.png", b"\x00\x05\x16\x07")
        z.writestr("Chapter 1/2.png", pages[2])
    pipeline.create_chapter_from_uploads("a1b2c3d5", [("chapter.cbz", archive.getvalue())])
    assert [cv2.imread(str(path))[0, 0, 0] for path in built["paths"]] == [2, 10]
