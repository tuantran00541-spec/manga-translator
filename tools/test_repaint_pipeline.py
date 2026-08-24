import os
import shutil
import sys
import threading
import time
import uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np

from app.config import PROCESSED_DIR, RAW_DIR
from app.manifest_utils import load_manifest_raw, save_manifest_raw
from app.pipeline import ChapterPipeline, write_image


class DummyInpainter:
    def inpaint(self, image: np.ndarray, boxes: list) -> np.ndarray:
        return image.copy()

    def inpaint_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return image.copy()


class DummyDetector:
    def detect(self, image: np.ndarray, *, parallel: bool = False) -> list:
        return []


def create_test_chapter(chapter_id: str, w: int = 200, h: int = 200) -> tuple[ChapterPipeline, Path, Path]:
    raw_dir = RAW_DIR / chapter_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = PROCESSED_DIR / chapter_id
    processed_dir.mkdir(parents=True, exist_ok=True)

    img_0 = np.full((h, w, 3), 255, dtype=np.uint8)
    img_1 = np.full((h, w, 3), 200, dtype=np.uint8)
    p0 = raw_dir / "000.png"
    p1 = raw_dir / "001.png"
    write_image(p0, img_0)
    write_image(p1, img_1)

    manifest = {
        "chapter_id": chapter_id,
        "source_url": None,
        "pages": [
            {
                "original": p0.as_posix(),
                "clean": (processed_dir / "clean_000.png").as_posix(),
                "boxes": [],
                "skipped": False,
                "excluded_regions": [],
                "source_page": 0,
                "slice_index": 0,
            },
            {
                "original": p1.as_posix(),
                "clean": (processed_dir / "clean_001.png").as_posix(),
                "boxes": [],
                "skipped": False,
                "excluded_regions": [],
                "source_page": 1,
                "slice_index": 0,
            },
        ],
    }
    write_image(processed_dir / "clean_000.png", img_0)
    write_image(processed_dir / "clean_001.png", img_1)
    save_manifest_raw(chapter_id, manifest)

    pipeline = ChapterPipeline()
    pipeline._inpainter = DummyInpainter()
    pipeline._detector = DummyDetector()
    return pipeline, p0, p1


def make_rect_mask(h: int, w: int, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    return mask


def read_disk_mask(chapter_id: str, filename: str) -> np.ndarray | None:
    p = PROCESSED_DIR / chapter_id / f"manual_mask_{filename}"
    if not p.exists():
        return None
    raw = np.fromfile(str(p), dtype=np.uint8)
    return cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)


def cleanup_chapter(chapter_id: str) -> None:
    shutil.rmtree(RAW_DIR / chapter_id, ignore_errors=True)
    shutil.rmtree(PROCESSED_DIR / chapter_id, ignore_errors=True)


PASS = 0
FAIL = 0


def report(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS: {name}")
    else:
        FAIL += 1
        print(f"  FAIL: {name} - {detail}")


def test_1_primary_sequential() -> None:
    cid = uuid.uuid4().hex[:8]
    pipeline, p0, _ = create_test_chapter(cid)
    try:
        mask_a = make_rect_mask(200, 200, 10, 10, 50, 50)
        mask_b = make_rect_mask(200, 200, 60, 60, 100, 100)
        mask_c = make_rect_mask(200, 200, 110, 110, 150, 150)

        pipeline.repaint_mask(cid, 0, mask_a)
        disk_a = read_disk_mask(cid, p0.name)
        report("Test 1a: submit A -> mask A exists", disk_a is not None and np.array_equal(disk_a, mask_a))

        pipeline.repaint_mask(cid, 0, mask_b)
        disk_ab = read_disk_mask(cid, p0.name)
        expected_ab = np.maximum(mask_a, mask_b)
        report("Test 1b: submit B -> mask A U B exists", disk_ab is not None and np.array_equal(disk_ab, expected_ab))

        pipeline.repaint_mask(cid, 0, mask_c)
        disk_abc = read_disk_mask(cid, p0.name)
        expected_abc = np.maximum(expected_ab, mask_c)
        report("Test 1c: submit C -> mask A U B U C exists", disk_abc is not None and np.array_equal(disk_abc, expected_abc))
    finally:
        cleanup_chapter(cid)


def test_2_combined_request() -> None:
    cid = uuid.uuid4().hex[:8]
    pipeline, p0, _ = create_test_chapter(cid)
    try:
        mask_a = make_rect_mask(200, 200, 10, 10, 50, 50)
        mask_b = make_rect_mask(200, 200, 60, 60, 100, 100)
        mask_c = make_rect_mask(200, 200, 110, 110, 150, 150)
        combined = np.maximum(np.maximum(mask_a, mask_b), mask_c)

        pipeline.repaint_mask(cid, 0, combined)
        disk = read_disk_mask(cid, p0.name)
        report("Test 2: submit combined A+B+C -> mask A U B U C exists", disk is not None and np.array_equal(disk, combined))
    finally:
        cleanup_chapter(cid)


def test_3_duplicate_mask() -> None:
    cid = uuid.uuid4().hex[:8]
    pipeline, p0, _ = create_test_chapter(cid)
    try:
        mask_a = make_rect_mask(200, 200, 20, 20, 80, 80)
        pipeline.repaint_mask(cid, 0, mask_a)
        pipeline.repaint_mask(cid, 0, mask_a)
        disk = read_disk_mask(cid, p0.name)
        report("Test 3: duplicate submit A -> mask A idempotent", disk is not None and np.array_equal(disk, mask_a))
    finally:
        cleanup_chapter(cid)


def test_4_partial_overlap() -> None:
    cid = uuid.uuid4().hex[:8]
    pipeline, p0, _ = create_test_chapter(cid)
    try:
        mask_a = make_rect_mask(200, 200, 20, 20, 60, 60)
        mask_b = make_rect_mask(200, 200, 40, 40, 90, 90)
        pipeline.repaint_mask(cid, 0, mask_a)
        pipeline.repaint_mask(cid, 0, mask_b)
        expected = np.maximum(mask_a, mask_b)
        disk = read_disk_mask(cid, p0.name)
        report("Test 4: partial overlap -> mask A U B intact", disk is not None and np.array_equal(disk, expected))
    finally:
        cleanup_chapter(cid)


def test_5_reset() -> None:
    cid = uuid.uuid4().hex[:8]
    pipeline, p0, _ = create_test_chapter(cid)
    try:
        mask = make_rect_mask(200, 200, 10, 10, 50, 50)
        pipeline.repaint_mask(cid, 0, mask)
        pipeline.reset_manual_mask(cid, 0)
        disk = read_disk_mask(cid, p0.name)
        m = load_manifest_raw(cid)
        report("Test 5: reset -> mask unlinked and removed from manifest", disk is None and "manual_mask" not in m["pages"][0])
    finally:
        cleanup_chapter(cid)


def test_6_reset_then_repaint() -> None:
    cid = uuid.uuid4().hex[:8]
    pipeline, p0, _ = create_test_chapter(cid)
    try:
        mask_a = make_rect_mask(200, 200, 10, 10, 50, 50)
        mask_b = make_rect_mask(200, 200, 60, 60, 90, 90)
        mask_c = make_rect_mask(200, 200, 120, 120, 160, 160)

        pipeline.repaint_mask(cid, 0, mask_a)
        pipeline.repaint_mask(cid, 0, mask_b)
        pipeline.reset_manual_mask(cid, 0)
        pipeline.repaint_mask(cid, 0, mask_c)

        disk = read_disk_mask(cid, p0.name)
        report("Test 6: reset then repaint C -> only C exists", disk is not None and np.array_equal(disk, mask_c))
    finally:
        cleanup_chapter(cid)


def test_7_failure_preserves_previous() -> None:
    cid = uuid.uuid4().hex[:8]
    pipeline, p0, _ = create_test_chapter(cid)
    try:
        mask_a = make_rect_mask(200, 200, 10, 10, 50, 50)
        mask_b = make_rect_mask(200, 200, 60, 60, 90, 90)
        pipeline.repaint_mask(cid, 0, mask_a)

        orig_do_reinpaint = pipeline._do_reinpaint

        def failing_reinpaint(*args, **kwargs):
            raise RuntimeError("Forced inpaint failure")

        pipeline._do_reinpaint = failing_reinpaint
        failed = False
        try:
            pipeline.repaint_mask(cid, 0, mask_b)
        except RuntimeError:
            failed = True

        pipeline._do_reinpaint = orig_do_reinpaint
        disk = read_disk_mask(cid, p0.name)
        m = load_manifest_raw(cid)
        report(
            "Test 7: failed repaint keeps previous mask A intact",
            failed and disk is not None and np.array_equal(disk, mask_a) and m["pages"][0].get("manual_mask") is not None
        )
    finally:
        cleanup_chapter(cid)


def test_8_concurrent_repaint_lost_update() -> None:
    cid = uuid.uuid4().hex[:8]
    pipeline, p0, _ = create_test_chapter(cid)
    try:
        mask_a = make_rect_mask(200, 200, 10, 10, 50, 50)
        mask_b = make_rect_mask(200, 200, 80, 80, 120, 120)
        barrier = threading.Barrier(2)

        def worker_a():
            barrier.wait()
            pipeline.repaint_mask(cid, 0, mask_a)

        def worker_b():
            barrier.wait()
            pipeline.repaint_mask(cid, 0, mask_b)

        t1 = threading.Thread(target=worker_a)
        t2 = threading.Thread(target=worker_b)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        expected = np.maximum(mask_a, mask_b)
        disk = read_disk_mask(cid, p0.name)
        report("Test 8: concurrent repaints on same page -> A U B accumulated without loss", disk is not None and np.array_equal(disk, expected))
    finally:
        cleanup_chapter(cid)


def test_9_reset_vs_repaint_stale_result() -> None:
    cid = uuid.uuid4().hex[:8]
    pipeline, p0, _ = create_test_chapter(cid)
    try:
        mask_a = make_rect_mask(200, 200, 10, 10, 50, 50)
        mask_b = make_rect_mask(200, 200, 80, 80, 120, 120)
        pipeline.repaint_mask(cid, 0, mask_a)

        def delayed_repaint():
            pipeline.repaint_mask(cid, 0, mask_b)

        def execute_reset():
            time.sleep(0.01)
            pipeline.reset_manual_mask(cid, 0)

        t1 = threading.Thread(target=delayed_repaint)
        t2 = threading.Thread(target=execute_reset)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        disk = read_disk_mask(cid, p0.name)
        m = load_manifest_raw(cid)
        is_consistent = (disk is None and "manual_mask" not in m["pages"][0]) or (
            disk is not None and (np.array_equal(disk, mask_b) or np.array_equal(disk, np.maximum(mask_a, mask_b)))
        )
        report("Test 9: reset vs repaint synchronization -> deterministic clean state", is_consistent)
    finally:
        cleanup_chapter(cid)


def test_10_cross_page_isolation() -> None:
    cid = uuid.uuid4().hex[:8]
    pipeline, p0, p1 = create_test_chapter(cid)
    try:
        mask_0 = make_rect_mask(200, 200, 10, 10, 50, 50)
        pipeline.repaint_mask(cid, 0, mask_0)

        disk_0 = read_disk_mask(cid, p0.name)
        disk_1 = read_disk_mask(cid, p1.name)
        m = load_manifest_raw(cid)

        report(
            "Test 10: page 0 repaint does not affect page 1",
            disk_0 is not None and disk_1 is None and "manual_mask" not in m["pages"][1] and m["pages"][0].get("manual_mask") is not None
        )
    finally:
        cleanup_chapter(cid)


def test_11_reload_reprocess() -> None:
    cid = uuid.uuid4().hex[:8]
    pipeline, p0, _ = create_test_chapter(cid)
    try:
        mask_a = make_rect_mask(200, 200, 10, 10, 40, 40)
        mask_b = make_rect_mask(200, 200, 70, 70, 110, 110)
        pipeline.repaint_mask(cid, 0, mask_a)
        pipeline.repaint_mask(cid, 0, mask_b)
        expected_ab = np.maximum(mask_a, mask_b)

        pipeline.process_pages(cid, [0])
        disk = read_disk_mask(cid, p0.name)
        m = load_manifest_raw(cid)
        report(
            "Test 11: process_pages preserves and reapplies persistent manual mask A U B",
            disk is not None and np.array_equal(disk, expected_ab) and m["pages"][0].get("manual_mask") is not None
        )
    finally:
        cleanup_chapter(cid)


def main() -> None:
    print("=== RUNNING REPAINT PIPELINE REGRESSION & CONCURRENCY TESTS ===")
    test_1_primary_sequential()
    test_2_combined_request()
    test_3_duplicate_mask()
    test_4_partial_overlap()
    test_5_reset()
    test_6_reset_then_repaint()
    test_7_failure_preserves_previous()
    test_8_concurrent_repaint_lost_update()
    test_9_reset_vs_repaint_stale_result()
    test_10_cross_page_isolation()
    test_11_reload_reprocess()

    print(f"\nTotal: {PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
