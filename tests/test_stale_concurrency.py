import os
import shutil
import sys
import threading
import uuid
from pathlib import Path
import cv2
import numpy as np

from app.config import PROCESSED_DIR, RAW_DIR
from app.detector.bubble_detector import BubbleBox
from app.manifest_utils import (
    capture_processing_state,
    get_manifest_lock,
    is_processing_state_current,
    load_manifest_raw,
    save_manifest_raw,
)
from app.pipeline import ChapterPipeline, write_image


class DummyInpainter:
    def inpaint(self, image: np.ndarray, boxes: list) -> np.ndarray:
        res = image.copy()
        for b in boxes:
            if hasattr(b, "x1"):
                x1, y1, x2, y2 = b.x1, b.y1, b.x2, b.y2
            else:
                x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
            cv2.rectangle(res, (x1, y1), (x2, y2), (0, 255, 0), -1)
        return res

    def inpaint_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        res = image.copy()
        if mask is not None:
            res[mask > 10] = [0, 0, 255]
        return res


class DummyDetector:
    def __init__(self, boxes: list[BubbleBox] | None = None):
        self._boxes = boxes or [
            BubbleBox(10, 10, 40, 40, 1.0, None),
            BubbleBox(50, 50, 80, 80, 1.0, None),
            BubbleBox(90, 90, 120, 120, 1.0, None),
        ]

    def detect(self, image: np.ndarray, *, parallel: bool = False) -> list[BubbleBox]:
        return list(self._boxes)


def create_test_chapter(
    chapter_id: str,
    w: int = 200,
    h: int = 200,
    initial_boxes: list[dict] | None = None,
    initial_clean: bool = True,
    excluded_regions: list[dict] | None = None,
) -> tuple[ChapterPipeline, Path]:
    raw_dir = RAW_DIR / chapter_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = PROCESSED_DIR / chapter_id
    processed_dir.mkdir(parents=True, exist_ok=True)

    img_0 = np.full((h, w, 3), 255, dtype=np.uint8)
    p0 = raw_dir / "000.png"
    write_image(p0, img_0)

    clean_path = (processed_dir / "clean_000.png").as_posix() if initial_clean else None
    if initial_clean:
        write_image(processed_dir / "clean_000.png", img_0)

    boxes = initial_boxes or []

    manifest = {
        "chapter_id": chapter_id,
        "source_url": None,
        "pages": [
            {
                "original": p0.as_posix(),
                "clean": clean_path,
                "boxes": boxes,
                "skipped": False,
                "excluded_regions": excluded_regions or [],
                "source_page": 0,
                "slice_index": 0,
            }
        ],
    }
    save_manifest_raw(chapter_id, manifest)

    pipeline = ChapterPipeline()
    pipeline._inpainter = DummyInpainter()
    pipeline._detector = DummyDetector()
    return pipeline, p0


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


def test_1_delete_one_by_one_regression() -> None:
    cid = uuid.uuid4().hex[:8]
    initial_boxes = [
        {"x1": 10, "y1": 10, "x2": 40, "y2": 40, "confidence": 1.0, "mask": None},
        {"x1": 50, "y1": 50, "x2": 80, "y2": 80, "confidence": 1.0, "mask": None},
        {"x1": 90, "y1": 90, "x2": 120, "y2": 120, "confidence": 1.0, "mask": None},
    ]
    pipeline, p0 = create_test_chapter(cid, initial_boxes=initial_boxes)
    try:
        process_started = threading.Event()
        mutation_done = threading.Event()
        orig_process_page = pipeline._process_page

        def hooked_process_page(img_path, processed_dir, excluded_regions=None, existing_boxes=None, *, parallel_detectors=False):
            result = orig_process_page(img_path, processed_dir, excluded_regions=excluded_regions, existing_boxes=existing_boxes, parallel_detectors=parallel_detectors)
            process_started.set()
            mutation_done.wait(timeout=5)
            return result

        pipeline._process_page = hooked_process_page

        def run_stale_process():
            pipeline.process_pages(cid, [0])

        t1 = threading.Thread(target=run_stale_process)
        t1.start()

        process_started.wait(timeout=5)

        pipeline.remove_box(cid, 0, 0)
        m1 = load_manifest_raw(cid)
        active_boxes_1 = [b for b in m1["pages"][0]["boxes"] if not b.get("removed")]
        report("Test 1a: delete A -> only B, C active", len(active_boxes_1) == 2 and active_boxes_1[0]["x1"] == 50)

        mutation_done.set()
        t1.join(timeout=5)

        m2 = load_manifest_raw(cid)
        active_boxes_2 = [b for b in m2["pages"][0]["boxes"] if not b.get("removed")]
        report(
            "Test 1b: stale process commit discarded -> A does NOT reappear",
            len(active_boxes_2) == 2 and active_boxes_2[0]["x1"] == 50 and active_boxes_2[1]["x1"] == 90,
            f"Active boxes count: {len(active_boxes_2)}",
        )

        pipeline.remove_box(cid, 0, 1)
        m3 = load_manifest_raw(cid)
        active_boxes_3 = [b for b in m3["pages"][0]["boxes"] if not b.get("removed")]
        report(
            "Test 1c: delete B -> only C active, A does NOT reappear",
            len(active_boxes_3) == 1 and active_boxes_3[0]["x1"] == 90,
            f"Active boxes: {active_boxes_3}",
        )

        pipeline.remove_box(cid, 0, 2)
        m4 = load_manifest_raw(cid)
        active_boxes_4 = [b for b in m4["pages"][0]["boxes"] if not b.get("removed")]
        report(
            "Test 1d: delete C -> 0 active boxes, previous boxes do NOT reappear",
            len(active_boxes_4) == 0,
            f"Active boxes: {active_boxes_4}",
        )
    finally:
        cleanup_chapter(cid)


def test_2_process_pages_vs_repaint_mask() -> None:
    cid = uuid.uuid4().hex[:8]
    pipeline, p0 = create_test_chapter(cid)
    try:
        mask_a = make_rect_mask(200, 200, 10, 10, 50, 50)
        process_started = threading.Event()
        mutation_done = threading.Event()
        orig_process_page = pipeline._process_page

        def hooked_process_page(img_path, processed_dir, excluded_regions=None, existing_boxes=None, *, parallel_detectors=False):
            result = orig_process_page(img_path, processed_dir, excluded_regions=excluded_regions, existing_boxes=existing_boxes, parallel_detectors=parallel_detectors)
            process_started.set()
            mutation_done.wait(timeout=5)
            return result

        pipeline._process_page = hooked_process_page

        t1 = threading.Thread(target=lambda: pipeline.process_pages(cid, [0]))
        t1.start()

        process_started.wait(timeout=5)

        pipeline.repaint_mask(cid, 0, mask_a)

        mutation_done.set()
        t1.join(timeout=5)

        disk_mask = read_disk_mask(cid, p0.name)
        m = load_manifest_raw(cid)
        report(
            "Test 2: process_pages vs repaint_mask -> mask A survives",
            disk_mask is not None and np.array_equal(disk_mask, mask_a) and m["pages"][0].get("manual_mask") is not None,
        )
    finally:
        cleanup_chapter(cid)


def test_3_process_pages_vs_add_manual_box() -> None:
    cid = uuid.uuid4().hex[:8]
    pipeline, p0 = create_test_chapter(cid)
    try:
        process_started = threading.Event()
        mutation_done = threading.Event()
        orig_process_page = pipeline._process_page

        def hooked_process_page(img_path, processed_dir, excluded_regions=None, existing_boxes=None, *, parallel_detectors=False):
            result = orig_process_page(img_path, processed_dir, excluded_regions=excluded_regions, existing_boxes=existing_boxes, parallel_detectors=parallel_detectors)
            process_started.set()
            mutation_done.wait(timeout=5)
            return result

        pipeline._process_page = hooked_process_page

        t1 = threading.Thread(target=lambda: pipeline.process_pages(cid, [0]))
        t1.start()

        process_started.wait(timeout=5)

        pipeline.add_manual_box(cid, 0, 50, 50, 100, 100)

        mutation_done.set()
        t1.join(timeout=5)

        m = load_manifest_raw(cid)
        boxes = m["pages"][0].get("boxes", [])
        manual_boxes = [b for b in boxes if b.get("manual")]
        report(
            "Test 3: process_pages vs add_manual_box -> manual box M remains present",
            len(manual_boxes) == 1 and manual_boxes[0]["x1"] == 50,
            f"Found manual boxes: {manual_boxes}",
        )
    finally:
        cleanup_chapter(cid)


def test_4_process_pages_vs_update_box() -> None:
    cid = uuid.uuid4().hex[:8]
    initial_boxes = [
        {"x1": 10, "y1": 10, "x2": 50, "y2": 50, "confidence": 1.0, "mask": None},
    ]
    pipeline, p0 = create_test_chapter(cid, initial_boxes=initial_boxes)
    try:
        process_started = threading.Event()
        mutation_done = threading.Event()
        orig_process_page = pipeline._process_page

        def hooked_process_page(img_path, processed_dir, excluded_regions=None, existing_boxes=None, *, parallel_detectors=False):
            result = orig_process_page(img_path, processed_dir, excluded_regions=excluded_regions, existing_boxes=existing_boxes, parallel_detectors=parallel_detectors)
            process_started.set()
            mutation_done.wait(timeout=5)
            return result

        pipeline._process_page = hooked_process_page

        t1 = threading.Thread(target=lambda: pipeline.process_pages(cid, [0]))
        t1.start()

        process_started.wait(timeout=5)

        pipeline.update_box(cid, 0, 0, 20, 20, 80, 80)

        mutation_done.set()
        t1.join(timeout=5)

        m = load_manifest_raw(cid)
        boxes = m["pages"][0].get("boxes", [])
        report(
            "Test 4: process_pages vs update_box -> updated coords (20,20,80,80) survive",
            len(boxes) == 1 and boxes[0]["x1"] == 20 and boxes[0]["y1"] == 20 and boxes[0]["x2"] == 80 and boxes[0]["y2"] == 80,
            f"Found boxes: {boxes}",
        )
    finally:
        cleanup_chapter(cid)


def test_5_process_pages_vs_reset_manual_mask() -> None:
    cid = uuid.uuid4().hex[:8]
    pipeline, p0 = create_test_chapter(cid)
    try:
        mask_a = make_rect_mask(200, 200, 10, 10, 50, 50)
        pipeline.repaint_mask(cid, 0, mask_a)

        process_started = threading.Event()
        mutation_done = threading.Event()
        orig_process_page = pipeline._process_page

        def hooked_process_page(img_path, processed_dir, excluded_regions=None, existing_boxes=None, *, parallel_detectors=False):
            result = orig_process_page(img_path, processed_dir, excluded_regions=excluded_regions, existing_boxes=existing_boxes, parallel_detectors=parallel_detectors)
            process_started.set()
            mutation_done.wait(timeout=5)
            return result

        pipeline._process_page = hooked_process_page

        t1 = threading.Thread(target=lambda: pipeline.process_pages(cid, [0]))
        t1.start()

        process_started.wait(timeout=5)

        pipeline.reset_manual_mask(cid, 0)

        mutation_done.set()
        t1.join(timeout=5)

        disk_mask = read_disk_mask(cid, p0.name)
        m = load_manifest_raw(cid)
        report(
            "Test 5: process_pages vs reset_manual_mask -> reset state survives, mask not resurrected",
            disk_mask is None and "manual_mask" not in m["pages"][0],
        )
    finally:
        cleanup_chapter(cid)


def test_6_process_pages_vs_mark_skipped() -> None:
    cid = uuid.uuid4().hex[:8]
    pipeline, p0 = create_test_chapter(cid)
    try:
        process_started = threading.Event()
        mutation_done = threading.Event()
        orig_process_page = pipeline._process_page

        def hooked_process_page(img_path, processed_dir, excluded_regions=None, existing_boxes=None, *, parallel_detectors=False):
            result = orig_process_page(img_path, processed_dir, excluded_regions=excluded_regions, existing_boxes=existing_boxes, parallel_detectors=parallel_detectors)
            process_started.set()
            mutation_done.wait(timeout=5)
            return result

        pipeline._process_page = hooked_process_page

        t1 = threading.Thread(target=lambda: pipeline.process_pages(cid, [0]))
        t1.start()

        process_started.wait(timeout=5)

        pipeline.mark_skipped(cid, [0], True)

        mutation_done.set()
        t1.join(timeout=5)

        m = load_manifest_raw(cid)
        report(
            "Test 6: process_pages vs mark_skipped -> skipped remains True and authoritative",
            m["pages"][0].get("skipped") is True and m["pages"][0].get("boxes") == [],
        )
    finally:
        cleanup_chapter(cid)


def test_7_process_pages_vs_save_excluded_regions() -> None:
    cid = uuid.uuid4().hex[:8]
    reg_x = [{"x1": 0, "y1": 0, "x2": 50, "y2": 50}]
    reg_y = [{"x1": 60, "y1": 60, "x2": 120, "y2": 120}]
    pipeline, p0 = create_test_chapter(cid, excluded_regions=reg_x)
    try:
        process_started = threading.Event()
        mutation_done = threading.Event()
        orig_process_page = pipeline._process_page

        def hooked_process_page(img_path, processed_dir, excluded_regions=None, existing_boxes=None, *, parallel_detectors=False):
            result = orig_process_page(img_path, processed_dir, excluded_regions=excluded_regions, existing_boxes=existing_boxes, parallel_detectors=parallel_detectors)
            process_started.set()
            mutation_done.wait(timeout=5)
            return result

        pipeline._process_page = hooked_process_page

        t1 = threading.Thread(target=lambda: pipeline.process_pages(cid, [0]))
        t1.start()

        process_started.wait(timeout=5)

        with get_manifest_lock(cid):
            m = load_manifest_raw(cid)
            m["pages"][0]["excluded_regions"] = reg_y
            save_manifest_raw(cid, m)

        mutation_done.set()
        t1.join(timeout=5)

        m_final = load_manifest_raw(cid)
        report(
            "Test 7: process_pages vs save_excluded_regions -> Y remains canonical",
            m_final["pages"][0].get("excluded_regions") == reg_y and len(m_final["pages"][0].get("boxes", [])) == 0,
        )
    finally:
        cleanup_chapter(cid)


def test_8_no_orphaned_tmp_files() -> None:
    cid = uuid.uuid4().hex[:8]
    pipeline, p0 = create_test_chapter(cid)
    try:
        pipeline.process_pages(cid, [0])
        processed_dir = PROCESSED_DIR / cid
        tmp_files = list(processed_dir.glob("*.tmp*"))
        report("Test 8a: normal process_pages leaves no .tmp files", len(tmp_files) == 0, f"Found: {tmp_files}")

        process_started = threading.Event()
        mutation_done = threading.Event()
        orig_process_page = pipeline._process_page

        def hooked_process_page(img_path, processed_dir, excluded_regions=None, existing_boxes=None, *, parallel_detectors=False):
            result = orig_process_page(img_path, processed_dir, excluded_regions=excluded_regions, existing_boxes=existing_boxes, parallel_detectors=parallel_detectors)
            process_started.set()
            mutation_done.wait(timeout=5)
            return result

        pipeline._process_page = hooked_process_page

        t1 = threading.Thread(target=lambda: pipeline.process_pages(cid, [0]))
        t1.start()
        process_started.wait(timeout=5)
        pipeline.add_manual_box(cid, 0, 10, 10, 40, 40)
        mutation_done.set()
        t1.join(timeout=5)

        tmp_files_after = list(processed_dir.glob("*.tmp*"))
        report("Test 8b: stale discarded process_pages leaves no .tmp files", len(tmp_files_after) == 0, f"Found: {tmp_files_after}")
    finally:
        cleanup_chapter(cid)


def test_9_uncontended_process_pages_succeeds() -> None:
    cid = uuid.uuid4().hex[:8]
    pipeline, p0 = create_test_chapter(cid, initial_clean=False)
    try:
        m = pipeline.process_pages(cid, [0])
        page = m["pages"][0]
        report(
            "Test 9: uncontended process_pages succeeds and commits clean + boxes",
            page.get("clean") is not None and len(page.get("boxes", [])) == 3,
            f"Page: {page}",
        )
    finally:
        cleanup_chapter(cid)


def main() -> None:
    print("=== RUNNING STALE-WRITE CONCURRENCY & REGRESSION TESTS ===")
    test_1_delete_one_by_one_regression()
    test_2_process_pages_vs_repaint_mask()
    test_3_process_pages_vs_add_manual_box()
    test_4_process_pages_vs_update_box()
    test_5_process_pages_vs_reset_manual_mask()
    test_6_process_pages_vs_mark_skipped()
    test_7_process_pages_vs_save_excluded_regions()
    test_8_no_orphaned_tmp_files()
    test_9_uncontended_process_pages_succeeds()

    print(f"\nTotal: {PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
