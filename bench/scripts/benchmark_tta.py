import os
import sys
import time
import glob
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.config import BUBBLE_DETECTOR_MODEL, BUBBLE_CONF_THRESHOLD, RAW_DIR
from app.detector.bubble_detector import YoloDetector, BubbleBox


def compute_iou(box1: BubbleBox, box2: BubbleBox) -> float:
    x1 = max(box1.x1, box2.x1)
    y1 = max(box1.y1, box2.y1)
    x2 = min(box1.x2, box2.x2)
    y2 = min(box1.y2, box2.y2)

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0

    area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1)
    area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1)
    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def run_benchmark():
    print("=" * 60)
    print(" 🚀 Test-Time Augmentation (TTA) Benchmark A/B Test ")
    print("=" * 60)

    if not BUBBLE_DETECTOR_MODEL.is_file():
        print(f"❌ Model file not found at: {BUBBLE_DETECTOR_MODEL}")
        return

    print("Initializing YoloDetectors...")
    detector_single = YoloDetector(BUBBLE_DETECTOR_MODEL, BUBBLE_CONF_THRESHOLD, use_tta=False)
    detector_tta = YoloDetector(BUBBLE_DETECTOR_MODEL, BUBBLE_CONF_THRESHOLD, use_tta=True)

    image_paths = []
    for ext in ("*.jpg", "*.png", "*.webp"):
        image_paths.extend(glob.glob(os.path.join(RAW_DIR, "**", ext), recursive=True))

    if not image_paths:
        print("⚠️ No real images found in data/raw/. Generating synthetic test image (1024x1400)...")
        synthetic_img = np.full((1400, 1024, 3), 245, dtype=np.uint8)
        cv2.ellipse(synthetic_img, (512, 300), (150, 80), 0, 0, 360, (0, 0, 0), 3)
        cv2.rectangle(synthetic_img, (200, 600), (800, 750), (20, 20, 20), 2)
        test_images = [("synthetic_test_page.png", synthetic_img)]
    else:
        test_images = []
        for p in image_paths[:10]:
            img = cv2.imread(p)
            if img is not None:
                test_images.append((os.path.basename(p), img))

    print(f"Loaded {len(test_images)} test image(s).\n")

    total_time_single = 0.0
    total_time_tta = 0.0
    total_boxes_single = 0
    total_boxes_tta = 0
    total_new_boxes = 0
    conf_diffs = []

    for name, img in test_images:
        h, w = img.shape[:2]

        t0 = time.perf_counter()
        boxes_single = detector_single.detect(img)
        t_single = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        boxes_tta = detector_tta.detect(img)
        t_tta = (time.perf_counter() - t0) * 1000

        total_time_single += t_single
        total_time_tta += t_tta
        total_boxes_single += len(boxes_single)
        total_boxes_tta += len(boxes_tta)

        matched_tta = set()
        new_in_page = 0
        for b_single in boxes_single:
            best_iou = 0.0
            best_b_tta = None
            for idx, b_tta in enumerate(boxes_tta):
                iou = compute_iou(b_single, b_tta)
                if iou > best_iou:
                    best_iou = iou
                    best_b_tta = (idx, b_tta)
            if best_iou >= 0.5 and best_b_tta is not None:
                matched_tta.add(best_b_tta[0])
                conf_diffs.append(best_b_tta[1].confidence - b_single.confidence)

        new_in_page = len(boxes_tta) - len(matched_tta)
        total_new_boxes += new_in_page

        print(f"📄 [{name}] ({w}x{h}px)")
        print(f"   • Single-Pass : {t_single:.1f}ms | Boxes: {len(boxes_single)}")
        print(f"   • 3-Pass TTA  : {t_tta:.1f}ms ({t_tta / max(1e-3, t_single):.2f}x) | Boxes: {len(boxes_tta)}")
        print(f"   • New Boxes   : {new_in_page}")

    print("\n" + "=" * 60)
    print(" 📊 BENCHMARK SUMMARY REPORT ")
    print("=" * 60)
    print(f"• Total Single-Pass Time : {total_time_single:.1f} ms")
    print(f"• Total 3-Pass TTA Time  : {total_time_tta:.1f} ms")
    speed_factor = total_time_tta / max(1e-3, total_time_single)
    time_increase_pct = (speed_factor - 1.0) * 100
    print(f"• Execution Overhead     : {speed_factor:.2f}x (+{time_increase_pct:.1f}%)")
    print(f"• Single-Pass Total Boxes: {total_boxes_single}")
    print(f"• TTA Total Boxes        : {total_boxes_tta}")
    print(f"• Discovered New Boxes   : {total_new_boxes}")
    if conf_diffs:
        avg_conf_diff = np.mean(conf_diffs)
        print(f"• Avg Confidence Change  : {avg_conf_diff:+.4f}")
    print("=" * 60)


if __name__ == "__main__":
    run_benchmark()
