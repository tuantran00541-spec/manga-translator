import sys
import cv2
import numpy as np
from app.detector.combined_detector import CombinedTextDetector
from app.detector.mask_builder import build_mask


def read_image(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def describe(label, boxes, img_w, img_h):
    page_area = img_w * img_h
    print(f"\n--- {label}: {len(boxes)} box ---")
    for b in boxes:
        bw, bh = b.x2 - b.x1, b.y2 - b.y1
        area_ratio = (bw * bh) / page_area
        width_ratio = bw / img_w
        aspect = bw / bh if bh > 0 else 0
        print(
            f"x1={b.x1} y1={b.y1} x2={b.x2} y2={b.y2} "
            f"conf={b.confidence:.2f} area%={area_ratio*100:.1f} "
            f"width%={width_ratio*100:.1f} aspect={aspect:.2f}"
        )


def main(path):
    image = read_image(path)
    h, w = image.shape[:2]
    print(f"image: {path} size={w}x{h}")

    detector = CombinedTextDetector()
    bubble_boxes = detector.bubble_detector.detect(image)
    text_boxes = detector.text_detector.detect(image)
    final_boxes = detector.detect(image)

    describe("bubble_detector (sau filter)", bubble_boxes, w, h)
    describe("text_detector (sau filter)", text_boxes, w, h)
    describe("final (bubble + text sau khi loại trùng)", final_boxes, w, h)

    mask = build_mask((h, w), final_boxes)
    covered = (mask > 0).sum() / mask.size
    print(f"\nTổng % diện tích ảnh bị mask (sẽ bị LaMa vẽ đè): {covered*100:.1f}%")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/raw/CHAPTER_ID/sliced/002_00.webp")
