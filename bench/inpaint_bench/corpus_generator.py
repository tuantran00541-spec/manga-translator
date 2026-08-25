from __future__ import annotations
import json
import math
import hashlib
from pathlib import Path
from typing import Any
import numpy as np
import cv2

from app.detector.bubble_detector import BubbleBox

BENCHMARK_SIZES = [
    (128, 128),
    (256, 256),
    (384, 384),
    (512, 512),
    (640, 640),
    (768, 768),
    (1024, 1024),
    (1536, 1024),
    (1000, 300),
    (300, 1000),
    (1600, 300),
    (300, 1600),
]

MASK_TYPES = [
    "M1_bubble_10pct",
    "M2_bubble_25pct",
    "M3_bubble_50pct",
    "M4_thin_horizontal",
    "M5_thin_vertical",
    "M6_irregular_blob",
    "M7_disconnected_multi",
    "M8_clustered_multi",
]


def compute_workload_sha256(case_meta: dict[str, Any], original_bytes: bytes, mask_bytes: bytes) -> str:
    orig_hash = hashlib.sha256(original_bytes).hexdigest()
    mask_hash = hashlib.sha256(mask_bytes).hexdigest()
    boxes = sorted(
        case_meta.get("boxes", []),
        key=lambda b: (b.get("x1", 0), b.get("y1", 0), b.get("x2", 0), b.get("y2", 0))
    )
    canonical = {
        "case_id": case_meta.get("case_id", ""),
        "expected_execution": case_meta.get("expected_execution", "model_required"),
        "expected_shortcut_type": case_meta.get("expected_shortcut_type"),
        "width": case_meta.get("width", 0),
        "height": case_meta.get("height", 0),
        "mask_type": case_meta.get("mask_type", ""),
        "original_sha256": orig_hash,
        "mask_sha256": mask_hash,
        "boxes": boxes,
    }
    canonical_json = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def generate_synthetic_image(
    width: int,
    height: int,
    execution_mode: str = "model_required",
    expected_shortcut_type: str | None = None,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.RandomState(seed)

    sc_type = expected_shortcut_type
    if not sc_type and execution_mode.startswith("shortcut_"):
        sc_type = execution_mode[len("shortcut_"):]

    if sc_type == "white":
        return np.full((height, width, 3), 255, dtype=np.uint8)

    if sc_type == "black":
        return np.full((height, width, 3), 0, dtype=np.uint8)

    if sc_type == "low_std":
        return np.full((height, width, 3), 128, dtype=np.uint8)

    base = np.full((height, width, 3), 128, dtype=np.uint8)

    step = 4
    dot_y, dot_x = np.mgrid[0:height:step, 0:width:step]
    base[dot_y, dot_x] = [70, 70, 70]

    step2 = 8
    diag_y, diag_x = np.mgrid[0:height:step2, 0:width:step2]
    base[diag_y, diag_x] = [185, 185, 185]

    num_lines = max(10, int((width + height) / 40))
    for _ in range(num_lines):
        x1 = rng.randint(0, width)
        y1 = rng.randint(0, height)
        x2 = rng.randint(0, width)
        y2 = rng.randint(0, height)
        color = int(rng.randint(65, 180))
        thickness = rng.randint(1, 3)
        cv2.line(base, (x1, y1), (x2, y2), (color, color, color), thickness)

    margin = max(4, int(min(width, height) * 0.05))
    cv2.rectangle(base, (margin, margin), (width - margin, height - margin), (55, 55, 55), 2)

    return base


def generate_mask_and_boxes(
    mask_type: str, width: int, height: int, seed: int = 42
) -> tuple[np.ndarray, list[BubbleBox], dict]:
    rng = np.random.RandomState(seed + 1000)
    mask = np.zeros((height, width), dtype=np.uint8)
    boxes: list[BubbleBox] = []

    total_pixels = width * height

    if mask_type == "M1_bubble_10pct":
        target_area = total_pixels * 0.10
        aspect = 1.2
        r_x = int(math.sqrt(target_area / (math.pi * aspect)))
        r_y = int(r_x * aspect)
        r_x = max(6, min(r_x, width // 2 - 4))
        r_y = max(6, min(r_y, height // 2 - 4))
        cx, cy = width // 2, height // 2
        cv2.ellipse(mask, (cx, cy), (r_x, r_y), 0, 0, 360, 255, -1)
        x1, y1 = max(0, cx - r_x), max(0, cy - r_y)
        x2, y2 = min(width, cx + r_x), min(height, cy + r_y)
        boxes.append(BubbleBox(x1, y1, x2, y2, 0.95))

    elif mask_type == "M2_bubble_25pct":
        target_area = total_pixels * 0.25
        aspect = 1.1
        r_x = int(math.sqrt(target_area / (math.pi * aspect)))
        r_y = int(r_x * aspect)
        r_x = max(8, min(r_x, width // 2 - 4))
        r_y = max(8, min(r_y, height // 2 - 4))
        cx, cy = width // 2, height // 2
        cv2.ellipse(mask, (cx, cy), (r_x, r_y), 0, 0, 360, 255, -1)
        x1, y1 = max(0, cx - r_x), max(0, cy - r_y)
        x2, y2 = min(width, cx + r_x), min(height, cy + r_y)
        boxes.append(BubbleBox(x1, y1, x2, y2, 0.95))

    elif mask_type == "M3_bubble_50pct":
        target_area = total_pixels * 0.50
        aspect = float(height) / max(1, width)
        bw = int(math.sqrt(target_area / max(1e-3, aspect)))
        bh = int(bw * aspect)
        bw = max(10, min(bw, width - 8))
        bh = max(10, min(bh, height - 8))
        x1 = (width - bw) // 2
        y1 = (height - bh) // 2
        x2 = x1 + bw
        y2 = y1 + bh
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        boxes.append(BubbleBox(x1, y1, x2, y2, 0.95))

    elif mask_type == "M4_thin_horizontal":
        bw = max(10, int(width * 0.85))
        bh = max(6, int(height * 0.12))
        x1 = (width - bw) // 2
        y1 = (height - bh) // 2
        x2 = x1 + bw
        y2 = y1 + bh
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        boxes.append(BubbleBox(x1, y1, x2, y2, 0.95))

    elif mask_type == "M5_thin_vertical":
        bw = max(6, int(width * 0.15))
        bh = max(10, int(height * 0.85))
        x1 = (width - bw) // 2
        y1 = (height - bh) // 2
        x2 = x1 + bw
        y2 = y1 + bh
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        boxes.append(BubbleBox(x1, y1, x2, y2, 0.95))

    elif mask_type == "M6_irregular_blob":
        cx, cy = width // 2, height // 2
        radius = min(width, height) * 0.35
        num_pts = 14
        angles = np.linspace(0, 2 * np.pi, num_pts, endpoint=False)
        pts = []
        for i, a in enumerate(angles):
            r = radius * (0.65 + 0.55 * rng.rand())
            px = int(cx + r * np.cos(a))
            py = int(cy + r * np.sin(a))
            px = max(2, min(width - 2, px))
            py = max(2, min(height - 2, py))
            pts.append([px, py])
        pts_arr = np.array(pts, dtype=np.int32)
        cv2.fillPoly(mask, [pts_arr], 255)
        x1, y1, bw, bh = cv2.boundingRect(pts_arr)
        boxes.append(BubbleBox(x1, y1, x1 + bw, y1 + bh, 0.95))

    elif mask_type == "M7_disconnected_multi":
        quadrants = [
            (int(width * 0.25), int(height * 0.25)),
            (int(width * 0.75), int(height * 0.25)),
            (int(width * 0.50), int(height * 0.75)),
        ]
        r_x = max(6, int(width * 0.10))
        r_y = max(6, int(height * 0.10))
        for qx, qy in quadrants:
            cv2.ellipse(mask, (qx, qy), (r_x, r_y), 0, 0, 360, 255, -1)
            bx1, by1 = max(0, qx - r_x), max(0, qy - r_y)
            bx2, by2 = min(width, qx + r_x), min(height, qy + r_y)
            boxes.append(BubbleBox(bx1, by1, bx2, by2, 0.95))

    elif mask_type == "M8_clustered_multi":
        center_x = width // 2
        center_y = height // 2
        bw = max(10, int(width * 0.20))
        bh = max(8, int(height * 0.10))
        spacing_y = max(4, int(bh * 0.25))

        for idx, offset_mult in enumerate([-1.1, 0.0, 1.1]):
            cy = int(center_y + offset_mult * (bh + spacing_y))
            cx = center_x + int((idx - 1) * bw * 0.2)
            bx1 = max(2, cx - bw // 2)
            by1 = max(2, cy - bh // 2)
            bx2 = min(width - 2, bx1 + bw)
            by2 = min(height - 2, by1 + bh)
            cv2.rectangle(mask, (bx1, by1), (bx2, by2), 255, -1)
            boxes.append(BubbleBox(bx1, by1, bx2, by2, 0.95))

    mask_area_pixels = int(np.count_nonzero(mask > 127))
    mask_ratio = float(mask_area_pixels) / float(total_pixels)
    num_labels, _, _, _ = cv2.connectedComponentsWithStats((mask > 127).astype(np.uint8))

    metadata = {
        "mask_type": mask_type,
        "width": width,
        "height": height,
        "mask_area_pixels": mask_area_pixels,
        "mask_ratio": round(mask_ratio, 4),
        "component_count": max(0, num_labels - 1),
        "box_count": len(boxes),
    }

    return mask, boxes, metadata


def generate_case(
    width: int,
    height: int,
    mask_type: str,
    execution_mode: str = "model_required",
    expected_shortcut_type: str | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[BubbleBox], dict]:
    img = generate_synthetic_image(
        width,
        height,
        execution_mode=execution_mode,
        expected_shortcut_type=expected_shortcut_type,
        seed=seed,
    )
    mask, boxes, meta = generate_mask_and_boxes(mask_type, width, height, seed=seed)

    mask_bool = mask > 127
    sc_type = expected_shortcut_type or (execution_mode[len("shortcut_"):] if execution_mode.startswith("shortcut_") else None)

    if sc_type == "black":
        img[mask_bool] = [0, 0, 0]
    else:
        img[mask_bool] = [255, 255, 255]

    for b in boxes:
        bw = b.x2 - b.x1
        bh = b.y2 - b.y1
        cv2.rectangle(img, (b.x1, b.y1), (b.x2, b.y2), (20, 20, 20), 1)
        num_text_lines = max(1, min(6, bh // 15))
        for line_i in range(num_text_lines):
            ly = b.y1 + int((line_i + 1) * (bh / (num_text_lines + 1)))
            lx1 = b.x1 + int(bw * 0.15)
            lx2 = b.x2 - int(bw * 0.15)
            if lx2 > lx1:
                cv2.line(img, (lx1, ly), (lx2, ly), (30, 30, 30), 2)

    case_id = f"syn_{width}x{height}_{mask_type}"
    meta["case_id"] = case_id
    meta["expected_execution"] = execution_mode
    meta["expected_shortcut_type"] = expected_shortcut_type
    meta["boxes"] = [{"x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2, "confidence": b.confidence} for b in boxes]

    # Compute content hashes
    _, img_encoded = cv2.imencode(".png", img)
    _, mask_encoded = cv2.imencode(".png", mask)
    img_bytes = img_encoded.tobytes()
    mask_bytes = mask_encoded.tobytes()

    meta["original_sha256"] = hashlib.sha256(img_bytes).hexdigest()
    meta["mask_sha256"] = hashlib.sha256(mask_bytes).hexdigest()
    meta["workload_sha256"] = compute_workload_sha256(meta, img_bytes, mask_bytes)

    return img, mask, boxes, meta


def generate_corpus(
    output_dir: Path | str,
    sizes: list[tuple[int, int]] | None = None,
    mask_types: list[str] | None = None,
    seed: int = 42,
) -> list[dict]:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    sizes = sizes or BENCHMARK_SIZES
    mask_types = mask_types or MASK_TYPES

    manifest_cases = []

    for w, h in sizes:
        for m_type in mask_types:
            case_seed = seed + w * 1000 + h * 10 + len(m_type)
            img, mask, boxes, meta = generate_case(
                w, h, m_type, execution_mode="model_required", expected_shortcut_type=None, seed=case_seed
            )
            case_id = meta["case_id"]
            case_dir = out_path / case_id
            case_dir.mkdir(parents=True, exist_ok=True)

            img_file = case_dir / "original.png"
            mask_file = case_dir / "mask.png"
            meta_file = case_dir / "metadata.json"

            cv2.imwrite(str(img_file), img)
            cv2.imwrite(str(mask_file), mask)
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            manifest_cases.append(meta)

    shortcut_modes = [
        ("shortcut", "white"),
        ("shortcut", "black"),
        ("shortcut", "low_std"),
    ]
    for mode_name, sc_type in shortcut_modes:
        case_id = f"syn_512x512_shortcut_{sc_type}"
        img, mask, boxes, meta = generate_case(
            512, 512, "M1_bubble_10pct", execution_mode=mode_name, expected_shortcut_type=sc_type, seed=seed
        )
        meta["case_id"] = case_id
        case_dir = out_path / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(case_dir / "original.png"), img)
        cv2.imwrite(str(case_dir / "mask.png"), mask)
        with open(case_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        manifest_cases.append(meta)

    index_file = out_path / "corpus_index.json"
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump({"total_cases": len(manifest_cases), "cases": manifest_cases}, f, indent=2)

    return manifest_cases


def load_corpus(corpus_dir: Path | str) -> list[dict]:
    corpus_path = Path(corpus_dir)
    if not corpus_path.is_dir():
        return []

    cases = []
    for item in sorted(corpus_path.iterdir()):
        if not item.is_dir():
            continue
        orig_file = item / "original.png"
        mask_file = item / "mask.png"
        meta_file = item / "metadata.json"

        if not orig_file.is_file() or not mask_file.is_file():
            continue

        meta = {}
        if meta_file.is_file():
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                meta = {}

        if "case_id" not in meta:
            meta["case_id"] = item.name

        if "expected_execution" not in meta:
            meta["expected_execution"] = "model_required"

        orig_bytes = open(orig_file, "rb").read()
        mask_bytes = open(mask_file, "rb").read()

        meta["original_sha256"] = hashlib.sha256(orig_bytes).hexdigest()
        meta["mask_sha256"] = hashlib.sha256(mask_bytes).hexdigest()
        meta["workload_sha256"] = compute_workload_sha256(meta, orig_bytes, mask_bytes)

        meta["original_path"] = str(orig_file.resolve())
        meta["mask_path"] = str(mask_file.resolve())
        meta["case_dir"] = str(item.resolve())
        cases.append(meta)

    return cases
