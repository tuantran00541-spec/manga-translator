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
    raw = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def generate_synthetic_image(
    w: int,
    h: int,
    archetype: str = "model_required",
    shortcut_type: str | None = None,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.RandomState(seed)
    img = np.zeros((h, w, 3), dtype=np.uint8)

    if archetype == "shortcut":
        if shortcut_type == "white":
            img.fill(250)
            noise = rng.randint(0, 5, (h, w, 3), dtype=np.uint8)
            img = np.clip(img.astype(np.int16) - noise, 235, 255).astype(np.uint8)
        elif shortcut_type == "black":
            img.fill(10)
            noise = rng.randint(0, 5, (h, w, 3), dtype=np.uint8)
            img = np.clip(img.astype(np.int16) + noise, 0, 30).astype(np.uint8)
        elif shortcut_type == "low_std":
            base_col = rng.randint(80, 180, (3,), dtype=np.uint8)
            img[:, :] = base_col
            noise = rng.randint(-3, 4, (h, w, 3))
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        else:
            img.fill(255)
    else:
        for _ in range(rng.randint(3, 8)):
            pt1 = (rng.randint(0, w), rng.randint(0, h))
            pt2 = (rng.randint(0, w), rng.randint(0, h))
            color = [int(c) for c in rng.randint(0, 256, (3,))]
            cv2.rectangle(img, pt1, pt2, color, -1)
        for _ in range(rng.randint(2, 6)):
            center = (rng.randint(0, w), rng.randint(0, h))
            radius = rng.randint(10, max(20, min(w, h) // 4))
            color = [int(c) for c in rng.randint(0, 256, (3,))]
            cv2.circle(img, center, radius, color, -1)

    return img


def generate_synthetic_mask(
    w: int,
    h: int,
    mask_type: str = "M1_bubble_10pct",
    seed: int = 42,
) -> tuple[np.ndarray, list[dict]]:
    rng = np.random.RandomState(seed)
    mask = np.zeros((h, w), dtype=np.uint8)
    boxes = []

    if mask_type == "M1_bubble_10pct":
        bw, bh = int(w * 0.35), int(h * 0.35)
        cx, cy = w // 2, h // 2
        x1, y1 = max(0, cx - bw // 2), max(0, cy - bh // 2)
        x2, y2 = min(w, x1 + bw), min(h, y1 + bh)
        cv2.ellipse(mask, (cx, cy), (bw // 2, bh // 2), 0, 0, 360, 255, -1)
        boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "confidence": 0.95})

    elif mask_type == "M2_bubble_25pct":
        bw, bh = int(w * 0.55), int(h * 0.55)
        cx, cy = w // 2, h // 2
        x1, y1 = max(0, cx - bw // 2), max(0, cy - bh // 2)
        x2, y2 = min(w, x1 + bw), min(h, y1 + bh)
        cv2.ellipse(mask, (cx, cy), (bw // 2, bh // 2), 0, 0, 360, 255, -1)
        boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "confidence": 0.95})

    elif mask_type == "M3_bubble_50pct":
        bw, bh = int(w * 0.8), int(h * 0.8)
        cx, cy = w // 2, h // 2
        x1, y1 = max(0, cx - bw // 2), max(0, cy - bh // 2)
        x2, y2 = min(w, x1 + bw), min(h, y1 + bh)
        cv2.ellipse(mask, (cx, cy), (bw // 2, bh // 2), 0, 0, 360, 255, -1)
        boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "confidence": 0.95})

    elif mask_type == "M4_thin_horizontal":
        bh = max(10, int(h * 0.08))
        bw = int(w * 0.8)
        cx, cy = w // 2, h // 2
        x1, y1 = max(0, cx - bw // 2), max(0, cy - bh // 2)
        x2, y2 = min(w, x1 + bw), min(h, y1 + bh)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "confidence": 0.95})

    elif mask_type == "M5_thin_vertical":
        bw = max(10, int(w * 0.08))
        bh = int(h * 0.8)
        cx, cy = w // 2, h // 2
        x1, y1 = max(0, cx - bw // 2), max(0, cy - bh // 2)
        x2, y2 = min(w, x1 + bw), min(h, y1 + bh)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "confidence": 0.95})

    elif mask_type == "M6_irregular_blob":
        center = (w // 2, h // 2)
        radius = int(min(w, h) * 0.3)
        pts = []
        for angle in range(0, 360, 30):
            rad = math.radians(angle)
            r = radius + rng.randint(-radius // 3, radius // 3)
            px = int(center[0] + r * math.cos(rad))
            py = int(center[1] + r * math.sin(rad))
            pts.append([px, py])
        pts_arr = np.array([pts], dtype=np.int32)
        cv2.fillPoly(mask, pts_arr, 255)
        x1, y1, bw, bh = cv2.boundingRect(pts_arr)
        boxes.append({"x1": x1, "y1": y1, "x2": x1 + bw, "y2": y1 + bh, "confidence": 0.95})

    elif mask_type == "M7_disconnected_multi":
        num_blobs = 3
        for i in range(num_blobs):
            cx = int((i + 1) * w / (num_blobs + 1))
            cy = int((i + 1) * h / (num_blobs + 1))
            rad_x, rad_y = max(5, int(w * 0.08)), max(5, int(h * 0.08))
            cv2.ellipse(mask, (cx, cy), (rad_x, rad_y), 0, 0, 360, 255, -1)
            boxes.append({"x1": cx - rad_x, "y1": cy - rad_y, "x2": cx + rad_x, "y2": cy + rad_y, "confidence": 0.90})

    elif mask_type == "M8_clustered_multi":
        num_blobs = 4
        center_x, center_y = w // 2, h // 2
        for _ in range(num_blobs):
            cx = center_x + rng.randint(-w // 6, w // 6)
            cy = center_y + rng.randint(-h // 6, h // 6)
            rad_x, rad_y = max(5, int(w * 0.07)), max(5, int(h * 0.07))
            cv2.ellipse(mask, (cx, cy), (rad_x, rad_y), 0, 0, 360, 255, -1)
            boxes.append({"x1": cx - rad_x, "y1": cy - rad_y, "x2": cx + rad_x, "y2": cy + rad_y, "confidence": 0.90})

    else:
        cv2.circle(mask, (w // 2, h // 2), min(w, h) // 4, 255, -1)
        boxes.append({"x1": w // 4, "y1": h // 4, "x2": 3 * w // 4, "y2": 3 * h // 4, "confidence": 0.90})

    return mask, boxes


def generate_case(
    w: int,
    h: int,
    mask_type: str,
    execution_mode: str = "model_required",
    expected_shortcut_type: str | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[BubbleBox], dict[str, Any]]:
    img = generate_synthetic_image(w, h, archetype=execution_mode, shortcut_type=expected_shortcut_type, seed=seed)
    mask, boxes_meta = generate_synthetic_mask(w, h, mask_type=mask_type, seed=seed)

    boxes = [
        BubbleBox(
            x1=int(b["x1"]),
            y1=int(b["y1"]),
            x2=int(b["x2"]),
            y2=int(b["y2"]),
            confidence=float(b["confidence"]),
            mask=None,
        )
        for b in boxes_meta
    ]

    case_meta = {
        "width": w,
        "height": h,
        "mask_type": mask_type,
        "expected_execution": execution_mode,
        "expected_shortcut_type": expected_shortcut_type,
        "boxes": boxes_meta,
        "seed": seed,
    }

    return img, mask, boxes, case_meta


def generate_corpus(
    out_dir: Path | str,
    sizes: list[tuple[int, int]] | None = None,
    mask_types: list[str] | None = None,
    seed: int = 42,
) -> list[dict]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    test_sizes = sizes or BENCHMARK_SIZES
    test_masks = mask_types or MASK_TYPES

    manifest_cases = []

    case_idx = 0
    for w, h in test_sizes:
        for m_type in test_masks:
            case_id = f"case_{case_idx:03d}_{w}x{h}_{m_type}"
            case_dir = out_path / case_id
            case_dir.mkdir(parents=True, exist_ok=True)

            img, mask, boxes, meta = generate_case(w, h, m_type, execution_mode="model_required", seed=seed + case_idx)
            meta["case_id"] = case_id
            meta["original_path"] = str((case_dir / "original.png").resolve())
            meta["mask_path"] = str((case_dir / "mask.png").resolve())

            cv2.imwrite(str(case_dir / "original.png"), img)
            cv2.imwrite(str(case_dir / "mask.png"), mask)

            orig_bytes = cv2.imencode(".png", img)[1].tobytes()
            mask_bytes = cv2.imencode(".png", mask)[1].tobytes()

            meta["original_sha256"] = hashlib.sha256(orig_bytes).hexdigest()
            meta["mask_sha256"] = hashlib.sha256(mask_bytes).hexdigest()
            meta["workload_sha256"] = compute_workload_sha256(meta, orig_bytes, mask_bytes)

            with open(case_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            manifest_cases.append(meta)
            case_idx += 1

    shortcut_archetypes = [
        ("shortcut_white", "shortcut", "white"),
        ("shortcut_black", "shortcut", "black"),
        ("shortcut_low_std", "shortcut", "low_std"),
    ]

    for label, mode, stype in shortcut_archetypes:
        w, h = 512, 512
        case_id = f"case_shortcut_{label}"
        case_dir = out_path / case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        img, mask, boxes, meta = generate_case(w, h, "M1_bubble_10pct", execution_mode=mode, expected_shortcut_type=stype, seed=seed + case_idx)
        meta["case_id"] = case_id
        meta["original_path"] = str((case_dir / "original.png").resolve())
        meta["mask_path"] = str((case_dir / "mask.png").resolve())

        cv2.imwrite(str(case_dir / "original.png"), img)
        cv2.imwrite(str(case_dir / "mask.png"), mask)

        orig_bytes = cv2.imencode(".png", img)[1].tobytes()
        mask_bytes = cv2.imencode(".png", mask)[1].tobytes()

        meta["original_sha256"] = hashlib.sha256(orig_bytes).hexdigest()
        meta["mask_sha256"] = hashlib.sha256(mask_bytes).hexdigest()
        meta["workload_sha256"] = compute_workload_sha256(meta, orig_bytes, mask_bytes)

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
