"""Measure inpaint quality with synthetic ground truth and a real-chapter audit.

The production chapter does not have a clean reference image, so it cannot
produce a certified background-fidelity score by itself.  This benchmark
therefore reports two separate things:

* synthetic cases with a known clean background: target MAE/PSNR, residual
  error, halo error, and exact preservation outside the authorized mask;
* a real chapter audit: post-inpaint residue records and exact/tolerant pixel
  changes outside the reconstructed production authority mask.

No production threshold or mask geometry is changed by this script.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _box_mask(image: np.ndarray, box, inpainter) -> np.ndarray:
    """Reconstruct the mask passed to the production inpaint call."""
    from app.detector.bubble_detector import BubbleBox
    from app.detector.mask_builder import build_mask

    h, w = image.shape[:2]
    cx1, cy1, cx2, cy2 = inpainter._compute_crop_region(
        box.x1, box.y1, box.x2, box.y2, w, h
    )
    local = BubbleBox(
        box.x1 - cx1,
        box.y1 - cy1,
        box.x2 - cx1,
        box.y2 - cy1,
        box.confidence,
        box.mask,
        source_model=box.source_model,
        class_id=box.class_id,
        class_name=box.class_name,
        semantic_type=box.semantic_type,
        mask_source=box.mask_source,
        safe_to_inpaint=bool(box.safe_to_inpaint),
        ocr_eligible=bool(box.ocr_eligible),
        needs_review=bool(box.needs_review),
        source_role=box.source_role,
        deferred_reason=box.deferred_reason,
    )
    crop = image[cy1:cy2, cx1:cx2]
    mask = build_mask(crop.shape[:2], [local], crop)
    full = np.zeros((h, w), dtype=np.uint8)
    full[cy1:cy2, cx1:cx2] = mask
    return full


def _metric_row(
    source: np.ndarray,
    reference: np.ndarray,
    output: np.ndarray,
    target_mask: np.ndarray,
    authority_mask: np.ndarray,
) -> dict:
    if source.shape != reference.shape or source.shape != output.shape:
        raise ValueError("source, reference, and output shapes must match")
    target = target_mask > 127
    authority = authority_mask > 127
    if not np.any(target):
        raise ValueError("synthetic target mask is empty")

    source_i = source.astype(np.int16)
    reference_i = reference.astype(np.int16)
    output_i = output.astype(np.int16)
    source_delta = np.max(np.abs(output_i - source_i), axis=2)
    reference_delta = np.abs(output_i - reference_i)
    target_delta = reference_delta[target]
    target_max_delta = np.max(target_delta, axis=1)
    target_mse = float(np.mean(np.square(target_delta.astype(np.float32))))
    target_mae = float(np.mean(target_delta))
    target_psnr = (
        10.0 * math.log10((255.0 * 255.0) / target_mse)
        if target_mse > 0.0
        else float("inf")
    )

    outside = ~authority
    outside_delta = source_delta[outside]
    halo = authority & ~target
    halo_delta = reference_delta[halo]
    changed_target = source_delta[target]
    return {
        "target_pixels": int(np.count_nonzero(target)),
        "authority_pixels": int(np.count_nonzero(authority)),
        "target_mae": round(target_mae, 4),
        "target_psnr_db": round(target_psnr, 4) if math.isfinite(target_psnr) else None,
        "target_residual_fraction_gt24": round(
            float(np.mean(target_max_delta > 24)), 6
        ),
        "target_source_changed_fraction": round(
            float(np.mean(changed_target > 0)), 6
        ),
        "halo_mae": round(float(np.mean(halo_delta)) if halo_delta.size else 0.0, 4),
        "outside_changed_pixels_exact": int(np.count_nonzero(outside_delta > 0)),
        "outside_changed_pixels_gt8": int(np.count_nonzero(outside_delta > 8)),
        "outside_max_delta": int(outside_delta.max()) if outside_delta.size else 0,
        "whole_image_mae": round(float(np.mean(reference_delta)), 4),
    }


def _text_mask(box: tuple[int, int, int, int], lines: list[str], scale: float = 1.0) -> np.ndarray:
    x1, y1, x2, y2 = box
    mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    font_scale = max(0.7, float(scale))
    thickness = max(2, int(round(3 * font_scale)))
    line_gap = int(round(78 * font_scale))
    start_y = int(round(105 * font_scale))
    for index, line in enumerate(lines):
        cv2.putText(
            mask,
            line,
            (32, start_y + index * line_gap),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            255,
            thickness,
            cv2.LINE_AA,
        )
    return mask


def _background(kind: str, height: int, width: int) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    if kind == "flat_white":
        return np.full((height, width, 3), 248, dtype=np.uint8)
    if kind == "flat_black":
        return np.full((height, width, 3), 18, dtype=np.uint8)
    if kind == "gradient":
        blue = 45 + 120 * xx / max(1, width - 1)
        green = 80 + 80 * yy / max(1, height - 1)
        red = 180 - 90 * xx / max(1, width - 1)
        return np.dstack((blue, green, red)).astype(np.uint8)
    if kind in {"textured", "long_text"}:
        base = np.zeros((height, width, 3), dtype=np.uint8)
        base[:, :, 0] = (35 + 70 * xx / max(1, width - 1)).astype(np.uint8)
        base[:, :, 1] = (55 + 80 * yy / max(1, height - 1)).astype(np.uint8)
        base[:, :, 2] = (90 + 50 * (xx + yy) / max(1, width + height - 2)).astype(np.uint8)
        for offset in range(-height, width, 90):
            cv2.line(base, (offset, 0), (offset + height, height), (125, 95, 55), 3)
        for y in range(40, height, 130):
            cv2.ellipse(base, (width // 2, y), (110, 34), 0, 0, 360, (75, 135, 155), -1)
        return base
    raise ValueError(f"unknown synthetic background {kind!r}")


def _synthetic_case(name: str) -> tuple[np.ndarray, np.ndarray, object, np.ndarray]:
    from app.detector.bubble_detector import BubbleBox

    if name in {"flat_white", "flat_black", "gradient", "textured"}:
        height, width = 640, 960
        box_coords = (80, 96, 880, 544)
        lines = ["REMOVE THIS TEXT", "KEEP THE BACKGROUND"]
        scale = 1.25
    elif name == "long_text":
        height, width = 760, 1400
        box_coords = (72, 72, 1328, 688)
        lines = ["A LONG TEXT REGION WITH DETAILS", "SHOULD RETAIN ITS BACKGROUND", "AFTER CLEANUP"]
        scale = 1.15
    else:
        raise ValueError(f"unknown synthetic case {name!r}")

    kind = "textured" if name == "long_text" else name
    reference = _background(kind, height, width)
    x1, y1, x2, y2 = box_coords
    local = _text_mask(box_coords, lines, scale)
    alpha = local.astype(np.float32) / 255.0
    if name in {"flat_white", "gradient", "textured", "long_text"}:
        color = np.array([18, 18, 18], dtype=np.float32)
    else:
        color = np.array([242, 242, 242], dtype=np.float32)
    source = reference.astype(np.float32).copy()
    crop = source[y1:y2, x1:x2]
    crop[:] = crop * (1.0 - alpha[:, :, None]) + color * alpha[:, :, None]
    source = np.clip(source, 0, 255).astype(np.uint8)
    box = BubbleBox(
        x1,
        y1,
        x2,
        y2,
        0.99,
        local,
        source_model="text_segmenter.onnx",
        class_name="text_comic",
        semantic_type="speech_bubble",
        mask_source="text_segmenter",
        safe_to_inpaint=True,
        ocr_eligible=True,
        needs_review=False,
        source_role="text_segmenter",
    )
    target = np.zeros((height, width), dtype=np.uint8)
    target[y1:y2, x1:x2] = local
    return source, reference, box, target


def _save_case_card(path: Path, source: np.ndarray, reference: np.ndarray, output: np.ndarray, title: str) -> None:
    from PIL import Image, ImageDraw, ImageFont

    cards = []
    for image, label in ((source, "SOURCE"), (reference, "GROUND TRUTH"), (output, "OUTPUT")):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        pil.thumbnail((420, 280), Image.Resampling.LANCZOS)
        card = Image.new("RGB", (420, 310), "white")
        card.paste(pil, ((420 - pil.width) // 2, 24))
        draw = ImageDraw.Draw(card)
        draw.text((8, 6), label, fill="black", font=ImageFont.load_default())
        cards.append(card)
    sheet = Image.new("RGB", (1260, 340), "white")
    for index, card in enumerate(cards):
        sheet.paste(card, (index * 420, 30))
    ImageDraw.Draw(sheet).text((8, 8), title, fill="black", font=ImageFont.load_default())
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=88)


def run_synthetic(out_dir: Path) -> dict:
    from app.inpaint.adaptive_fast_inpainter import AdaptiveFastInpainter

    names = ["flat_white", "flat_black", "gradient", "textured", "long_text"]
    inpainter = AdaptiveFastInpainter()
    requested_workers = max(1, int(os.getenv("MANGA_INPAINT_ACCURACY_WORKERS", "1")))
    prepare = getattr(inpainter, "prepare_for_page_workers", None)
    if callable(prepare):
        prepare(requested_workers)
    preload_started = time.perf_counter()
    inpainter.preload()
    preload_ms = (time.perf_counter() - preload_started) * 1000.0

    rows = []
    for name in names:
        source, reference, box, target = _synthetic_case(name)
        authority = _box_mask(source, box, inpainter)
        started = time.perf_counter()
        output = inpainter.inpaint(source, [box])
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        production_quality = _metric_row(
            source, reference, output, target, authority
        )
        row = {
            "case": name,
            "production": {
                "elapsed_ms": round(elapsed_ms, 3),
                "inpaint_metrics": inpainter.last_metrics(),
                "quality": production_quality,
            },
        }
        # The real pipeline may correctly select a cheap flat/Telea path.  A
        # separate forced-LaMa pass is still needed to benchmark the neural
        # inpainting model itself on the same known background.
        cx1, cy1, cx2, cy2 = inpainter._compute_crop_region(
            box.x1, box.y1, box.x2, box.y2, source.shape[1], source.shape[0]
        )
        inpainter._begin_metrics(boxes=1)
        lama_started = time.perf_counter()
        lama_output = inpainter._lama_fill(
            source.copy(),
            source[cy1:cy2, cx1:cx2],
            authority[cy1:cy2, cx1:cx2],
            (cx1, cy1, cx2, cy2),
        )
        lama_elapsed_ms = (time.perf_counter() - lama_started) * 1000.0
        lama_quality = _metric_row(
            source, reference, lama_output, target, authority
        )
        row["lama_only"] = {
            "elapsed_ms": round(lama_elapsed_ms, 3),
            "inpaint_metrics": inpainter.last_metrics(),
            "quality": lama_quality,
        }
        rows.append(row)
        _save_case_card(
            out_dir / f"synthetic-{name}-production.jpg",
            source,
            reference,
            output,
            f"{name} production | target MAE={production_quality['target_mae']}",
        )
        _save_case_card(
            out_dir / f"synthetic-{name}-lama-only.jpg",
            source,
            reference,
            lama_output,
            f"{name} LaMa-only | target MAE={lama_quality['target_mae']}",
        )

    aggregate = {
        "cases": rows,
        "preload_ms": round(preload_ms, 3),
        "outside_changed_pixels_exact": sum(
            row["production"]["quality"]["outside_changed_pixels_exact"]
            + row["lama_only"]["quality"]["outside_changed_pixels_exact"]
            for row in rows
        ),
        "outside_changed_pixels_gt8": sum(
            row["production"]["quality"]["outside_changed_pixels_gt8"]
            + row["lama_only"]["quality"]["outside_changed_pixels_gt8"]
            for row in rows
        ),
        "production_mean_target_mae": round(
            float(
                np.mean(
                    [row["production"]["quality"]["target_mae"] for row in rows]
                )
            ),
            4,
        ),
        "lama_only_mean_target_mae": round(
            float(
                np.mean(
                    [row["lama_only"]["quality"]["target_mae"] for row in rows]
                )
            ),
            4,
        ),
        "production_mean_target_residual_fraction_gt24": round(
            float(
                np.mean(
                    [
                        row["production"]["quality"][
                            "target_residual_fraction_gt24"
                        ]
                        for row in rows
                    ]
                )
            ),
            6,
        ),
        "lama_only_mean_target_residual_fraction_gt24": round(
            float(
                np.mean(
                    [
                        row["lama_only"]["quality"][
                            "target_residual_fraction_gt24"
                        ]
                        for row in rows
                    ]
                )
            ),
            6,
        ),
    }
    aggregate["authority_safety"] = "pass" if aggregate["outside_changed_pixels_exact"] == 0 else "fail"
    return aggregate


def _real_chapter_audit(chapter_id: str) -> dict:
    from app.image_io import read_image
    from app.manifest_utils import load_manifest_raw
    from app.inpaint.fast_lama_inpainter import FastInpainter
    from scripts.model_e2e_gate import _authority_mask

    manifest = load_manifest_raw(chapter_id)
    helper = FastInpainter()
    page_rows = []
    total_pixels = 0
    total_authority = 0
    total_changed = 0
    total_inside_changed = 0
    total_outside_exact = 0
    total_outside_gt8 = 0
    residue_records = 0
    pages_with_residue = 0
    for page_index, page in enumerate(manifest.get("pages", [])):
        original_path = Path(str(page.get("original") or ""))
        clean_path = Path(str(page.get("clean") or ""))
        if not original_path.is_file() or not clean_path.is_file():
            page_rows.append({"page_index": page_index, "clean_missing": True})
            continue
        original = read_image(original_path)
        clean = read_image(clean_path)
        if original.shape != clean.shape:
            raise RuntimeError(f"page {page_index}: original/clean shape mismatch")
        authority = _authority_mask(original, page.get("boxes", []) or [], helper) > 127
        delta = np.max(
            np.abs(clean.astype(np.int16) - original.astype(np.int16)), axis=2
        )
        changed = delta > 0
        outside = ~authority
        residue = int(
            ((page.get("processing_metrics") or {}).get("detector") or {}).get(
                "post_inpaint_residue", 0
            )
            or 0
        )
        pixels = int(changed.size)
        authority_pixels = int(np.count_nonzero(authority))
        inside_changed = int(np.count_nonzero(changed & authority))
        outside_exact = int(np.count_nonzero(changed & outside))
        outside_gt8 = int(np.count_nonzero((delta > 8) & outside))
        total_pixels += pixels
        total_authority += authority_pixels
        total_changed += int(np.count_nonzero(changed))
        total_inside_changed += inside_changed
        total_outside_exact += outside_exact
        total_outside_gt8 += outside_gt8
        residue_records += residue
        pages_with_residue += int(residue > 0)
        page_rows.append(
            {
                "page_index": page_index,
                "clean_missing": False,
                "pixels": pixels,
                "authority_pixels": authority_pixels,
                "changed_pixels": int(np.count_nonzero(changed)),
                "changed_inside_authority": inside_changed,
                "outside_changed_pixels_exact": outside_exact,
                "outside_changed_pixels_gt8": outside_gt8,
                "outside_max_delta": int(delta[outside].max()) if np.any(outside) else 0,
                "residue_records": residue,
            }
        )

    processed_pages = [row for row in page_rows if not row.get("clean_missing")]
    return {
        "chapter_id": chapter_id,
        "pages": len(page_rows),
        "processed_pages": len(processed_pages),
        "total_pixels": total_pixels,
        "authority_pixels": total_authority,
        "changed_pixels": total_changed,
        "changed_inside_authority": total_inside_changed,
        "outside_changed_pixels_exact": total_outside_exact,
        "outside_changed_pixels_gt8": total_outside_gt8,
        "outside_safety": "pass" if total_outside_exact == 0 else "fail",
        "changed_authority_coverage_pct": round(
            100.0 * total_inside_changed / max(1, total_authority), 4
        ),
        "residue_records": residue_records,
        "pages_with_residue": pages_with_residue,
        "residue_record_rate_per_authorized_box_pct": round(
            100.0
            * residue_records
            / max(
                1,
                sum(
                    bool(box.get("safe_to_inpaint"))
                    for page in manifest.get("pages", [])
                    for box in (page.get("boxes") or [])
                ),
            ),
            4,
        ),
        "worst_pages": sorted(
            page_rows,
            key=lambda row: (
                int(row.get("outside_changed_pixels_exact", 0)),
                int(row.get("residue_records", 0)),
            ),
            reverse=True,
        )[:12],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chapter-id", default="5e8ce7e3")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmark-results/shadow-slave-ch1/inpaint-accuracy.json"),
    )
    parser.add_argument("--skip-synthetic", action="store_true")
    parser.add_argument("--skip-real", action="store_true")
    args = parser.parse_args()

    result = {
        "benchmark": "inpaint-accuracy-ground-truth-and-real-audit-v1",
        "chapter_id": args.chapter_id,
        "source_url": os.getenv("CHAPTER_URL"),
        "source_revision": os.getenv("GITHUB_SHA"),
        "model_files": sorted(path.name for path in (ROOT / "models").glob("*.onnx")),
    }
    if not args.skip_synthetic:
        result["synthetic"] = run_synthetic(
            Path("benchmark-results/shadow-slave-ch1/compact/inpaint-accuracy")
        )
    if not args.skip_real:
        result["real_chapter"] = _real_chapter_audit(args.chapter_id)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("INPAINT_ACCURACY=" + json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
