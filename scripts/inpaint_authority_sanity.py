from __future__ import annotations

from pathlib import Path


def _require(path: str, *needles: str) -> None:
    source = Path(path).read_text(encoding="utf-8")
    missing = [needle for needle in needles if needle not in source]
    if missing:
        joined = "\n  ".join(missing)
        raise SystemExit(f"{path} is missing inpaint-authority contract markers:\n  {joined}")


def main() -> None:
    _require(
        "app/pipeline.py",
        "Persisted review-only detector masks are evidence, not erase",
        "if overlap_context_only and not geometry_overridden:",
        "if not (safe_to_inpaint or geometry_overridden or explicit_manual):",
        "safe_to_inpaint=safe_to_inpaint",
        "if geometry_overridden or explicit_manual:",
        "box_object.allow_rectangle_fallback = True",
        "manual_lama_mask_posix: str | None = None",
        "(manual_mask_path, False)",
        "(manual_lama_mask_path, True)",
    )
    _require(
        "app/detector/mask_builder.py",
        "AUTO_DESTRUCTIVE_MASK_SOURCES = frozenset(",
        '"text_segmenter"',
        '"bubble_flat_contrast"',
        '"opencv_mser"',
        "def is_destructive_box_authorized(box: BubbleBox) -> bool:",
        'getattr(box, "safe_to_inpaint", False)',
        "or _rectangle_fallback_allowed(box)",
        "if not is_destructive_box_authorized(box):",
        "Skipping non-authorized destructive mask",
    )
    _require(
        "app/detector/combined_detector.py",
        'mask_source="bubble_flat_contrast"',
        "safe_to_inpaint=True",
    )
    _require(
        "app/inpaint/lama_inpainter.py",
        "source_model=b.source_model",
        "mask_source=b.mask_source",
        "safe_to_inpaint=bool(b.safe_to_inpaint)",
        "ocr_eligible=bool(b.ocr_eligible)",
        "needs_review=bool(b.needs_review)",
    )
    _require(
        "scripts/model_e2e_gate.py",
        "from app.detector.mask_builder import AUTO_DESTRUCTIVE_MASK_SOURCES",
        "mask_source not in AUTO_DESTRUCTIVE_MASK_SOURCES",
        "source_model=box.source_model",
        "mask_source=box.mask_source",
        "safe_to_inpaint=bool(box.safe_to_inpaint)",
        "ocr_eligible=bool(box.ocr_eligible)",
        "needs_review=bool(box.needs_review)",
        "if not args.allow_empty_cleanup:",
        '"model E2E produced no authorized cleanup mask pixels; inpaint path was not exercised"',
        '"cleanup_evidence": cleanup_evidence',
        'box_counts["ocr_eligible"] += int(bool(record.get("ocr_eligible")))',
    )
    print("Inpaint destructive-authority source contract OK")


if __name__ == "__main__":
    main()
