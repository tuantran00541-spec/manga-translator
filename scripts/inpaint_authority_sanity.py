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
    )
    _require(
        "app/detector/mask_builder.py",
        "def is_destructive_box_authorized(box: BubbleBox) -> bool:",
        "getattr(box, \"safe_to_inpaint\", False)",
        "or _rectangle_fallback_allowed(box)",
        "if not is_destructive_box_authorized(box):",
        "Skipping non-authorized destructive mask",
    )
    _require(
        "app/inpaint/lama_inpainter.py",
        "source_model=b.source_model",
        "mask_source=b.mask_source",
        "safe_to_inpaint=bool(b.safe_to_inpaint)",
        "ocr_eligible=bool(b.ocr_eligible)",
        "needs_review=bool(b.needs_review)",
    )
    print("Inpaint destructive-authority source contract OK")


if __name__ == "__main__":
    main()
